import json
import numpy as np
import logging
import os
from typing import Any, List

logger = logging.getLogger(__name__)

class OutputExporter:
    """Medical Output Formats Exporter."""
    
    @staticmethod
    def export(result: Any, output_dir: str, formats: List[str]) -> None:
        os.makedirs(output_dir, exist_ok=True)
        volume = result.FinalVolume if hasattr(result, "FinalVolume") else result
        metadata = result.Metadata if hasattr(result, "Metadata") else {}
        
        safe_metadata = {k: str(v) for k, v in dict(metadata).items()}
        formats_lower = [f.lower() for f in formats]
        
        if "npy" in formats_lower or "numpy" in formats_lower:
            np.save(os.path.join(output_dir, "output.npy"), volume)
            logger.info("Exported NumPy volume.")
            
        if "json" in formats_lower:
            with open(os.path.join(output_dir, "report.json"), "w") as f:
                json.dump(safe_metadata, f, indent=4)
            logger.info("Exported JSON report.")
                
        if "png" in formats_lower:
            try:
                from PIL import Image
                norm_vol = np.clip(volume * 255.0, 0, 255).astype(np.uint8)
                img = Image.fromarray(norm_vol[norm_vol.shape[0]//2]) # Export middle slice
                img.save(os.path.join(output_dir, "output.png"))
                logger.info("Exported PNG slice.")
            except ImportError:
                logger.error("Pillow not installed, skipping PNG export.")
            
        if "tiff" in formats_lower:
            try:
                import tifffile
                tiff_vol = np.clip(volume * 65535.0, 0, 65535).astype(np.uint16)
                tifffile.imwrite(os.path.join(output_dir, "output.tiff"), tiff_vol)
                logger.info("Exported 16-bit TIFF volume.")
            except ImportError:
                logger.error("tifffile not installed, skipping TIFF export.")
            
        if "dcm" in formats_lower or "dicom" in formats_lower:
            try:
                import pydicom
                from pydicom.dataset import FileDataset, FileMetaDataset
                file_meta = FileMetaDataset()
                file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
                file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
                ds = FileDataset(os.path.join(output_dir, "output.dcm"), {}, file_meta=file_meta, preamble=b"\0" * 128)
                ds.PatientName = "Anonymous"
                ds.PixelData = volume.tobytes()
                ds.save_as(os.path.join(output_dir, "output.dcm"))
                logger.info("Exported DICOM volume.")
            except ImportError:
                logger.error("pydicom not installed, skipping DICOM export.")
            
        if "nii.gz" in formats_lower or "nifti" in formats_lower:
            logger.warning("NIfTI export is a future placeholder and not currently supported.")
