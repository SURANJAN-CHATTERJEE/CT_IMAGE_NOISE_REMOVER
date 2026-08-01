import logging
import pathlib
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Union

import numpy as np
import pydicom

logger = logging.getLogger(__name__)


class InputManagerError(Exception):
    """Base exception for input management errors."""
    pass


class InvalidDicomError(InputManagerError):
    """Raised when a DICOM file is corrupted or missing required tags."""
    pass


class SeriesConstructionError(InputManagerError):
    """Raised when slices cannot be formed into a valid 3D volume."""
    pass


class MetadataError(InputManagerError):
    """Raised when critical metadata is missing or inconsistent."""
    pass


class UnsupportedModalityError(InputManagerError):
    """Raised when a non-CT modality is encountered."""
    pass


@dataclass(frozen=True)
class InputVolume:
    """
    Immutable representation of a loaded medical imaging volume.
    """
    Volume: np.ndarray
    VolumeShape: Tuple[int, ...]
    VolumeDtype: str
    EstimatedMemoryMB: float
    VoxelSpacing: Tuple[float, float, float]
    SliceThickness: float
    ImageOrientation: Tuple[float, float, float, float, float, float]
    ImagePosition: Tuple[float, float, float]
    PixelSpacing: Tuple[float, float]
    Rows: int
    Columns: int
    NumberOfSlices: int
    SeriesInstanceUID: str
    StudyInstanceUID: str
    PatientPosition: str
    Metadata: Dict[str, Any]


class InputManager:
    """
    Safely imports medical CT imaging data into the framework.
    Serves as the primary entry point for raw imaging data.
    """
    
    SPATIAL_TOLERANCE: float = 1e-4
    SPACING_TOLERANCE: float = 1e-3

    def load(self, path: Union[str, pathlib.Path]) -> InputVolume:
        """
        Loads a medical volume from a single file or a directory.
        
        Args:
            path (Union[str, pathlib.Path]): Path to a DICOM file or directory containing a DICOM series.
            
        Returns:
            InputVolume: An immutable dataclass containing the volume and metadata.
            
        Raises:
            InputManagerError: For general loading failures.
            InvalidDicomError: For corrupted DICOM files.
            SeriesConstructionError: For invalid or mixed series.
            MetadataError: For missing or inconsistent metadata.
            UnsupportedModalityError: For unsupported imaging modalities.
        """
        start_time = time.time()
        input_path = pathlib.Path(path).resolve()
        
        if not input_path.exists():
            raise InputManagerError(f"Input path does not exist: {input_path}")
            
        logger.info(f"Initiating load sequence for: {input_path}")
        
        if input_path.is_file():
            volume, rejected_count = self._load_from_file(input_path)
        elif input_path.is_dir():
            volume, rejected_count = self._load_from_directory(input_path)
        else:
            raise InputManagerError(f"Path is neither a file nor a directory: {input_path}")
            
        total_time = time.time() - start_time
        
        logger.info("--- Loading Summary ---")
        logger.info(f"Patient ID: {volume.Metadata.get('PatientID', 'UNKNOWN')}")
        logger.info(f"Series UID: {volume.SeriesInstanceUID}")
        logger.info(f"Number of slices: {volume.NumberOfSlices}")
        logger.info(f"Volume Shape: {volume.VolumeShape}")
        logger.info(f"Volume Dtype: {volume.VolumeDtype}")
        logger.info(f"Estimated Memory: {volume.EstimatedMemoryMB:.2f} MB")
        logger.info(f"Slice Thickness: {volume.SliceThickness:.4f} mm")
        logger.info(f"Voxel Spacing: {volume.VoxelSpacing}")
        logger.info(f"Total Loading Time: {total_time:.2f} s")
        logger.info(f"Number of rejected files: {rejected_count}")
        logger.info("-----------------------")
        
        return volume

    def _load_from_file(self, file_path: pathlib.Path) -> Tuple[InputVolume, int]:
        """
        Loads a volume from a single file.
        
        Args:
            file_path (pathlib.Path): The path to the file.
            
        Returns:
            Tuple[InputVolume, int]: The constructed volume and the count of rejected files (0).
            
        Raises:
            NotImplementedError: If the format is not currently supported.
        """
        if file_path.suffix.lower() in [".nii", ".nii.gz", ".mha", ".mhd"]:
            raise NotImplementedError(f"Format {file_path.suffix} architecture ready but not implemented.")
            
        logger.debug("Loading single DICOM file.")
        dataset = self._read_dicom(file_path)
        return self._construct_volume([dataset]), 0

    def _load_from_directory(self, dir_path: pathlib.Path) -> Tuple[InputVolume, int]:
        """
        Loads a volume from a directory of DICOM files.
        
        Args:
            dir_path (pathlib.Path): The directory path.
            
        Returns:
            Tuple[InputVolume, int]: The constructed volume and the count of rejected files.
            
        Raises:
            InputManagerError: If the directory is empty.
            SeriesConstructionError: If no valid DICOM files are found.
        """
        files = [f for f in dir_path.iterdir() if f.is_file()]
        if not files:
            raise InputManagerError(f"Directory is empty: {dir_path}")
            
        logger.debug(f"Found {len(files)} files in directory. Reading headers.")
        
        datasets = []
        rejected_count = 0
        for file_path in files:
            try:
                datasets.append(self._read_dicom(file_path))
            except InvalidDicomError as error:
                rejected_count += 1
                logger.debug(f"Skipping file {file_path}: {error}")
                
        if not datasets:
            raise SeriesConstructionError(f"No valid DICOM files found in {dir_path}")
            
        return self._construct_volume(datasets), rejected_count

    def _read_dicom(self, file_path: pathlib.Path) -> pydicom.dataset.FileDataset:
        """
        Safely reads a DICOM file and validates its basic integrity.
        
        Args:
            file_path (pathlib.Path): The path to the DICOM file.
            
        Returns:
            pydicom.dataset.FileDataset: The parsed DICOM dataset.
            
        Raises:
            InvalidDicomError: If parsing fails or pixel data is missing.
            MetadataError: If critical spatial tags are missing.
        """
        try:
            dataset = pydicom.dcmread(str(file_path), force=True)
        except Exception as error:
            raise InvalidDicomError(f"Failed to parse DICOM {file_path}: {error}")
            
        if not hasattr(dataset, "pixel_array"):
            raise InvalidDicomError(f"DICOM file lacks pixel data: {file_path}")
            
        required_tags = [
            "ImagePositionPatient", 
            "ImageOrientationPatient", 
            "PixelSpacing", 
            "Rows", 
            "Columns",
            "SeriesInstanceUID",
            "StudyInstanceUID"
        ]
        
        missing = [tag for tag in required_tags if not hasattr(dataset, tag)]
        if missing:
            raise MetadataError(f"DICOM file {file_path} missing required metadata: {missing}")
            
        return dataset

    def _construct_volume(self, datasets: List[pydicom.dataset.FileDataset]) -> InputVolume:
        """
        Builds a 3D volume and constructs the immutable output object.
        
        Args:
            datasets (List[pydicom.dataset.FileDataset]): A list of valid DICOM datasets.
            
        Returns:
            InputVolume: The complete, sorted, and validated volume object.
        """
        self._validate_single_series(datasets)
        
        sorted_datasets = self._sort_slices(datasets)
        
        self._validate_consistency(sorted_datasets)
        
        ref = sorted_datasets[0]
        
        modality = str(getattr(ref, "Modality", "UNKNOWN")).upper()
        if modality != "CT":
            raise UnsupportedModalityError(f"Unsupported modality: {modality}. Only CT is currently supported.")
        
        logger.debug("Stacking slices into 3D volume.")
        arrays = [ds.pixel_array.astype(np.float32) for ds in sorted_datasets]
        volume = np.stack(arrays, axis=0)
        
        # IMPROVEMENT 1: Make volume read-only
        volume.flags.writeable = False
        
        # IMPROVEMENT 2: Volume details
        vol_shape = volume.shape
        vol_dtype = str(volume.dtype)
        est_mem_mb = float(volume.nbytes / (1024 * 1024))
        
        img_ori = (
            float(ref.ImageOrientationPatient[0]),
            float(ref.ImageOrientationPatient[1]),
            float(ref.ImageOrientationPatient[2]),
            float(ref.ImageOrientationPatient[3]),
            float(ref.ImageOrientationPatient[4]),
            float(ref.ImageOrientationPatient[5])
        )
        
        img_pos = (
            float(ref.ImagePositionPatient[0]),
            float(ref.ImagePositionPatient[1]),
            float(ref.ImagePositionPatient[2])
        )
        
        pixel_spacing = (float(ref.PixelSpacing[0]), float(ref.PixelSpacing[1]))
        
        num_slices = len(sorted_datasets)
        if num_slices > 1:
            z_spacing = self._compute_slice_spacing(sorted_datasets)
        else:
            z_spacing = float(getattr(ref, "SliceThickness", 1.0))
            
        voxel_spacing = (z_spacing, pixel_spacing[0], pixel_spacing[1])
        slice_thickness = float(getattr(ref, "SliceThickness", z_spacing))
        
        metadata = self._extract_supplementary_metadata(ref)
        metadata["VolumeShape"] = vol_shape
        metadata["VolumeDtype"] = vol_dtype
        metadata["EstimatedMemoryMB"] = est_mem_mb
        
        logger.info("Volume constructed successfully.")
        
        return InputVolume(
            Volume=volume,
            VolumeShape=vol_shape,
            VolumeDtype=vol_dtype,
            EstimatedMemoryMB=est_mem_mb,
            VoxelSpacing=voxel_spacing,
            SliceThickness=slice_thickness,
            ImageOrientation=img_ori,
            ImagePosition=img_pos,
            PixelSpacing=pixel_spacing,
            Rows=int(ref.Rows),
            Columns=int(ref.Columns),
            NumberOfSlices=num_slices,
            SeriesInstanceUID=str(ref.SeriesInstanceUID),
            StudyInstanceUID=str(ref.StudyInstanceUID),
            PatientPosition=str(getattr(ref, "PatientPosition", "UNKNOWN")),
            Metadata=metadata
        )

    def _validate_single_series(self, datasets: List[pydicom.dataset.FileDataset]) -> None:
        """Ensures all slices belong to the exact same series."""
        series_uids = {str(ds.SeriesInstanceUID) for ds in datasets}
        if len(series_uids) > 1:
            raise SeriesConstructionError(f"Mixed series detected: found {len(series_uids)} unique SeriesInstanceUIDs.")

    def _sort_slices(self, datasets: List[pydicom.dataset.FileDataset]) -> List[pydicom.dataset.FileDataset]:
        """
        Sorts datasets spatially based on ImagePositionPatient and ImageOrientationPatient,
        and detects duplicate slices.
        """
        ref = datasets[0]
        iop = np.array(ref.ImageOrientationPatient, dtype=np.float64)
        row_cosine = iop[:3]
        col_cosine = iop[3:]
        normal = np.cross(row_cosine, col_cosine)
        
        def calculate_z(ds: pydicom.dataset.FileDataset) -> float:
            ipp = np.array(ds.ImagePositionPatient, dtype=np.float64)
            return float(np.dot(ipp, normal))
            
        sorted_datasets = sorted(datasets, key=calculate_z)
        
        positions = set()
        for ds in sorted_datasets:
            pos_tuple = (
                round(float(ds.ImagePositionPatient[0]), 4),
                round(float(ds.ImagePositionPatient[1]), 4),
                round(float(ds.ImagePositionPatient[2]), 4)
            )
            if pos_tuple in positions:
                raise SeriesConstructionError(
                    f"Duplicate slice detected. Multiple slices exist at spatial position: {pos_tuple}."
                )
            positions.add(pos_tuple)
            
        return sorted_datasets

    def _compute_slice_spacing(self, sorted_datasets: List[pydicom.dataset.FileDataset]) -> float:
        """
        Computes the spacing between slices and validates uniformity.
        """
        if len(sorted_datasets) < 2:
            raise SeriesConstructionError("Cannot compute slice spacing for a single slice.")
            
        ref = sorted_datasets[0]
        iop = np.array(ref.ImageOrientationPatient, dtype=np.float64)
        normal = np.cross(iop[:3], iop[3:])
        
        spacings = []
        for i in range(1, len(sorted_datasets)):
            ipp1 = np.array(sorted_datasets[i-1].ImagePositionPatient, dtype=np.float64)
            ipp2 = np.array(sorted_datasets[i].ImagePositionPatient, dtype=np.float64)
            dist = np.dot(ipp2 - ipp1, normal)
            spacings.append(abs(dist))
            
        mean_spacing = float(np.mean(spacings))
        std_spacing = float(np.std(spacings))
        
        if std_spacing > self.SPACING_TOLERANCE:
            raise SeriesConstructionError(f"Inconsistent slice spacing detected. Std dev: {std_spacing:.6f}")
            
        if mean_spacing < self.SPACING_TOLERANCE:
            raise SeriesConstructionError("Slices appear to have zero spacing (overlapping slices).")
            
        return mean_spacing

    def _validate_consistency(self, sorted_datasets: List[pydicom.dataset.FileDataset]) -> None:
        """
        Validates that all slices share consistent spatial properties.
        """
        ref = sorted_datasets[0]
        ref_rows = ref.Rows
        ref_cols = ref.Columns
        ref_ps = tuple(float(x) for x in ref.PixelSpacing)
        ref_ori = tuple(float(x) for x in ref.ImageOrientationPatient)
        
        for i, ds in enumerate(sorted_datasets):
            if ds.Rows != ref_rows or ds.Columns != ref_cols:
                raise SeriesConstructionError(f"Inconsistent image dimensions at slice {i}.")
                
            ps = tuple(float(x) for x in ds.PixelSpacing)
            if abs(ps[0] - ref_ps[0]) > self.SPATIAL_TOLERANCE or abs(ps[1] - ref_ps[1]) > self.SPATIAL_TOLERANCE:
                raise SeriesConstructionError(f"Inconsistent PixelSpacing at slice {i}.")
                
            ori = tuple(float(x) for x in ds.ImageOrientationPatient)
            for ref_val, val in zip(ref_ori, ori):
                if abs(ref_val - val) > self.SPATIAL_TOLERANCE:
                    raise SeriesConstructionError(f"Inconsistent ImageOrientationPatient at slice {i}.")

    def _extract_supplementary_metadata(self, ref: pydicom.dataset.FileDataset) -> Dict[str, Any]:
        """
        Extracts non-critical but useful metadata for the pipeline.
        """
        return {
            "Modality": str(getattr(ref, "Modality", "UNKNOWN")),
            "Manufacturer": str(getattr(ref, "Manufacturer", "UNKNOWN")),
            "RescaleIntercept": float(getattr(ref, "RescaleIntercept", 0.0)),
            "RescaleSlope": float(getattr(ref, "RescaleSlope", 1.0)),
            "PatientID": str(getattr(ref, "PatientID", "UNKNOWN")),
            "KVP": float(getattr(ref, "KVP", 0.0)),
            "XRayTubeCurrent": float(getattr(ref, "XRayTubeCurrent", 0.0))
        }
