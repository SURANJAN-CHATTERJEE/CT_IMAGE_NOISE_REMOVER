import concurrent.futures
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Any, Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import pydicom

logger = logging.getLogger(__name__)


class DatasetError(Exception):
    """Base exception for dataset management errors."""
    pass


class InvalidDatasetError(DatasetError):
    """Raised when the dataset structure or files are fundamentally invalid."""
    pass


class DuplicateSeriesError(DatasetError):
    """Raised when a SeriesInstanceUID is assigned to multiple StudyInstanceUIDs."""
    pass


class DuplicateStudyError(DatasetError):
    """Raised when a StudyInstanceUID is assigned to multiple PatientIDs."""
    pass


@dataclass(frozen=True)
class DatasetIndex:
    """
    Immutable representation of an indexed dataset.
    """
    DatasetRoot: str
    PatientTable: pd.DataFrame
    StudyTable: pd.DataFrame
    SeriesTable: pd.DataFrame
    FileTable: pd.DataFrame
    Statistics: Mapping[str, Any]
    Manifest: Mapping[str, Any]
    TrainPatients: Tuple[str, ...]
    ValidationPatients: Tuple[str, ...]
    TestPatients: Tuple[str, ...]
    DuplicateSeries: Tuple[str, ...]
    DuplicateStudies: Tuple[str, ...]
    ProcessingTime: float
    LayerVersion: str

    def export_csv(self, output_dir: str) -> None:
        """Exports the dataset tables to CSV files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self.PatientTable.to_csv(out_path / "patients.csv", index=False)
        self.StudyTable.to_csv(out_path / "studies.csv", index=False)
        self.SeriesTable.to_csv(out_path / "series.csv", index=False)
        self.FileTable.to_csv(out_path / "files.csv", index=False)

    def export_json(self, output_file: str) -> None:
        """Exports the dataset manifest to a JSON file."""
        manifest_dict = dict(self.Manifest)
        with open(output_file, 'w') as f:
            json.dump(manifest_dict, f, indent=4)


class CTDatasetIndexer:
    """
    Engine for indexing and managing CT DICOM datasets.
    This is the exclusive layer responsible for dataset metadata management.
    """
    
    VERSION: str = "1.0.1"

    def build(
        self,
        dataset_root: str,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> DatasetIndex:
        """
        Builds the dataset index by scanning and extracting DICOM metadata.
        
        Args:
            dataset_root: Path to the dataset directory.
            train_ratio: Ratio of patients for training.
            validation_ratio: Ratio of patients for validation.
            test_ratio: Ratio of patients for testing.
            random_seed: Seed for random splitting.
            
        Returns:
            DatasetIndex: An immutable dataclass containing the dataset metadata and splits.
            
        Raises:
            InvalidDatasetError: If dataset path is invalid or files are missing.
            DuplicateSeriesError: If hierarchy constraints are violated.
            DuplicateStudyError: If hierarchy constraints are violated.
            DatasetError: If split validation fails.
        """
        start_time = time.time()
        
        if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
            raise InvalidDatasetError("Train, validation, and test ratios must sum to 1.0")
            
        root_path = Path(dataset_root).resolve()
        if not root_path.exists():
            raise InvalidDatasetError(f"Dataset root does not exist: {root_path}")
        if not root_path.is_dir():
            raise InvalidDatasetError(f"Dataset root is not a directory: {root_path}")
        if not os.access(root_path, os.R_OK):
            raise InvalidDatasetError(f"Dataset root is not readable: {root_path}")
            
        all_files = [p for p in root_path.rglob("*") if p.is_file()]
        if not all_files:
            raise InvalidDatasetError(f"No files found in {root_path}")
            
        total_files = len(all_files)
        logger.info(f"Scanning {total_files} files in {root_path} in parallel...")
        
        results = []
        rejected_files = 0
        missing_metadata = 0
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(self._read_dicom_header, f): f for f in all_files}
            for future in concurrent.futures.as_completed(future_to_file):
                res = future.result()
                if res["status"] == "ok":
                    results.append(res["data"])
                elif res["status"] == "invalid_dicom":
                    rejected_files += 1
                elif res["status"] == "missing_metadata":
                    missing_metadata += 1
                    
        valid_files = len(results)
        if not results:
            raise InvalidDatasetError("No valid DICOM files with required metadata found.")
            
        results.sort(key=lambda x: (
            x.get('PatientID', ''),
            x.get('StudyInstanceUID', ''),
            x.get('SeriesInstanceUID', ''),
            x.get('SOPInstanceUID', '')
        ))
            
        df = pd.DataFrame(results)
        
        sop_counts = df['SOPInstanceUID'].value_counts()
        duplicate_sops = sop_counts[sop_counts > 1].index.tolist()
        num_duplicates = len(duplicate_sops)
        if duplicate_sops:
            logger.warning(f"Found {num_duplicates} duplicate SOPInstanceUIDs. Retaining first occurrences.")
            df = df.drop_duplicates(subset=['SOPInstanceUID'], keep='first')
            
        series_study_mapping = df.groupby('SeriesInstanceUID')['StudyInstanceUID'].nunique()
        invalid_series = series_study_mapping[series_study_mapping > 1].index.tolist()
        if invalid_series:
            raise DuplicateSeriesError(f"SeriesInstanceUIDs mapped to multiple studies: {invalid_series}")
            
        study_patient_mapping = df.groupby('StudyInstanceUID')['PatientID'].nunique()
        invalid_studies = study_patient_mapping[study_patient_mapping > 1].index.tolist()
        if invalid_studies:
            raise DuplicateStudyError(f"StudyInstanceUIDs mapped to multiple patients: {invalid_studies}")
            
        file_table = df.copy()
        
        series_table = df.groupby('SeriesInstanceUID').agg({
            'PatientID': 'first',
            'StudyInstanceUID': 'first',
            'Modality': 'first',
            'Manufacturer': 'first',
            'BodyPartExamined': 'first',
            'SOPInstanceUID': 'count'
        }).reset_index().rename(columns={'SOPInstanceUID': 'SliceCount'})
        
        study_table = series_table.groupby('StudyInstanceUID').agg({
            'PatientID': 'first',
            'SeriesInstanceUID': 'count'
        }).reset_index().rename(columns={'SeriesInstanceUID': 'SeriesCount'})
        
        patient_table = study_table.groupby('PatientID').agg({
            'StudyInstanceUID': 'count'
        }).reset_index().rename(columns={'StudyInstanceUID': 'StudyCount'})
        
        patients = sorted(patient_table['PatientID'].unique().tolist())
        rng = random.Random(random_seed)
        rng.shuffle(patients)
        
        num_patients = len(patients)
        train_end = int(num_patients * train_ratio)
        val_end = train_end + int(num_patients * validation_ratio)
        
        train_patients = tuple(patients[:train_end])
        val_patients = tuple(patients[train_end:val_end])
        test_patients = tuple(patients[val_end:])
        
        train_set = set(train_patients)
        val_set = set(val_patients)
        test_set = set(test_patients)
        
        if train_set.intersection(val_set) or train_set.intersection(test_set) or val_set.intersection(test_set):
            raise DatasetError("Data leakage detected: patient overlap across splits.")
        
        thickness_dist = df['SliceThickness'].dropna().value_counts().to_dict()
        spacing_dist = df['PixelSpacing'].dropna().value_counts().to_dict()
        spacing_dist_str = {str(k): v for k, v in spacing_dist.items()}
        
        statistics = {
            "TotalFilesScanned": total_files,
            "ValidDicomFiles": valid_files,
            "RejectedInvalidDicom": rejected_files,
            "RejectedMissingMetadata": missing_metadata,
            "DuplicateSOPInstances": num_duplicates,
            "Patients": int(patient_table.shape[0]),
            "Studies": int(study_table.shape[0]),
            "Series": int(series_table.shape[0]),
            "Slices": int(file_table.shape[0]),
            "Manufacturers": file_table['Manufacturer'].dropna().unique().tolist(),
            "Modalities": file_table['Modality'].dropna().unique().tolist(),
            "BodyPartsExamined": file_table['BodyPartExamined'].dropna().unique().tolist(),
            "SliceThicknessDistribution": thickness_dist,
            "PixelSpacingDistribution": spacing_dist_str
        }
        
        manifest = {
            "DatasetRoot": str(root_path),
            "SplitConfiguration": {
                "TrainRatio": train_ratio,
                "ValidationRatio": validation_ratio,
                "TestRatio": test_ratio,
                "RandomSeed": random_seed
            },
            "Statistics": statistics,
            "Splits": {
                "Train": list(train_patients),
                "Validation": list(val_patients),
                "Test": list(test_patients)
            },
            "IndexingTimestamp": datetime.now().isoformat()
        }
        
        processing_time = time.time() - start_time
        speed = total_files / processing_time if processing_time > 0 else 0
        
        frozen_statistics = MappingProxyType(statistics)
        frozen_manifest = MappingProxyType(manifest)
        
        logger.info(f"Total scanned files: {total_files}")
        logger.info(f"Valid DICOM files: {valid_files}")
        logger.info(f"Rejected files: {rejected_files + missing_metadata}")
        logger.info(f"Processing speed: {speed:.2f} files/sec")
        logger.info(f"Dataset indexed in {processing_time:.2f} s")
        logger.info(f"Patients: {statistics['Patients']}, Studies: {statistics['Studies']}, Series: {statistics['Series']}")
        
        return DatasetIndex(
            DatasetRoot=str(root_path),
            PatientTable=patient_table,
            StudyTable=study_table,
            SeriesTable=series_table,
            FileTable=file_table,
            Statistics=frozen_statistics,
            Manifest=frozen_manifest,
            TrainPatients=train_patients,
            ValidationPatients=val_patients,
            TestPatients=test_patients,
            DuplicateSeries=tuple(invalid_series),
            DuplicateStudies=tuple(invalid_studies),
            ProcessingTime=processing_time,
            LayerVersion=self.VERSION
        )

    def _read_dicom_header(self, filepath: Path) -> dict:
        """
        Reads only the header of a DICOM file to extract metadata.
        """
        try:
            ds = pydicom.dcmread(str(filepath), stop_before_pixels=True)
        except Exception:
            return {"status": "invalid_dicom", "file": str(filepath)}
            
        patient_id = getattr(ds, "PatientID", None)
        study_uid = getattr(ds, "StudyInstanceUID", None)
        series_uid = getattr(ds, "SeriesInstanceUID", None)
        sop_uid = getattr(ds, "SOPInstanceUID", None)
        
        if not all([patient_id, study_uid, series_uid, sop_uid]):
            return {"status": "missing_metadata", "file": str(filepath)}
            
        modality = getattr(ds, "Modality", "UNKNOWN")
        manufacturer = getattr(ds, "Manufacturer", "UNKNOWN")
        body_part = getattr(ds, "BodyPartExamined", "UNKNOWN")
        kernel = getattr(ds, "ConvolutionKernel", "UNKNOWN")
        patient_position = getattr(ds, "PatientPosition", "UNKNOWN")
        
        slice_thickness = getattr(ds, "SliceThickness", None)
        pixel_spacing = getattr(ds, "PixelSpacing", None)
        kvp = getattr(ds, "KVP", None)
        recon_diameter = getattr(ds, "ReconstructionDiameter", None)
        rows = getattr(ds, "Rows", None)
        columns = getattr(ds, "Columns", None)
        
        if isinstance(kernel, list):
            kernel = "_".join([str(k) for k in kernel])
            
        data = {
            "FilePath": str(filepath),
            "PatientID": str(patient_id),
            "StudyInstanceUID": str(study_uid),
            "SeriesInstanceUID": str(series_uid),
            "SOPInstanceUID": str(sop_uid),
            "Modality": str(modality),
            "Manufacturer": str(manufacturer),
            "BodyPartExamined": str(body_part),
            "ConvolutionKernel": str(kernel),
            "PatientPosition": str(patient_position),
            "SliceThickness": float(slice_thickness) if slice_thickness is not None else np.nan,
            "PixelSpacing": tuple(float(x) for x in pixel_spacing) if pixel_spacing else None,
            "KVP": float(kvp) if kvp is not None else np.nan,
            "ReconstructionDiameter": float(recon_diameter) if recon_diameter is not None else np.nan,
            "Rows": int(rows) if rows is not None else -1,
            "Columns": int(columns) if columns is not None else -1
        }
        
        return {"status": "ok", "data": data}
