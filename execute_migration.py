import os
import shutil

base_dir = r"e:\INTERNSHIPS\001 = INP V1"
src_dir = os.path.join(base_dir, "src")

# Target folders
folders = [
    "core",
    "pipeline/preprocessing/dicom",
    "pipeline/preprocessing/hu",
    "pipeline/preprocessing/windowing",
    "pipeline/preprocessing/normalization",
    "pipeline/preprocessing/validation",
    "pipeline/characterization/generators",
    "pipeline/characterization/detectors",
    "pipeline/characterization/descriptors",
    "pipeline/characterization/reports",
    "pipeline/routing",
    "pipeline/fusion/strategies",
    "pipeline/fusion/confidence",
    "pipeline/restoration/refinement",
    "pipeline/restoration/enhancement",
    "pipeline/restoration/quality_control",
    "pipeline/evaluation/metrics",
    "pipeline/evaluation/benchmarking",
    "pipeline/evaluation/reports",
    "pipeline/explainability/gradcam",
    "pipeline/explainability/shap",
    "pipeline/explainability/visualization",
    "pipeline/explainability/reports",
    "experts/base",
    "experts/statistical/gaussian",
    "experts/statistical/poisson",
    "experts/statistical/speckle",
    "experts/statistical/salt_pepper",
    "experts/statistical/mixed",
    "experts/statistical/rician",
    "experts/statistical/gamma",
    "experts/artifacts/metal",
    "experts/artifacts/beam_hardening",
    "experts/artifacts/motion",
    "experts/artifacts/ring",
    "experts/artifacts/scatter",
    "experts/artifacts/photon_starvation",
    "experts/anatomy/brain",
    "experts/anatomy/lung",
    "experts/anatomy/abdomen",
    "experts/anatomy/cardiac",
    "experts/future",
    "interfaces",
    "io/loaders",
    "io/writers",
    "io/exporters",
    "training/datasets",
    "training/dataloaders",
    "training/losses",
    "training/optimizers",
    "training/schedulers",
    "training/callbacks",
    "training/trainers",
    "training/checkpoints",
    "utils"
]

inits = [
    "core",
    "pipeline",
    "pipeline/characterization",
    "pipeline/routing",
    "pipeline/fusion",
    "pipeline/restoration",
    "pipeline/evaluation",
    "pipeline/explainability",
    "experts",
    "interfaces",
    "io",
    "training"
]

for f in folders:
    os.makedirs(os.path.join(src_dir, f), exist_ok=True)

for i in inits:
    init_file = os.path.join(src_dir, i, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("")

placeholders = [
    "core/config.py",
    "core/constants.py",
    "core/exceptions.py",
    "core/logger.py",
    "core/registry.py",
    "core/metadata.py",
    "interfaces/expert.py",
    "interfaces/fusion.py",
    "interfaces/router.py",
    "interfaces/evaluator.py",
    "interfaces/reporter.py",
    "pipeline/routing/routing_rules.py",
    "pipeline/routing/routing_utils.py",
    "pipeline/fusion/fusion_engine.py"
]

for p in placeholders:
    p_path = os.path.join(src_dir, p)
    if not os.path.exists(p_path):
        with open(p_path, "w") as f:
            f.write("")

moves = {
    "preprocessing/dicom_loader.py": "pipeline/preprocessing/dicom/dicom_loader.py",
    "preprocessing/hu_converter.py": "pipeline/preprocessing/hu/hu_converter.py",
    "preprocessing/windowing.py": "pipeline/preprocessing/windowing/windowing.py",
    "preprocessing/normalization.py": "pipeline/preprocessing/normalization/normalization.py",
    "preprocessing/dataset_indexer.py": "training/datasets/dataset_indexer.py",
    "noise_generation/noise_generator.py": "pipeline/characterization/generators/noise_generator.py",
    "noise_characterization/noise_characterizer.py": "pipeline/characterization/detectors/noise_characterizer.py",
    "routing/adaptive_router.py": "pipeline/routing/adaptive_router.py"
}

for old, new in moves.items():
    old_path = os.path.join(src_dir, old)
    new_path = os.path.join(src_dir, new)
    if os.path.exists(old_path):
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.move(old_path, new_path)

import_map = {
    "from src.preprocessing.dicom_loader": "from src.pipeline.preprocessing.dicom.dicom_loader",
    "from src.preprocessing.hu_converter": "from src.pipeline.preprocessing.hu.hu_converter",
    "from src.preprocessing.windowing": "from src.pipeline.preprocessing.windowing.windowing",
    "from src.preprocessing.normalization": "from src.pipeline.preprocessing.normalization.normalization",
    "from src.preprocessing.dataset_indexer": "from src.training.datasets.dataset_indexer",
    "from src.noise_generation.noise_generator": "from src.pipeline.characterization.generators.noise_generator",
    "from src.noise_characterization.noise_characterizer": "from src.pipeline.characterization.detectors.noise_characterizer",
    "from src.routing.adaptive_router": "from src.pipeline.routing.adaptive_router",
    
    "import src.preprocessing.dicom_loader": "import src.pipeline.preprocessing.dicom.dicom_loader",
    "import src.preprocessing.hu_converter": "import src.pipeline.preprocessing.hu.hu_converter",
    "import src.preprocessing.windowing": "import src.pipeline.preprocessing.windowing.windowing",
    "import src.preprocessing.normalization": "import src.pipeline.preprocessing.normalization.normalization",
    "import src.preprocessing.dataset_indexer": "import src.training.datasets.dataset_indexer",
    "import src.noise_generation.noise_generator": "import src.pipeline.characterization.generators.noise_generator",
    "import src.noise_characterization.noise_characterizer": "import src.pipeline.characterization.detectors.noise_characterizer",
    "import src.routing.adaptive_router": "import src.pipeline.routing.adaptive_router"
}

for new in moves.values():
    new_path = os.path.join(src_dir, new)
    if os.path.exists(new_path):
        with open(new_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        for old_import, new_import in import_map.items():
            content = content.replace(old_import, new_import)
            
        with open(new_path, "w", encoding="utf-8") as f:
            f.write(content)

print("Migration completed.")
