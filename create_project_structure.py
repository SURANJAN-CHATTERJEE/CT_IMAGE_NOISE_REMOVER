import pathlib

def create_structure():
    base_dir = pathlib.Path(__file__).parent.resolve()
    
    folders = [
        "datasets",
        "checkpoints",
        "configs",
        "logs",
        "outputs",
        "reports",
        "experiments",
        "notebooks",
        "docs",
        "src/preprocessing",
        "src/noise_generation",
        "src/characterization",
        "src/routing",
        "src/models",
        "src/fusion",
        "src/restoration",
        "src/evaluation",
        "src/utils"
    ]
    
    for folder in folders:
        folder_path = base_dir / folder
        if folder_path.exists():
            print(f"Already existed: {folder}")
        else:
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {folder}")

if __name__ == "__main__":
    create_structure()
