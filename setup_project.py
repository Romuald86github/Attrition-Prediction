import os

# Define the project structure
project_structure = [
    "mage-infrastructure",
    "src/data",
    "src/features",
    "src/models",
    "src/app",
    "src/monitoring",
    "data/raw",
    "data/processed",
    "models",
    "terraform",
    "ui_scripts",
    ".github/workflows",
]

# Create directories and subdirectories
for directory in project_structure:
    os.makedirs(directory, exist_ok=True)
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            with open(os.path.join(dir_path, ".gitkeep"), "w"):
                pass

# Create empty files
empty_files = [
    "mage-infrastructure/main.tf",
    "mage-infrastructure/variables.tf",
    "mage-infrastructure/outputs.tf",
    "src/data/data_cleaning.py",
    "src/features/feature_selection.py",
    "src/features/preprocessing.py",
    "src/models/training.py",
    "src/app/app.py",
    "src/app/requirements.txt",
    "src/app/Dockerfile",
    "src/monitoring/github_actions.yml",
    "src/monitoring/evidently_monitoring.py",
    "src/pipeline.yaml",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "models/.gitkeep",
    "terraform/main.tf",
    "terraform/variables.tf",
    "terraform/outputs.tf",
    "ui_scripts/interface.py",
    "README.md",
    "requirements.txt",
    ".github/workflows/ci_cd.yml",
]

for file_path in empty_files:
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, "w"):
        pass