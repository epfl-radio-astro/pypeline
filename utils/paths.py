import os
from pathlib import Path

def get_datasets_path():
    datasets_dir = Path.joinpath(Path(__file__).absolute().parents[1], "datasets")
    if not os.path.isdir(datasets_dir):
        print(f"Fatal  : datasets_dir {datasets_dir} not existing!")
    return datasets_dir
