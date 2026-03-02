import zipfile
import os
from pathlib import Path

problem_root_dir = 'dmoj_problems'

os.chdir(problem_root_dir)

for problem_path in sorted(os.listdir('.')):
    if not os.path.isdir(problem_path):
        continue
    print(f"Processing {os.path.abspath(problem_path)}")
    problem_name = os.path.basename(problem_path)
    
    zips = [
        p for p in Path(problem_path).iterdir()
        if p.is_file() and p.suffix == ".zip" and not p.name.endswith(".old.zip")
    ]
    
    if len(zips) == 0:
        print(f"Error: no zip is found")
    
    for zip_path in zips:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(problem_path)
