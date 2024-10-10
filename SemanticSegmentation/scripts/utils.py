import os
import glob

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def list_files(directory, extensions):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f'*{ext}')))
    return files
