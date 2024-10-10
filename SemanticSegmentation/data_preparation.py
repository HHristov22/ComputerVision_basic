# data_preparation.py

import os
import urllib.request
import tarfile
from scripts.utils import create_dir, list_files

def download_dataset(data_dir):
    url = 'http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz'
    annotations_url = 'http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz'

    images_path = os.path.join(data_dir, 'images')
    annotations_path = os.path.join(data_dir, 'annotations')

    if not os.path.exists(images_path):
        print("Изтегляне на изображения...")
        urllib.request.urlretrieve(url, 'images.tar.gz')
        tar = tarfile.open('images.tar.gz')
        tar.extractall(path=data_dir)
        tar.close()
        os.remove('images.tar.gz')

    if not os.path.exists(annotations_path):
        print("Изтегляне на анотациите...")
        urllib.request.urlretrieve(annotations_url, 'annotations.tar.gz')
        tar = tarfile.open('annotations.tar.gz')
        tar.extractall(path=data_dir)
        tar.close()
        os.remove('annotations.tar.gz')

def prepare_data(data_dir):
    images_path = os.path.join(data_dir, 'images')
    masks_path = os.path.join(data_dir, 'annotations', 'trimaps')

    image_files = list_files(images_path, ['.jpg'])
    mask_files = list_files(masks_path, ['.png'])

    print(f"Брой изображения: {len(image_files)}")
    print(f"Брой маски: {len(mask_files)}")

if __name__ == "__main__":
    data_dir = './data'
    create_dir(data_dir)
    download_dataset(data_dir)
    prepare_data(data_dir)
