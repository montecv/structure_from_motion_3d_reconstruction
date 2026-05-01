import tqdm
import os
import random
from tqdm import tqdm
import cv2

image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}


def get_image_files(dir, max_views=-1, shuffle=False):
    files = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.splitext(f)[1].lower() in image_extensions]
    if shuffle:
        random.shuffle(files)
    if max_views > 0:
        files = files[:max_views]
    return files


def load_images(files, resize_factor=1.0):
    images = []
    for file in tqdm(files, desc='Loading images'):
        images.append(cv2.imread(file))
    if resize_factor != 1.0:
        images = [cv2.resize(image, (int(image.shape[1] * resize_factor), int(image.shape[0] * resize_factor))) for image in images]
    return images
