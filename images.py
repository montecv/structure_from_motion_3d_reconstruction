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


def load_video(video_path, stride=5, resize_factor=1.0):
    VIDEO_SAVE_DIR = 'video_images'

    cap = cv2.VideoCapture(video_path)
    os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)

    frames = []
    frame_id = 0
    saved_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % stride == 0:
            if resize_factor != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
            frames.append(frame)
            frame_path = os.path.join(VIDEO_SAVE_DIR, f"frame_{saved_id}.png")
            cv2.imwrite(frame_path, frame)
            saved_id += 1

        frame_id += 1

    cap.release()
    print(f"Extracted {len(frames)} frames from video")
    return frames