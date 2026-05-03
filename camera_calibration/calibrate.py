import argparse
import cv2
import numpy as np
import os


def calibrate_camera(chessboard_size, square_size, images_path):
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    images = [i for i in os.listdir(images_path) if i.endswith('.jpg')]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for im_name in images:
        print(im_name)
        path = os.path.join(images_path, im_name)
        img = cv2.imread(path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners2)

            os.makedirs(f'{images_path}/corners', exist_ok=True)
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imwrite(f'{images_path}/corners/{im_name}', img)

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return K, dist, gray.shape[::-1]

def rescale_intrinsics(K, dist, old_size, new_size):
    old_w, old_h = old_size
    new_w, new_h = new_size

    scale_x = new_w / old_w
    scale_y = new_h / old_h

    fx = K[0, 0] * scale_x
    fy = K[1, 1] * scale_y
    cx = K[0, 2] * scale_x
    cy = K[1, 2] * scale_y

    K_new = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    return K_new, dist

def parse_args():
    parser = argparse.ArgumentParser(description="Camera calibration tool")

    parser.add_argument("--images_path", type=str, required=True, help="Path to images folder")
    parser.add_argument("--cols", type=int, required=True, help="Chessboard columns (inner corners)")
    parser.add_argument("--rows", type=int, required=True, help="Chessboard rows (inner corners)")
    parser.add_argument("--square_size", type=float, required=True, help="Square size (meters)")

    parser.add_argument("--video_width", type=int, default=None)
    parser.add_argument("--video_height", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    chessboard_size = (args.cols, args.rows)
    square_size = args.square_size
    images_path = args.images_path

    print("\n CALIBRATION FROM IMAGES ")

    camera_matrix, dist_coeffs, image_size = calibrate_camera(chessboard_size, square_size, images_path)

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    f = (fx + fy) / 2
    k1 = dist_coeffs[0][0]

    width, height = image_size

    print("Image resolution:", width, "x", height)
    print("focal_length =", f)
    print("principal_point =", [cx, cy])
    print("distortion_coefficients =", k1)

    print("\n=== COPY-PASTE FOR RadialCameraModel ===")
    print(f"RadialCameraModel(np.array([{width}, {height}]), {f}, np.array([{cx}, {cy}]), {k1})")

    if args.video_width and args.video_height:
        video_size = (args.video_width, args.video_height)
        print("\n RESCALED FOR VIDEO ")

        photo_size = (width, height)

        K_new, dist_new = rescale_intrinsics(camera_matrix, dist_coeffs, photo_size, video_size)

        fx_new = K_new[0, 0]
        fy_new = K_new[1, 1]
        cx_new = K_new[0, 2]
        cy_new = K_new[1, 2]

        f_new = (fx_new + fy_new) / 2
        k1_new = dist_new[0][0]

        print("Video resolution:", video_size[0], "x", video_size[1])
        print("focal_length =", f_new)
        print("principal_point =", [cx_new, cy_new])
        print("distortion_coefficients =", k1_new)

        print("\n=== COPY-PASTE FOR RadialCameraModel ===")
        print("\n# FROM VIDEO")
        print(f"RadialCameraModel(np.array([{video_size[0]}, {video_size[1]}]), {f_new}, np.array([{cx_new}, {cy_new}]), {k1_new})")


