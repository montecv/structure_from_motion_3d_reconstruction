import cv2
import numpy as np


def triangulate_points(points1, points2, T1, T2, reprojection_threshold=5e-3):
    #triangulate using linear algebra
    #construct linear system
    points = np.zeros((points1.shape[0], 4))
    A_r = T2[:2, :]
    A_l = T1[:2, :]
    A = np.vstack((A_l, A_r))
    B = np.array([T1[2, :],
        T1[2, :],
        T2[2, :],
        T2[2, :]])
    for i in range(len(points1)):
        x1 = points1[i]
        x2 = points2[i]
        d = np.array([x1[0], x1[1], x2[0], x2[1]])
        S = A - np.diag(d) @ B
        #find nullspace
        _, _, V = np.linalg.svd(S)
        x = V[-1]
        x = x / x[3]
        points[i] = x

    #check reprojection error
    points1_transformed = T1 @ points.T
    points1_reprojected = points1_transformed / points1_transformed[2, :]
    error1 = np.linalg.norm(points1_reprojected[:2, :].T - points1, axis=1)

    points2_transformed = T2 @ points.T
    points2_reprojected = points2_transformed / points2_transformed[2, :]
    error2 = np.linalg.norm(points2_reprojected[:2, :].T - points2, axis=1)

    inliers = np.logical_and(error1 < reprojection_threshold, error2 < reprojection_threshold)

    #check positive depth
    inliers = np.logical_and(inliers, points1_transformed[2, :] > 0, points2_transformed[2, :] > 0)

    #check points far from camera
    inliers = np.logical_and(inliers, points1_transformed[2, :] < 50, points2_transformed[2, :] < 50)
    return points[:, :3], inliers > 0


def two_view_geometry(keypoints1, keypoints2, matches, reprojection_threshold):
    points1 = np.array([keypoints1[match[0]] for match in matches])
    points2 = np.array([keypoints2[match[1]] for match in matches])
    keypoints_ids_1 = np.array([match[0] for match in matches])
    keypoints_ids_2 = np.array([match[1] for match in matches])

    E, mask = cv2.findEssentialMat(points1, points2, threshold=reprojection_threshold)
    points1 = points1[mask.flatten() > 0]
    points2 = points2[mask.flatten() > 0]
    keypoints_ids_1 = keypoints_ids_1[mask.flatten() > 0]
    keypoints_ids_2 = keypoints_ids_2[mask.flatten() > 0]


    _, R, t, mask = cv2.recoverPose(E, points1, points2)
    T_c1_c0 = np.hstack((R, t))
    # T_c0_c1 = np.hstack((R.T, -R.T @ t))
    points1 = points1[mask.flatten() > 0]
    points2 = points2[mask.flatten() > 0]
    keypoints_ids_1 = keypoints_ids_1[mask.flatten() > 0]
    keypoints_ids_2 = keypoints_ids_2[mask.flatten() > 0]
    # T = T_c1_c0

    points, inliers = triangulate_points(points1, points2, np.eye(4), T_c1_c0, reprojection_threshold)

    points = points[inliers]
    keypoints_ids_1 = keypoints_ids_1[inliers]
    keypoints_ids_2 = keypoints_ids_2[inliers]
    return T_c1_c0, points, keypoints_ids_1, keypoints_ids_2