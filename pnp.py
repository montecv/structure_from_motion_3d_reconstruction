import numpy as np
import cv2


def solve_pnp(points, landmark_ids, keypoints, keypoints_ids, reprojection_threshold=5e-3):
    '''
    solve pnp problem using 3d points and 2d keypoints
    '''

    #get 3d points and 2d keypoints
    points = points[landmark_ids]
    keypoints = np.array([keypoints[keypoints_id] for keypoints_id in keypoints_ids])

    #solve pnp
    _, rvec, tvec, inliers = cv2.solvePnPRansac(points, keypoints, np.eye(3), np.zeros(5), confidence=0.99, reprojectionError=reprojection_threshold, iterationsCount=1000)
    if inliers is None or len(inliers) < 100:
        print('No inliers found')
        return None, None, None
    print('Found {} inliers'.format(len(inliers)))
    R, _ = cv2.Rodrigues(rvec)
    T = np.hstack((R, tvec))

    #filter outliers
    landmark_ids = np.array(landmark_ids)[inliers.flatten()]
    keypoints_ids = np.array(keypoints_ids)[inliers.flatten()]

    return T, landmark_ids, keypoints_ids