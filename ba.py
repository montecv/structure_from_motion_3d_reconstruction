import gtsam
import numpy as np


def bundle_adjustment(poses, points, features, observations, reprojection_threshold=5e-3):
    #bundle adjustment using gtsam
    #create factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    fix_first = False
    #add camera poses
    for i, pose in enumerate(poses):
        if pose is None:
            continue
        pose = gtsam.Pose3(gtsam.Rot3(pose[:3, :3]), gtsam.Point3(pose[:3, 3]))
        initial_estimate.insert(gtsam.symbol('p', i), pose.inverse())
        if fix_first:
            graph.add(gtsam.PriorFactorPose3(gtsam.symbol(9, i), gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]))))
            fix_first = False

    #add 3d points
    for i, point in enumerate(points):
        point = gtsam.Point3(point)
        initial_estimate.insert(gtsam.symbol('l', i), point)

    #add reprojection factors
    noise_model = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
    huber = gtsam.noiseModel.mEstimator.Huber(reprojection_threshold)
    noise_model = gtsam.noiseModel.Robust(huber, noise_model)

    fx, fy, s, u0, v0 = 1., 1., 0.0, 0., 0.  # where fx and fy are focal lengths, s is skew, u0 and v0 are principal point coordinates

    for point_id, obs in enumerate(observations):
        for (image_id, kp_id) in obs:
            observed_feature = features[image_id][0][kp_id]
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                gtsam.Point2(observed_feature[:2]),
                noise_model,
                gtsam.symbol('p', image_id),  # Image symbol
                gtsam.symbol('l', point_id),  # Landmark symbol
                gtsam.Cal3_S2(fx, fy, s, u0, v0)  # Camera model
            ))

    #optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    params.setlambdaInitial(1e-3)
    params.setMaxIterations(10)
    params.setlambdaUpperBound(1e7)
    params.setlambdaLowerBound(1e-7)
    params.setRelativeErrorTol(1e-5)

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    #update poses and points
    for i in range(len(poses)):
        if poses[i] is not None:
            poses[i] = np.eye(4) 
            pose = result.atPose3(gtsam.symbol('p', i)).inverse()
            poses[i][:3, :3] = pose.rotation().matrix()
            poses[i][:3, 3] = pose.translation()

    for i in range(len(points)):
        point = result.atPoint3((gtsam.symbol('l', i)))
        points[i] = point
