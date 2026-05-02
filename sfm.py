import argparse
import os 
import cv2
import numpy as np

from camera_model import parse_camera_model
from images import *
from features import extract_features, match_features, cross_check
from two_view_geometry import *
from visualize import *
from pnp import solve_pnp
from ba import bundle_adjustment


class SFM:
    def __init__(self, features, matches, camera_model, args):
        self.features = features
        self.matches = matches
        self.camera_model = camera_model
        self.reprojection_threshold = args.reprojection_threshold
        self.sequential = args.sequential

        #scene
        #poses of cameras (from world to camera)
        self.poses = [None for _ in range(len(self.features))]

        #set of 3d points
        self.points = []

        #observations of 3d points, observations[i] is a list of tuples (j, k) where j is the index of the image and k is the index of the keypoint in the image
        self.observations = []

        #observations lookup table, for each image and keypoint index, it stores the index of the 3d point
        self.observations_lookup = {}
    
        self.failed = False

        self.used_image_ids = []  # list of used images in 3d modelling
    
    def is_finished(self):
        return self.failed or all([pose is not None for pose in self.poses])

    def find_initial_pair(self):
        size = len(self.features)
        if self.sequential:
            max_i, max_j = 0, 1
            max_matches = 0
            for i in range(size - 1):
                j = i + 1
                if len(self.matches[i][j]) > max_matches:
                    max_matches = len(self.matches[i][j])
                    max_i, max_j = i, j

            print(f"Sequential initial pair: {max_i} {max_j}")
            return max_i, max_j
        else:  # brute-force
            max_number = 0
            max_i, max_j = 0, 1
            for i in range(size):
                for j in range(i + 1, size):
                    if len(self.matches[i][j]) > max_number:
                        max_number = len(self.matches[i][j])
                        max_i, max_j = i, j
            print(f"Brute-force initial pair: {max_i} {max_j}")
            return max_i, max_j

    def initialize(self):
        #find two images with most matches
        max_i, max_j = self.find_initial_pair()

        T, points, kp_id1, kp_id2 = two_view_geometry(self.features[max_i][0], self.features[max_j][0], self.matches[max_i][max_j], self.reprojection_threshold)

        self.poses[max_i] = np.eye(4)  # 128-arrange
        self.poses[max_j] = T

        self.points = points  # coordinates of confirmed 3d-points
        
        #add observations
        self.observations = []  # [[(image1_id, image1_kp_id), (image2_id, image2_kp_id)], ...]
        for i, j in zip(kp_id1, kp_id2):
            self.observations.append([(max_i, i), (max_j, j)])

        #add observations to lookup table
        for i, obs in enumerate(self.observations):
            for j in obs:
                assert(self.observations_lookup.get(j) is None)
                self.observations_lookup[j] = i

        # self.observations_lookup - (image_id, image_kp_id): номер в self.observations
        print(f"Intialized with {len(points)} points")
        self.used_image_ids.extend([max_i, max_j])

    def sort_views(self):
        #sort views by number of observations with 3d points
        views = []
        for i in range(len(self.features)):
            if i not in self.used_image_ids:
                custom_metric = sum([len(self.matches[i][im_id]) for im_id in self.used_image_ids])
                views.append([i, custom_metric])
        views.sort(key=lambda x: x[1], reverse=True)
        return views

    def add_view(self, i):
        """
        Adds a new camera view to the current SfM reconstruction.
        This function integrates image i into the existing 3D scene by:
        1. Finding correspondences between image i and already reconstructed views.
        2. Associating 2D keypoints from image i with existing 3D landmarks.
        3. Estimating the camera pose of image i using PnP.
        4. Updating observations (2D–3D correspondences).
        5. Triangulating new 3D points using newly established views.
        """
        print(f"Adding view {i}")
        #find 3d points in common with the best view
        matches = set()

        for j in self.used_image_ids:
            for new_im_descriptor_id, used_im_descriptor_id in self.matches[i][j]:
                if (j, used_im_descriptor_id) in self.observations_lookup:
                    landmark_id = self.observations_lookup[(j, used_im_descriptor_id)]
                    matches.add((landmark_id, new_im_descriptor_id))

        landmark_ids = [match[0] for match in matches]
        descriptor_ids = [match[1] for match in matches]

        if len(landmark_ids) < 10:
            print('Not enough points in common')
            return False
        
        print(f'Found {len(landmark_ids)} points in common')
        T, landmark_ids, descriptor_ids = solve_pnp(self.points, landmark_ids, self.features[i][0], descriptor_ids, self.reprojection_threshold)

        if T is None:
            print("Failed to add view")
            return False
        
        #remove descriptors that were matched to multiple landmarks
        #count how many times each descriptor was matched
        descriptor_count = {}
        for d in descriptor_ids:
            descriptor_count[d] = descriptor_count.get(d, 0) + 1
        #remove descriptors that were matched to multiple landmarks
        landmark_ids = [landmark_ids[i] for i, d in enumerate(descriptor_ids) if descriptor_count[d] == 1]
        descriptor_ids = [d for d in descriptor_ids if descriptor_count[d] == 1]

        self.poses[i] = T

        #add observations
        assert(self.check_observations_consisntency())
        for l, d in zip(landmark_ids, descriptor_ids):
            assert(len(self.observations) > l)
            self.observations[l].append((i, d))
            assert self.observations_lookup.get((i, d)) is None, (i, d)
            self.observations_lookup[(i, d)] = l
        
        assert(self.check_observations_consisntency())
        #triangulate new points
        for j in range(len(self.features)):
            if self.poses[j] is None or i == j:
                continue
            self.trinagulate_points(i, j)
            assert(self.check_observations_consisntency())
        print(f"Scene has {len(self.points)} points".format())
        print(f"Scene has {len([pose for pose in self.poses if pose is not None])} cameras")
        self.used_image_ids.append(i)
        return True
    
    def trinagulate_points(self, i, j):
        #collect matches that do not have 3d points
        matches = self.matches[i][j]
        if len(matches) < 50:
            return
        descritors_ids_i = []
        descritors_ids_j = []
        for match in matches:
            if self.observations_lookup.get((i, match[0])) is None and self.observations_lookup.get((j, match[1])) is None:
                descritors_ids_i.append(match[0])
                descritors_ids_j.append(match[1])

        descriptors_i = np.array([self.features[i][0][d] for d in descritors_ids_i])
        descriptors_j = np.array([self.features[j][0][d] for d in descritors_ids_j])

        if (descriptors_i.shape[0] < 10):
            return

        points, inliers = triangulate_points(descriptors_i, descriptors_j, self.poses[i], self.poses[j], self.reprojection_threshold)

        points = points[inliers]
        descritors_ids_i = np.array(descritors_ids_i)[inliers]
        descritors_ids_j = np.array(descritors_ids_j)[inliers]
        #if not enough points were triangulated probably something went wrong
        if (len(points) < 10):
            return
        print(f"Triangulated {len(points)} points from view {i} and {j}")

        #add points
        self.points = np.vstack((self.points, points))

        #add observations
        for d_i, d_j in zip(descritors_ids_i, descritors_ids_j):
            self.observations.append([(i, d_i), (j, d_j)])
            assert(self.observations_lookup.get((i, d_i)) is None)
            assert(self.observations_lookup.get((j, d_j)) is None)
            self.observations_lookup[(i, d_i)] = len(self.observations) - 1
            self.observations_lookup[(j, d_j)] = len(self.observations) - 1

    def add_next_view(self):
        views = self.sort_views()
        for view in views:
            if self.add_view(view[0]):
                return
        print("Failed to add a view")
        self.failed = True
        

    def visualize(self, filename):
        draw_scene(self.points, filename)

    def check_observations_consisntency(self):
        # check if observations are consistent with points and lookup
        for i, obs in enumerate(self.observations):
            for j in obs:
                if self.observations_lookup.get(j) != i:
                    print(f"Observation {j} does not match lookup table, expected {i} got {self.observations_lookup.get(j)}")

                    print(self.observations[i])
                    print(self.observations[self.observations_lookup.get(j)])
                    return False
                
        for i, obs in self.observations_lookup.items():
            if i not in self.observations[obs]:
                print(f"Lookup table {i} does not match observations, expected {obs} got {self.observations[obs]}")
                return False

        if len(self.points) != len(self.observations):
            print("Number of points and observations do not match")
            return False
        
        return True
    
    def bundle_adjustment(self):
        bundle_adjustment(self.poses, self.points, self.features, self.observations, self.reprojection_threshold)
        self.filter_outliers()
        assert(self.check_observations_consisntency())

    def filter_outliers(self):
        #filter out outliers by checking reprojection error
        observations_deleted = 0
        for i, obs in enumerate(self.observations):
            for j in obs:
                point = self.points[i]
                pose = self.poses[j[0]]
                point = pose @ np.hstack((point, 1))
                point = point / point[2]
                error = np.linalg.norm(point[:2] - self.features[j[0]][0][j[1]])
                if error > self.reprojection_threshold:
                    self.observations_lookup.pop(j)
                    self.observations[i].remove(j)
                    observations_deleted += 1
        print(f"Filtered {observations_deleted} observations")

        #remove points with less than 2 observations
        to_delete = set()
        for i, obs in enumerate(self.observations):
            if len(obs) < 2:
                to_delete.add(i)
        self.points = np.delete(self.points, list(to_delete), axis=0)
        self.observations = [obs for i, obs in enumerate(self.observations) if i not in to_delete]
        print(f"Filtered {len(to_delete)} points")

        #rebuild lookup table
        self.observations_lookup = {}
        for i, obs in enumerate(self.observations):
            for j in obs:
                self.observations_lookup[j] = i


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple sfm pipeline')
    parser.add_argument('--input', type=str, help='input folder')
    parser.add_argument('--video', type=str, default=None, help='input video file instead of image folder')
    parser.add_argument('--frame_stride', type=int, default=10, help='extract every N-th frame from video')
    parser.add_argument('--camera_model', type=str, help='camera model', default='SIMPLE_RADIAL 3072 2304 2559.68 1536 1152 -0.0204997')
    parser.add_argument('--max_views', type=int, help='maximum number of views', default=-1)
    parser.add_argument('--shuffle', action='store_true', help='shuffle images')
    parser.add_argument('--resize_factor', type=float, help='resize factor', default=1.0)
    parser.add_argument('--num_features', type=int, help='number of features', default=10000)
    parser.add_argument('--recalculate', action='store_true', help='recalculate features and matches')
    parser.add_argument('--reprojection_threshold', type=float, help='reprojection threshold', default=1e-3)
    parser.add_argument('--ba_frequency', type=int, help='bundle adjustment frequency', default=5)
    parser.add_argument('--vis_frequency', type=int, help='visualization frequency', default=5)
    parser.add_argument('--sequential', type=int, help='number of previous frames to match (0 = full brute-force)', default=0)
    args = parser.parse_args()

    # load camera model
    camera_model = parse_camera_model(args.camera_model)
    if args.resize_factor != 1.0:
        camera_model.resize(args.resize_factor)
    print(f'Camera model: {camera_model}')

    # force recalculation if needed
    if (not os.path.exists('features.npy')) or (not os.path.exists('matches.npy')):
        args.recalculate = True

    # dump arguments to file
    with open('args.txt', 'w') as f:
        f.write(str(args) + '\n')

    #load images
    if args.recalculate:
        # CASE 1: VIDEO INPUT
        if args.video is not None:
            print(f"Loading video: {args.video}")
            images = load_video(args.video, stride=args.frame_stride, resize_factor=args.resize_factor)
            # fake file names for compatibility
            files = [f"frame_{i}" for i in range(len(images))]

        # CASE 2: IMAGE FOLDER (original pipeline)
        else:
            files = get_image_files(args.input, args.max_views, args.shuffle)
            print(args.input)
            with open('image_names.txt', 'w') as f:
                for file in files:
                    f.write(file + '\n')
            images = load_images(files, args.resize_factor)
    else:
        files = [line.rstrip('\n') for line in open('image_names.txt')]

    #extract features
    if args.recalculate:
        features = extract_features(images, args.num_features)
        #unproject all features
        for i in range(len(features)):
            features[i][0] = camera_model.unproject(features[i][0])
        if len(features) > 0:
            np.save('features.npy', features, allow_pickle=True)
    else:
        print("Loading features from features.npy...")
        features = np.load('features.npy', allow_pickle=True)

    if args.recalculate:
        matches = match_features(features)
        matches = cross_check(matches)
        if len(matches) > 0:
            np.save('matches.npy', matches, allow_pickle=True)
    else:
        print("Loading matches from matches.npy...")
        matches = np.load('matches.npy', allow_pickle=True)

    sfm = SFM(features, matches, camera_model, args)
    sfm.initialize()
    assert(sfm.check_observations_consisntency())

    iteration = 0
    while not sfm.is_finished():
        sfm.add_next_view()
        assert(sfm.check_observations_consisntency())
        iteration += 1
        if iteration % args.ba_frequency == 0:
            sfm.bundle_adjustment()
        if args.vis_frequency > 0 and iteration % args.vis_frequency == 0:
            sfm.visualize(f"scene_{iteration}")

    sfm.visualize(f"scene_{iteration}")

    print("\n\nFINISHED\n\n")
