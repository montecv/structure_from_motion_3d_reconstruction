from tqdm import tqdm
import cv2
import numpy as np


def extract_features(images, num_features):
    features = []
    for image in tqdm(images, desc='Extracting features'):
        #detect keypoints
        detector = cv2.SIFT_create(nfeatures = num_features)
        keypoints, descriptors = detector.detectAndCompute(image, None)
        #transform keypoints to numpy array
        keypoints = np.array([keypoint.pt for keypoint in keypoints])
        features.append((keypoints, descriptors))
    return np.array(features, dtype=object)


def match_features(features):
    #match features using flann, match each image to every other image
    matches = []
    for i in tqdm(range(len(features)), desc='Matching features'):
        matches.append([])
        for j in range(len(features)):
            if i == j:
                matches[i].append(None)
                continue
            #create flann matcher
            matcher = cv2.FlannBasedMatcher_create()
            #find matches
            m = matcher.knnMatch(features[i][1], features[j][1], k=2)
            good = []
            for m1, m2 in m:
                if m1.distance < 0.7 * m2.distance:
                    #convert match to tupple
                    good.append((m1.queryIdx, m1.trainIdx))
            matches[i].append(good)

    return np.array(matches, dtype=object)


def cross_check(matches):
    #cross check matches, only keep matches that are mutual
    for i in range(len(matches)):
        for j in range(len(matches[i])):
            if matches[i][j] is None:
                continue
            matches[i][j] = [match for match in matches[i][j] if match in [(match[1], match[0]) for match in matches[j][i]]]
    return matches
