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


def match_features(features, sequential=0):
    matches = []
    n = len(features)

    matcher = cv2.FlannBasedMatcher_create()
    for i in tqdm(range(n), desc='Matching features'):
        matches.append([])

        # decide which images to match with
        if sequential == 0:
            js = range(n)  # brute-force
        else:
            js = range(max(0, i - sequential), i)  # window backward only

        for j in js:
            if i == j:
                matches[i].append(None)
                continue

            m = matcher.knnMatch(features[i][1], features[j][1], k=2)

            good = []
            for pair in m:
                if len(pair) < 2:
                    continue

                m1, m2 = pair
                if m1.distance < 0.7 * m2.distance:
                    good.append((m1.queryIdx, m1.trainIdx))

            matches[i].append(good)

        # fill missing forward entries (important for indexing consistency)
        if sequential != 0:
            for _ in range(n - len(matches[i])):
                matches[i].append(None)

    return np.array(matches, dtype=object)

def cross_check(matches):
    #cross check matches, only keep matches that are mutual
    for i in range(len(matches)):
        for j in range(len(matches[i])):
            if matches[i][j] is None:
                continue
            matches[i][j] = [match for match in matches[i][j] if match in [(match[1], match[0]) for match in matches[j][i]]]
    return matches
