import cv2
import numpy as np


def find_correspondences(img1, img2, method="SIFT", dist="euclidean", threshold=100, ratio=0.75, fb_consistency=True, filter=True):
    if method == "SIFT":
        detector = cv2.SIFT_create()
    elif method == "ORB":
        detector = cv2.ORB_create()
    #    Fast brief
    elif method == "FAST":
        detector = cv2.FastFeatureDetector_create()
        
    # Detect keypoints and descriptors
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

    # Match descriptors
    matches = matcher(descriptors1, descriptors2, threshold, ratio,  dist)
    if fb_consistency:
        b_matches = matcher(descriptors2, descriptors1, threshold, ratio,  dist)
        matches = forward_backward_consisitency(matches, b_matches)

    if filter:
        try:
            matches = filter_matches(matches, keypoints1, keypoints2)
        except Exception as e:
            print(f"RANSAC failed. passing unfiltered correspondences. {e}")
            
    if len(matches) == 0:
        print("No matches found")
        return None, None, None

    queryIdxs = np.array([match.queryIdx for match in matches])
    trainIdxs = np.array([match.trainIdx for match in matches])
    distances = np.array([match.distance for match in matches])

    # dtype = [('queryIdx', int), ('trainIdx', int), ('distance', float)]
    matches_array = np.array(list(zip(queryIdxs, trainIdxs, distances)))    
    
    return keypoints1, keypoints2, matches_array[matches_array[:,2].argsort()]


def matcher(des1, des2, threshold=70, ratio=0.75, dist="euclidean"):

    matches = []

    for i, d1 in enumerate(des1):
        # Calculate Euclidean distances between d1 and all descriptors in des2
        distances = np.sqrt(((des2 - d1) ** 2).sum(axis=1))

        # Sort the distances and get the best two matches using numpy argsort or argmin
        best_match = np.argmin(distances)
        second_best_match = np.argsort(distances)[1]

        # Calculate the ratio between the best and second-best match
        # if distances[best_match] < threshold and distances[best_match] / distances[second_best_match] < ratio:
        #     matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=best_match, _distance=distances[best_match]))

        if distances[best_match] / distances[second_best_match] < ratio:
            matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=best_match, _distance=distances[best_match]))
        # Append the best match to the matches list
            
    return matches
            

def forward_backward_consisitency(f_matches, b_matches):
    # Convert backward matches to a lookup table for quick search
    backward_lookup = {m.queryIdx: m.trainIdx for m in b_matches}

    # Consistency check
    consistent_matches = []

    for fm in f_matches:
        if backward_lookup.get(fm.trainIdx) == fm.queryIdx:
            consistent_matches.append(fm)

    return consistent_matches

def filter_matches(matches, keypoints1, keypoints2):
    # Convert keypoints to numpy arrays
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Apply RANSAC to filter matches
    
    _, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

    return [match for match, m in zip(matches, mask.ravel().tolist()) if m == 1]