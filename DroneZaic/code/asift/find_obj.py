import numpy as np
import cv2 as cv

# This function initializes the feature detector and matcher
def init_feature(name):
    chunks = name.split('-')
    feature_name = chunks[0]
    matcher_name = chunks[1] if len(chunks) > 1 else 'bf'

    if feature_name == 'sift':
        detector = cv.SIFT_create()
    elif feature_name == 'surf':
        detector = cv.SURF_create()
    elif feature_name == 'orb':
        detector = cv.ORB_create(nfeatures=5000)
    elif feature_name == 'brisk':
        detector = cv.BRISK_create()
    else:
        return None, None

    norm = cv.NORM_L2 if feature_name == 'sift' or feature_name == 'surf' else cv.NORM_HAMMING
    if matcher_name == 'flann':
        if norm == cv.NORM_L2:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else: # For ORB/BRISK (NORM_HAMMING)
            FLANN_INDEX_LSH = 6
            index_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        search_params = dict(checks=50)
        matcher = cv.FlannBasedMatcher(index_params, search_params)
    elif matcher_name == 'bf':
        matcher = cv.BFMatcher(norm)
    else:
        return None, None

    return detector, matcher

# This function filters raw matches based on ratio test and extracts points/indices
def filter_matches(kp1, kp2, raw_matches, ratio=0.75):
    p1, p2 = [], [] # Points (x,y) for homography calculation
    kp_pairs_indices = [] # Indices of matched keypoints for explore_match (visualization)

    for m_pair in raw_matches: # raw_matches is a list of lists (e.g., [[dmatch1, dmatch2], ...])
        if len(m_pair) == 2 and m_pair[0].distance < m_pair[1].distance * ratio:
            # Good match found (passes ratio test)
            m = m_pair[0] # The best DMatch object
            p1.append( kp1[m.queryIdx].pt ) # Add the point from the keypoint object
            p2.append( kp2[m.trainIdx].pt ) # Add the point from the keypoint object
            kp_pairs_indices.append( (m.queryIdx, m.trainIdx) ) # Store (index_in_kp1_list, index_in_kp2_list)

    return np.float32(p1), np.float32(p2), kp_pairs_indices

# Updated explore_match: Stripped of all GUI elements for headless Docker execution.
# This function will now do nothing if called in this context.
# If you need match visualization, consider adding cv2.imwrite here.
def explore_match(win, img1_orig, img2_orig, kp1_all, kp2_all, kp_pairs_indices, status=None, H=None):
    pass # This function will do nothing in a headless environment.
    # If you want to save a visual representation of matches, you could add:
    # if kp_pairs_indices and img1_orig is not None and img2_orig is not None:
    #     # Create DMatch objects from indices for cv2.drawMatches
    #     dmatches = [cv.DMatch(queryIdx, trainIdx, 0) for queryIdx, trainIdx in kp_pairs_indices]
    #     img_matches = cv.drawMatches(img1_orig, kp1_all, img2_orig, kp2_all,
    #                                  dmatches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #     # Define a save path for the visualization image
    #     # For example: cv.imwrite("match_visualization.png", img_matches)
    #     # You would need to ensure 'win' (window name) is suitable for a filename or pass a save_path.