import numpy as np
import cv2 as cv

# This function initializes the feature detector and matcher
def init_feature(name: str):
    """
    Initializes a feature detector and matcher based on the provided name.

    Args:
        name (str): A string specifying the feature detector and matcher,
                    e.g., 'sift-flann', 'orb-bf'.

    Returns:
        tuple: A tuple containing the initialized detector and matcher objects,
               or (None, None) if the name is not recognized.
    """
    chunks = name.split('-')
    feature_name = chunks[0]
    matcher_name = chunks[1] if len(chunks) > 1 else 'bf'

    if feature_name == 'sift':
        detector = cv.SIFT_create()
    elif feature_name == 'surf':
        detector = cv.SURF_create()
    elif feature_name == 'orb':
        # Default nfeatures is 5000, which you've already tuned
        detector = cv.ORB_create(nfeatures=10000) 
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
# UPDATED: Made the ratio test slightly looser to allow more matches to pass
def filter_matches(kp1: list, kp2: list, raw_matches: list, ratio: float = 0.70): # Changed ratio from 0.5 to 0.6
    """
    Filters raw DMatch objects based on Lowe's ratio test and extracts
    corresponding keypoints and their indices.

    Args:
        kp1 (list): List of keypoints from the query image.
        kp2 (list): List of keypoints from the train image.
        raw_matches (list): A list of lists of DMatch objects, typically
                            from `matcher.knnMatch(..., k=2)`.
        ratio (float): The ratio threshold for the ratio test (m.distance < n.distance * ratio).

    Returns:
        tuple: A tuple containing:
            - p1 (np.ndarray): Nx2 array of (x,y) coordinates of matched keypoints from kp1.
            - p2 (np.ndarray): Nx2 array of (x,y) coordinates of matched keypoints from kp2.
            - kp_pairs_indices (list): List of (queryIdx, trainIdx) tuples for visualization.
    """
    p1, p2 = [], [] # Points (x,y) for homography calculation
    kp_pairs_indices = [] # Indices of matched keypoints for explore_match (visualization)

    for m_pair in raw_matches: # raw_matches is a list of lists (e.g., [[dmatch1, dmatch2], ...])
        # Ensure m_pair has at least 2 matches before accessing m_pair[0] and m_pair[1]
        if len(m_pair) >= 2 and m_pair[0].distance < m_pair[1].distance * ratio:
            # Good match found (passes ratio test)
            m = m_pair[0] # The best DMatch object
            p1.append( kp1[m.queryIdx].pt ) # Add the point from the keypoint object
            p2.append( kp2[m.trainIdx].pt ) # Add the point from the keypoint object
            kp_pairs_indices.append( (m.queryIdx, m.trainIdx) ) # Store (index_in_kp1_list, index_in_kp2_list)

    return np.float32(p1), np.float32(p2), kp_pairs_indices

# Updated explore_match: Stripped of all GUI elements for headless Docker execution.
# This function will now do nothing if called in this context.
# If you need match visualization, consider adding cv2.imwrite here.
def explore_match(win: str, img1_orig: np.ndarray, img2_orig: np.ndarray, 
                  kp1_all: list, kp2_all: list, kp_pairs_indices: list, 
                  status: np.ndarray = None, H: np.ndarray = None):
    """
    Placeholder function for match visualization in a headless environment.
    If visualization is needed, cv2.imwrite calls could be added here
    to save images of matches.

    Args:
        win (str): Window name (unused in headless).
        img1_orig (np.ndarray): Original query image.
        img2_orig (np.ndarray): Original train image.
        kp1_all (list): All keypoints from query image.
        kp2_all (list): All keypoints from train image.
        kp_pairs_indices (list): Indices of matched keypoint pairs.
        status (np.ndarray, optional): Mask indicating inliers/outliers from homography. Defaults to None.
        H (np.ndarray, optional): Computed homography matrix. Defaults to None.
    """
    pass # This function will do nothing in a headless environment.
    # If you want to save a visual representation of matches, you could add:
    # if kp_pairs_indices and img1_orig is not None and img2_orig is not None:
    #     # Create DMatch objects from indices for cv2.drawMatches
    #     # If status is provided, draw only inliers
    #     if status is not None:
    #         dmatches_to_draw = [cv.DMatch(queryIdx, trainIdx, 0) for (queryIdx, trainIdx), flag in zip(kp_pairs_indices, status.ravel()) if flag]
    #     else:
    #         dmatches_to_draw = [cv.DMatch(queryIdx, trainIdx, 0) for queryIdx, trainIdx in kp_pairs_indices]
    #
    #     if dmatches_to_draw:
    #         img_matches = cv.drawMatches(img1_orig, kp1_all, img2_orig, kp2_all,
    #                                      dmatches_to_draw, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #         # Define a save path for the visualization image (e.g., using 'win' or a custom path)
    #         # cv.imwrite(f"match_visualization_{win}.png", img_matches)