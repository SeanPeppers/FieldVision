import cv2
import numpy as np
import sys # Added for debug prints

def surf(img1, img2):
    """
    Computes homography between two grayscale images using SIFT features.
    Named 'surf' for compatibility with existing calls in surf_homography_estimation.py.
    """
    # Initialize SIFT detector
    # ORB was previously used with: orb = cv2.ORB_create(nfeatures=5000)
    sift = cv2.SIFT_create() # <<<<< CHANGE HERE: Swapping from ORB to SIFT

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Ensure descriptors are not None before proceeding
    if des1 is None or des2 is None:
        print("DEBUG: No descriptors found for one or both images. Cannot compute homography.", file=sys.stderr)
        return None # No descriptors found, cannot compute homography

    # BFMatcher with default parameters for SIFT (NORM_L2)
    # Flann-based Matcher is often preferred for SIFT/SURF due to speed on large datasets
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params) # <<<<< CHANGE HERE: Using Flann-based Matcher for SIFT

    # Match descriptors using k-NN (k=2 for ratio test)
    matches = flann.knnMatch(des1, des2, k=2) # <<<<< CHANGE HERE: Using Flann matcher

    # Apply Lowe's ratio test
    good_matches = []
    # Common ratio test threshold for SIFT/SURF is often 0.7 or 0.75
    # You might need to tune this as well if stitching issues persist.
    ratio_thresh = 0.75 # <<<<< You can tune this
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    if len(good_matches) > 4: # Need at least 4 points to find a homography
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find Homography using RANSAC
        # Reprojection error threshold. Experiment with values like 1.0, 2.0, 3.0.
        # A lower value means stricter inlier criteria, potentially more accurate H.
        reproj_thresh = 5.0 # <<<<< CURRENTLY 2.0, you can tune this further
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)

        if H is not None:
            return H
        else:
            print("DEBUG: Homography matrix is None (too many outliers/no solution).", file=sys.stderr)
            return None # Homography not found (e.g., too many outliers)
    else:
        print("DEBUG: Not enough good matches ({} < 4) to compute homography.".format(len(good_matches)), file=sys.stderr)
        return None # Not enough good matches to compute homography