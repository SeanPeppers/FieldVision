import cv2
import numpy as np

def surf(img1, img2):
    """
    Computes homography between two grayscale images using ORB features.
    Named 'surf' for compatibility with existing calls in surf_homography_estimation.py.
    """
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=5000) # Increased nfeatures for potentially better match density

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Ensure descriptors are not None before proceeding
    if des1 is None or des2 is None:
        # print("DEBUG: No descriptors found for one or both images. Cannot compute homography.", file=sys.stderr) # Can add debug if needed
        return None # No descriptors found, cannot compute homography

    # BFMatcher with NORM_HAMMING for ORB binary descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Match descriptors using k-NN (k=2 for ratio test)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance: # Common ratio test threshold
            good_matches.append(m)

    if len(good_matches) > 4: # Need at least 4 points to find a homography
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find Homography using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # RANSAC method with reprojection error 5.0

        if H is not None:
            return H
        else:
            # print("DEBUG: Homography matrix is None (too many outliers/no solution).", file=sys.stderr) # Can add debug if needed
            return None # Homography not found (e.g., too many outliers)
    else:
        # print("DEBUG: Not enough good matches ({} < 4) to compute homography.".format(len(good_matches)), file=sys.stderr) # Can add debug if needed
        return None # Not enough good matches to compute homography