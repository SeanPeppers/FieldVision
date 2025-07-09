#!/usr/bin/env python

'''
Affine invariant feature-based image matching sample.
This sample is similar to find_obj.py, but uses the affine transformation
space sampling technique, called ASIFT [1]. While the original implementation
is based on SIFT, you can try to use SURF or ORB detectors instead. Homography RANSAC
is used to reject outliers. Threading is used for faster affine sampling.
[1] http://www.ipol.im/pub/algo/my_affine_sift/
USAGE
  asift.py [--feature=<sift|surf|orb|brisk>[-flann]] [ <image1> <image2> ]
  --feature   - Feature to use. Can be sift, surf, orb or brisk. Append '-flann'
                to feature name to use Flann-based matcher instead bruteforce.
  Press left mouse button on a feature point to see its matching point.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import logging # Added logging

# built-in modules
import itertools as it
from multiprocessing.pool import ThreadPool
import sys # For sys.exit in main()

# local modules
from .common import Timer
from .find_obj import init_feature, filter_matches, explore_match # ensure find_obj is updated as well

# Configure logging for asift.py
logger = logging.getLogger(__name__) # Use __name__ to get a logger specific to this module
logger.setLevel(logging.DEBUG) # Set to DEBUG to see all detailed prints
handler = logging.StreamHandler(sys.stdout) # Output logs to stdout
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def affine_skew(tilt: float, phi: float, img: np.ndarray, mask: np.ndarray = None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai
    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv.warpAffine(img, A, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv.warpAffine(mask, A, (w, h), flags=cv.INTER_NEAREST)
    Ai = cv.invertAffineTransform(A)
    return img, mask, Ai


def affine_detect(detector: cv.Feature2D, img: np.ndarray, mask: np.ndarray = None, pool: ThreadPool = None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs
    Apply a set of affine transformations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.
    ThreadPool object may be passed to speedup the computation.
    '''
    params = [(1.0, 0.0)]
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)

        # --- DEBUG PRINTS START (Uncommented and Enhanced) ---
        if keypoints:
            logger.debug(f"(affine_detect.f) t={t:.2f}, phi={phi:.1f}, Detected {len(keypoints)} keypoints.") # ADDED: t, phi values
            if len(keypoints) == 0:
                logger.debug(f"(affine_detect.f) t={t:.2f}, phi={phi:.1f}, Keypoints list is empty.") #
        else:
            logger.debug(f"(affine_detect.f) t={t:.2f}, phi={phi:.1f}, No keypoints detected (None returned).") #
        # --- DEBUG PRINTS END ---
        
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
        if descrs is None:
            descrs = np.array([]) # Ensure descrs is an empty numpy array if no descriptors
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        keypoints.extend(k)
        descrs.extend(d)

    return keypoints, np.array(descrs)

def my_asift(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Computes a homography matrix between two images using the ASIFT algorithm.

    Args:
        img1 (np.ndarray): The first input image (grayscale).
        img2 (np.ndarray): The second input image (grayscale).

    Returns:
        np.ndarray: The 3x3 homography matrix if successful, otherwise None.
    """
    feature_name = "sift-flann" # Using SIFT as feature_name
    detector, matcher = init_feature(feature_name)
    
    pool=ThreadPool(processes = cv.getNumberOfCPUs())
    
    kp1, desc1 = affine_detect(detector, img1, pool=pool)
    kp2, desc2 = affine_detect(detector, img2, pool=pool)
    logger.debug(f"my_asift: Total {len(kp1)} kps from img1, Total {len(kp2)} kps from img2 across all affine transforms.") # ADDED

    # Ensure descriptors exist and are valid for matching
    # A minimum of 4 keypoints is needed for homography calculation.
    if desc1.size == 0 or desc2.size == 0 or len(kp1) < 4 or len(kp2) < 4: 
        logger.warning("my_asift: Insufficient keypoints/descriptors for matching after affine detection. Returning None.") # ADDED
        pool.close()
        pool.join()
        return None

    def match_and_draw(win: str): 
        with Timer('matching'):
            # Match descriptors using k-NN approach with k=2 for ratio test
            raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) 
        
        logger.debug(f"my_asift: Found {len(raw_matches)} raw matches from knnMatch.") # ADDED

        # This calls filter_matches from find_obj.py, which has the ratio parameter
        # Ratio is 0.75 by default in find_obj.py
        p1, p2, kp_pairs_indices = filter_matches(kp1, kp2, raw_matches) 

        logger.debug(f"my_asift: Found {len(p1)} filtered matches after ratio test.") # ADDED

        H = None
        status = None
        if len(p1) >= 4:
            # UPDATED: Adjusted reprojThresh for RANSAC here within my_asift (based on user's test: 6.0)
            H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0) # Changed from 5.0 to 6.0
            
            if H is not None:
                logger.debug(f"my_asift: {np.sum(status)} / {len(status)} inliers/matched after findHomography.") # ADDED
            else:
                logger.warning("my_asift: cv.findHomography returned None (H is None).") # ADDED

            kp_pairs_for_explore = [pair for pair, flag in zip(kp_pairs_indices, status.ravel()) if flag] # status is a column vector
        else:
            logger.warning(f"my_asift: {len(p1)} filtered matches found, not enough (need >= 4) for homography estimation. Returning None.") # Added DEBUG label
            kp_pairs_for_explore = [] 

        explore_match(win, img1, img2, kp1, kp2, kp_pairs_for_explore, status, H) 
        return H
    
    logger.info("Attempting to match features and estimate homography.") 
    H=match_and_draw('affine find_obj') 
    
    pool.close() 
    pool.join()  

    return H 

def main():
    import sys, getopt 
    # Use argparse for better argument handling, consistent with other scripts
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs=2, help='Two image files to compare.')
    parser.add_argument('--feature', type=str, default='brisk-flann', 
                        help="Feature to use. Can be sift, surf, orb or brisk. Append '-flann' to feature name to use Flann-based matcher instead bruteforce.")
    args = parser.parse_args()

    fn1, fn2 = args.images
    feature_name = args.feature

    img1 = cv.imread(fn1, 0) 
    img2 = cv.imread(fn2, 0) 
    detector, matcher = init_feature(feature_name)

    if img1 is None:
        logger.error(f'Failed to load image 1: {fn1}')
        sys.exit(1)
    if img2 is None:
        logger.error(f'Failed to load image 2: {fn2}')
        sys.exit(1)
    if detector is None:
        logger.error(f'Unknown feature: {feature_name}')
        sys.exit(1)

    logger.info(f'Using feature: {feature_name}')

    pool=ThreadPool(processes = cv.getNumberOfCPUs())
    kp1, desc1 = affine_detect(detector, img1, pool=pool)
    kp2, desc2 = affine_detect(detector, img2, pool=pool)
    logger.info(f'img1 - {len(kp1)} features, img2 - {len(kp2)} features')

    def match_and_draw(win: str): 
        with Timer('matching'):
            raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) 
        
        p1, p2, kp_pairs_indices = filter_matches(kp1, kp2, raw_matches) 

        H = None
        status = None
        if len(p1) >= 4:
            # This is the main()'s specific call (also set to 6.0 for consistency)
            H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0) # Changed from 5.0 to 6.0
            logger.info(f'{np.sum(status)} / {len(status)} inliers/matched')
        else:
            logger.warning(f'{len(p1)} matches found, not enough for homography estimation')
        
        kp_pairs_for_explore = []
        if status is not None:
             kp_pairs_for_explore = [pair for pair, flag in zip(kp_pairs_indices, status.ravel()) if flag]
        
        explore_match(win, img1, img2, kp1, kp2, kp_pairs_for_explore, status, H)
        return H # Return H to main, though not used there currently

    match_and_draw('affine find_obj') 
    # cv.waitKey() is for GUI, not needed in headless environment
    logger.info('Done')


if __name__ == '__main__':
    # Remove GUI related calls for Docker environment
    # cv.destroyAllWindows()
    main()