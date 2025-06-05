#!/usr/bin/env python

import cv2
import numpy as np

def surf(gray1, gray2):
   #convert images to grayscale
   #gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
   #gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

   #initialize SURF detector
   surf_detector = cv2.ORB_create() # Renamed 'surf' variable to 'surf_detector' to avoid conflict with the function name

   #detect keypoints and compute descriptors
   keypoints1, descriptors1 = surf_detector.detectAndCompute(gray1, None)
   keypoints2, descriptors2 = surf_detector.detectAndCompute(gray2, None)

   # Check if descriptors are found
   if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) == 0:
       print "No descriptors found for one or both images." # Python 2.7 print statement
       return None # Return None if no descriptors for matching

   #initialize matcher
   matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
   #matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

   #match keypoints
   matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

   #apply Lowe's ratio test to filter matches
   good_matches = []
   for m, n in matches:
       if m.distance < 0.75 * n.distance:
           good_matches.append(m)

   #estimate homography
   if len(good_matches) > 4:
       src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
       dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
       
       #use RANSAC to estimate homography
       H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

       #print H
       print "Homography Matrix:" # Python 2.7 print statement
       print H                      # Python 2.7 print statement
       return H # Return H if successful
   else:
       print "Not enough matches to compute homography." # Python 2.7 print statement
       return None # Explicitly return None if homography cannot be computed