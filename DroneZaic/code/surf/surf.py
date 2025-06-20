#!/usr/bin/env python

import cv2
import numpy as np

def surf(gray1, gray2):
   #convert images to grayscale
   #gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
   #gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

   #initialize SURF detector
   surf = cv2.xfeatures2d.SURF_create()

   #detect keypoints and compute descriptors
   keypoints1, descriptors1 = surf.detectAndCompute(gray1, None)
   keypoints2, descriptors2 = surf.detectAndCompute(gray2, None)

   #initialize matcher
   matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

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
      print("Homography Matrix:")
      print(H)
   else:
      print("Not enough matches to compute homography.")
   return H
   
	
