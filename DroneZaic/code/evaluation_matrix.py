#!/bin/bash

import cv2
import math
import argparse
import numpy as np
import os

'''
compute the area of a rectangle in mosaic image
'''

polygon_points = []

def mouse_callback(event, x, y, flags, param):
    global polygon_points, image_with_polygon

    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        print("Point {} selected at (x={}, y={})".format(len(polygon_points), x, y))

        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        if len(polygon_points) > 1:
            cv2.line(image, polygon_points[-2], polygon_points[-1], (255, 0, 0), 5)

        w,h,c = image.shape
        cv2.imshow("RawImage", image)


def count_total_pixel(points):
    points_array = np.array(points, dtype=np.int32)

    mask = np.zeros_like(image[:, :, 0])
    cv2.fillPoly(mask, [points_array], 255)

    #count the number of white pixels (pixels inside the polygon) in the mask
    num_pixels_inside = np.sum(mask == 255)
    w,h = mask.shape
    #win_resize("mask", w, h, scale)
    #cv2.imshow("mask", mask)

    x, y, w, h = cv2.boundingRect(points_array)
    mask_color = cv2.merge([mask] * 3) 
    cv2.rectangle(mask_color, (x, y), (x + w, y + h), (0, 0, 255), 10)
    
    
    #cv2.polylines(mask_color, [np.array(rect_points)], 0, (0, 0, 255), 5)
    win_resize("mask", w, h, scale)
    cv2.imshow("mask", mask_color)
    #cv2.imshow("mask", mask)

    #count_total_pixel(rect_points)

    num_pixels_rect = w*h
    
    return num_pixels_inside, num_pixels_rect, mask_color



def win_resize(win_name, width, height, scale):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, np.round(width/scale), np.round(height/scale))

def connect_first_last_point(image, points):
    if len(points) > 1:
        cv2.line(image, points[0], points[-1], (255, 0, 0), 5)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-image_path', type=str, nargs='+', help="paths to one or more images or image directories")
    parser.add_argument('-save_path', dest='save_path', default="./EVAL", type=str, help="path to save result")
    parser.add_argument('-fname', '--fname', dest='fname', default='ASIFT', help='filename')
    parser.add_argument('-scale', '--scale', dest='scale', default=1, type=int, help='image size scale')
    parser.add_argument('-pxr', '--pxr', dest='pxr', default=1, type=float, help='pixel to cm ratio')
    parser.add_argument('-actual_area', '--actual_area', dest='actual_area', type=int, help='actual area or ground truth')
    args = parser.parse_args()

    actual_area = args.actual_area
    image_path = args.image_path
    scale = args.scale
    pxr = args.pxr
    save_path = args.save_path
    fname = args.fname

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    
    print(image_path)
    image = cv2.imread(image_path[0])
    width, height, c = image.shape
    
    #resize the window size
    win_resize("RawImage", width, height, scale)
    

    image_with_polygon = image.copy()

    cv2.imshow("RawImage", image)
    
    cv2.setMouseCallback("RawImage", mouse_callback)

    
    
    cv2.waitKey(0)

    connect_first_last_point(image, polygon_points)
    cv2.imshow("RawImage", image)
    
    num_pixels_inside,  num_pixels_rectangle, mask = count_total_pixel(polygon_points)

    geometry_eval = num_pixels_inside/np.float(num_pixels_rectangle)
    print("***************************************")
    print("number of pixel inside rectangle geometry: ", num_pixels_rectangle)
    print("Number of pixels inside the polygon:", num_pixels_inside)
    print("=======================================")
    print("Area inside the polygon in cm:", num_pixels_inside*pxr*pxr)
    print("geometry evaluation : ", geometry_eval)
    print("=======================================")
    print("APE: ", np.abs(actual_area-(num_pixels_inside*pxr*pxr))/(num_pixels_inside*pxr*pxr)*100)
    
   

    cv2.setMouseCallback("RawImage", lambda *args: None)

    print(save_path+"/"+fname+"_"+str(num_pixels_inside)+'.png')

    cv2.imwrite((save_path+"/"+fname+"_"+str(num_pixels_inside*pxr*pxr)+'.png'), image)
    cv2.imwrite((save_path+"/"+fname+"_"+str(geometry_eval)+'_mask.png'), mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
