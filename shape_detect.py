#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:18:27 2020

@author: jericolinux

PseudoCode
Get Input Images; sample shape, shape in larger image
Slide a certain window across the evaluation form
Determine which window contains the smaller image
    Algorithm must work despite of any rotation or enhancements of shape
return the window (Prototype return)
return the edge location (final return)
"""

import numpy as np
import cv2
from PIL import Image
from shape_extract import preprocess, extract_boundbox
from sklearn.metrics import mean_squared_error 
import time
import matplotlib.pyplot as plt

# def get_moment(img):
#     # // Calculate Moments
#     Moments moments = moments(img, false);
     
#     // Calculate Hu Moments
#     double huMoments[7];
#     HuMoments(moments, huMoments);
    
#     # Log scale hu moments
#     for i in range(0,7):
#       huMoments[i] = -1* copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])))
    
#     huMoments = np.ndarray(huMoments)
    
#     print(huMoments, type(huMoments))
    
#     return huMoments
    
    
    

# # Algorithm to check if target has the same pattern
# def is_shape_similar(target, img):
#     # // Calculate Moments
#     Moments moments = moments(im, false);
     
#     # // Calculate Hu Moments
#     double huMoments[7];
#     HuMoments(moments, huMoments);
    
#     # Log scale hu moments
#     for i in range(0,7):
#       huMoments[i] = -1* copysign(1.0, huMoments[i]) * log10(abs(huMoments[i])))
    
#     print(huMoments)


def draw_border(img, point1, point2, point3, point4, line_length):

    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    x4, y4 = point4    

    cv2.circle(img, (x1, y1), 3, (255, 0, 255), -1)    #-- top_left
    cv2.circle(img, (x2, y2), 3, (255, 0, 255), -1)    #-- bottom-left
    cv2.circle(img, (x3, y3), 3, (255, 0, 255), -1)    #-- top-right
    cv2.circle(img, (x4, y4), 3, (255, 0, 255), -1)    #-- bottom-right

    cv2.line(img, (x1, y1), (x1 , y1 + line_length), (255, 0, 0), 2)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length , y1), (255, 0, 0), 2)

    cv2.line(img, (x2, y2), (x2 , y2 - line_length), (255, 0, 0), 2)  #-- bottom-left
    cv2.line(img, (x2, y2), (x2 + line_length , y2), (255, 0, 0), 2)

    cv2.line(img, (x3, y3), (x3 - line_length, y3), (255, 0, 0), 2)  #-- top-right
    cv2.line(img, (x3, y3), (x3, y3 + line_length), (255, 0, 0), 2)

    cv2.line(img, (x4, y4), (x4 , y4 - line_length), (255, 0, 0), 2)  #-- bottom-right
    cv2.line(img, (x4, y4), (x4 - line_length , y4), (255, 0, 0), 2)

    return img



## Algorithm prototype
def get_bubble_corners(filename):
    # Required Vars
    # Input variables
    targets = [
        ['./img/shape1.png', 'upper_left'],
        ['./img/shape2.png', 'upper_right'],
        ['./img/shape3.png', 'lower_left'],
        ['./img/shape4.png', 'lower_right']
        ]
    
    
    corners = []
    for target in targets:
        ## Get the inputs
        targetShape = cv2.imread(target[0])
        imgTarget = cv2.imread(fileTarget)
        
        start_time = time.time()
        # print(target)
        loc, proc_img = match_shape(targetShape,imgTarget, target[1])
        corners.append(find_corner(proc_img, loc, imgTarget, target[1]))
        runtime = time.time()-start_time
        print("Program runtime: ", runtime) 

    # print(corners)
    return corners
    
def find_corner(inspect_box, loc, imgTarget, corner):
    
    # Get the dimension of each imaage
    target_shape = inspect_box.shape
    img_shape = imgTarget.shape
    
    # Reference points
    upper_left_pt = [0,0]
    lower_left_pt = [int(img_shape[0]), 0]
    upper_right_pt = [0,int(img_shape[1])]
    lower_right_pt = [int(img_shape[0]), int(img_shape[1])]
    
    if (corner == 'upper_left'):
        ref_pt = upper_left_pt
    elif (corner == 'lower_left'):
        ref_pt = lower_left_pt
    elif (corner == 'upper_right'):
        ref_pt = upper_right_pt
    else:
        ref_pt = lower_right_pt
    
    
    # This finds the element index which is the edge of the image
    dst = cv2.cornerHarris(inspect_box,2,3,0.04)
    
    # Create predicted label for edge points
    pts = np.where(dst>0.1)
    # print(type(pts))
    # print(len(pts[0]), len(pts[1]))
    
    corner_pts = []
    mse_list = []
    for i in range(len(pts[0])):
        corner_pt = [pts[0][i]+loc[0],pts[1][i]+loc[1]]
        corner_pts.append([pts[1][i]+loc[1],pts[0][i]+loc[0]])
        mse_list.append(mean_squared_error(corner_pt, ref_pt))
        
        # print(ref_pt, corner_pt, [pts[0][i],pts[1][i]])
        # print(mean_squared_error(corner_pt, ref_pt))
    
    
    # print(mse_list)
    
    min_indx = np.where(mse_list==min(mse_list))[0][0]
    
    # print(min_indx)
    
    target_corner = corner_pts[min_indx]
    
    print(target_corner)
    
    return target_corner

def match_shape(target, imgTarget, corner):
    # Preprocess the larger image
    target = preprocess(target)
    imgTarget_preprocessed = preprocess(imgTarget)
    
    # Get the dimension of each imaage
    target_shape = target.shape
    img_shape = imgTarget_preprocessed.shape
    
    # browse_max = [img_shape[1]-target_shape[1], 
    #               img_shape[0]-target_shape[0]]
    
    
    # This one splits the image into six
    # Allowing analysis only on the important parts
    upper_left_browse = [0,0,
                        int(img_shape[0]/2)-target_shape[0],
                        int(img_shape[1]/3)-target_shape[1]]
    lower_left_browse = [int(img_shape[0]/4)*3,0,
                         int(img_shape[0])-target_shape[0],
                         int(img_shape[1]/3)-target_shape[1]]
    upper_right_browse = [0, int(img_shape[1]/3)*2,
                          int(img_shape[0]/2)-target_shape[0],
                          int(img_shape[1])-target_shape[1]]
    lower_right_browse = [int(img_shape[0]/3)*2, 
                          int(img_shape[1]/3)*2,
                          int(img_shape[0])-target_shape[0],
                          int(img_shape[1])-target_shape[1]]
    
    
    # This variables allows which indicator will be targeted
    # print(corner, 'upper_left')
    if (corner == 'upper_left'):
        # print("I GOT EM!!")
        # print(upper_left_browse)
        browse = upper_left_browse
    elif (corner == 'lower_left'):
        browse = lower_left_browse
    elif (corner == 'upper_right'):
        browse = upper_right_browse
    else:
        browse = lower_right_browse
    
    
    # print(lower_left_pt)
    # print(img_shape)
    # print(browse)
    
    # This loop finds the window that is close to the target shape
    get_out = 0
    for row in range(browse[0],browse[2],8):
        for col in range(browse[1],browse[3],8):
        
            inspect_loc = [(col,row), (col+target_shape[1], row+target_shape[0])]
            
            inspect_box = extract_boundbox(imgTarget_preprocessed, inspect_loc)
            
            orig_box = extract_boundbox(imgTarget, inspect_loc)
            
            
            d1 = cv2.matchShapes(target,inspect_box,cv2.CONTOURS_MATCH_I1,0)
            d4 = cv2.matchTemplate(target,inspect_box,cv2.TM_CCOEFF_NORMED)
            # current_match = [d1,d2,d3]
            # print(d1,d4)
            
            # cv2.imshow('Random', inspect_box)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            # print("MAO")
            
            
            # cv2.imshow('BOID', target)
            # cv2.waitKey(0)
            
            
            # cv2.destroyAllWindows()
                
            
            # break
            
                
            # This condition only passes the ones that are quite similar
            if d1 < 0.002  and d4 > 0.7:
                inspect_box_loc = [row,col]
                
                # dst = cv2.cornerHarris(inspect_box,2,3,0.04)
                
                # # #result is dilated for marking the corners, not important
                # dst = cv2.dilate(dst,None)
                
                # # Threshold for an optimal value, it may vary depending on the image.
                # orig_box[dst>0.01*dst.max()]=[0,0,255]
                
                # cv2.imshow('dst',orig_box)
                # if cv2.waitKey(0) & 0xff == 27:
                #     cv2.destroyAllWindows()
                
                return inspect_box_loc, inspect_box
                
                # print("Target Corner is: ", target_corner)
                
                
                
                # cv2.imwrite('./img/shapeofme.png', orig_box)
                    
                # Create reference labels for the edge
                # [for i in range(len(points[0]))]
                
                # print(pts)
                # print(mean_squared_error())
                # The one that is closest to the most lower left point
                # exit()
                
               
                
                get_out = 1
                
            
                break
        
        if get_out: break
    
    return target_corner

def extract_evals(fileTarget)
    # Find the corners
    orig_corners = get_bubble_corners(fileTarget)

    img = cv2.imread(fileTarget)
    
    ## Visual Debugging for checking if the corner acquired is correct
    # line_length = 15
    # fin_img = draw_border(img, orig_corners[0], orig_corners[1], orig_corners[2], 
    #                       orig_corners[3], line_length)
    
    # # cv2.imshow('fin_img', fin_img)
    
    # # plt.imshow(fin_img)
    # # plt.show()
    
    # # exit()
    
    
    rows, cols = [1033-94, 1608-92]
    
    pts1 = np.float32(orig_corners)
    pts2 = np.float32([[0,0], [cols,0],[0,rows],[cols,rows]])
    # print(pts1)
    # print(pts2)
    # print(pts1.shape, pts2.shape)
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    
    dst = cv2.warpPerspective(img,M,(cols,rows), flags=cv2.INTER_LINEAR)
    
    ## Visual Debugging to check if transform is correct
    # plt.subplot(121),plt.imshow(img),plt.title('Input')
    # plt.subplot(122),plt.imshow(dst),plt.title('Output')
    # plt.show()
    
    return dst

# fileTarget = './img/CHE-TALK-11.png'
fileTarget = './img/EEEI-TALK-03.png'

cv2.imwrite(dst, './corrected/'+fileTarget)