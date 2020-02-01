#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 12:54:51 2020

@author: jericolinux
"""

import cv2
import numpy as np

dest_folder = 'img'

# Extract bounding box
def extract_boundbox(img, position):
    """
    Extracts the region of interest
    """
    pos1 = position[0]
    pos2 = position[1]
    return img[pos1[1]:pos2[1], pos1[0]:pos2[0]]

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_binary(img):
    return cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

def to_negative(img):
    return cv2.bitwise_not(img)

def im_close(img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    

def preprocess(img):
    return im_close(to_negative(to_binary(to_grayscale(img))))


if __name__ == "__main__":
    path = f'./img/CHE-TALK-11.png'
    img = cv2.imread(path)
    
    # Get the shapes to detect
    shape1 = [(78,81),(251,213)]
    shape2 = [(1540,78),(1639,185)]
    shape3 = [(77,959),(172,1042)]
    shape4 = [(1528,960),(1630,1048)]
    
    shape1 = extract_boundbox(img, shape1)
    shape2 = extract_boundbox(img, shape2)
    shape3 = extract_boundbox(img, shape3)
    shape4 = extract_boundbox(img, shape4)
    
    # shape1 =  preprocess(shape1)
    # shape2 =  preprocess(shape2)
    # shape3 =  preprocess(shape3)
    # shape4 =  preprocess(shape4)
    
    
    cv2.imwrite(f'./{dest_folder}/shape1.png', shape1)
    cv2.imwrite(f'./{dest_folder}/shape2.png', shape2)
    cv2.imwrite(f'./{dest_folder}/shape3.png', shape3)
    cv2.imwrite(f'./{dest_folder}/shape4.png', shape4)