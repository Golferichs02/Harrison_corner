""" harrison_corners.py
This script reads two images provided by the user via terminal and performs a corner feature extraction
with harris_corner, then it identifies descriptors for the images with BRIEF, and computes the matches with BRUTE MATCHER
to show the common corners previosuly detected in both images.

Authors: Emilio Arredondo Payán (628971) & Jorge Alberto Rosales de Golferichs (625544) 
Contacts: emilio.arredondop@udem.edu, jorge.rosalesd@udem.edu
Organisation: Universidad de Monterrey
First created on wensday 24 april 2024

example of usage: python ".\harris_corners.py" --image1 .\image1.jpg --image2 .\image2.jpg --resize 50

"""
import numpy as np
import cv2 as cv
import argparse
from numpy.typing import NDArray

def user_interaction() ->argparse.ArgumentParser:
    """
    Interact with the user to get the path for the images.
    
    Returns:
        args: Path for the image.
    """
    parser = argparse.ArgumentParser(description='Corner Detection')
    parser.add_argument('-i', '--image1',
                        type=str,
                        required=True,
                        help="Path to the image file where corners will be detected")
    parser.add_argument('-o', '--image2',
                        type=str,
                        required=True,
                        help="Path to the image file where corners will be detected")
    parser.add_argument('--resize',
                        type= int,
                        required= True,
                        help= "Percentage to resize the image")
    args = parser.parse_args()
    return args

def extract_corners_and_descriptors(image:NDArray)-> tuple[list[cv.KeyPoint], NDArray]:
    """
    Extracts corner points and descriptors from an image using Harris corner detection.
    Parameters:
        image: Image to perform corener detection

    Returns:
        kp: List containing the points where a corener was detected.
        des: Array containing the descriptors of the provided image.
    """

    # Harris corner detection/
    harris_corners1 = cv.cornerHarris(image, blockSize=2, ksize=3, k=0.04)
# Threshold for corner detection
    threshold = 0.01 * harris_corners1.max()
    thresholded_img1 = np.zeros_like(image)
    thresholded_img1[harris_corners1 > threshold] = 255
    keypoints1 = np.argwhere(thresholded_img1 == 255).tolist()

# Convert keypoints to list of KeyPoint objects
    keypoints1 = [cv.KeyPoint(x[1], x[0], 3) for x in keypoints1]

# Initiate BRIEF extractor
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

# compute the descriptors with BRIEF
    kp, des = brief.compute(image, keypoints1)
    return kp, des

def draw_matches(img1:NDArray, img2:NDArray, descriptors1:NDArray, descriptors2:NDArray, kp1:list[cv.KeyPoint],kp2:list[cv.KeyPoint]):
    """
    Draws matches between two images based on their descriptors and keypoints.

    Parameters:
        img1: Original Image
        img2: Image to compare or to match the first image
        descriptors1: List of descriptors of the first image
        descriptors1: List of descriptors of the second image
        kp1: List containing the keypoints of the first image
        kp2: List containing the keypoints of the second image

    Returns:
        None
    """
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descriptors1,descriptors2,k=2)

# Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.70*n.distance:
            good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('Matched Features', img3)
    close_windows()
    return None

def load_image(path:str)->cv:
    """
    Loads an image from the specified path in grayscale.
    
    Parameters:
        path: The path to the image file.
    
    Returns:
        img: The loaded image in grayscale.
    """
    img = cv.imread(path,cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("One or both images not found. Please check the paths.")
    return img

def close_windows()->None:
    """
    Waits for a key press in the display window and then closes all opened windows.
    """
    cv.waitKey(0)
    cv.destroyAllWindows()

def resize(img:NDArray, per:int) -> NDArray:
    """
    Resize an image to a specified percentage    
    Parameters:
        img: Image to perfor the resize.
        per: Percentage to which the resize will be performed
    Returns:
        img_resize: Resized image
    """
    # Resize images
    width = int(img.shape[1] * per / 100)
    height = int(img.shape[0] * per / 100)
    dim = (width, height)
    img_resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return img_resized

def pipeline()->None:
    """
    Main function that sets up the pipeline for corner detection and feature matching.
    """
    args = user_interaction()
    img1 = load_image(args.image1)
    img2 = load_image(args.image2)
    img1 = resize(img1,args.resize)
    img2 = resize(img2,args.resize)
    kp1, descriptors1 = extract_corners_and_descriptors(img1)
    kp2, descriptors2 = extract_corners_and_descriptors(img2)
    draw_matches(img1, img2, descriptors1, descriptors2, kp1,kp2)
    return None

if __name__ == "__main__":
    pipeline()