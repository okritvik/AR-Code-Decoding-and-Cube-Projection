"""
@author: Kumara Ritvik Oruganti
"""

import numpy as np
import cv2
import scipy
import os
import copy
import matplotlib.pyplot as plt
import math


def convert_to_binary(frame):
    """
    This function converts the color image to gray image

    Parameters
    ----------
    frame : numpy array
        BGR frame.

    Returns
    -------
    numpy array
        Gray Scale Image.

    """
    new_frame = copy.deepcopy(frame)
    return cv2.cvtColor(new_frame,cv2.COLOR_BGR2GRAY)

def detect_corners(ifft_image,gray_frame,frame):
        """
        This function detects the corners from the edge image using Shi-Tomasi corner detector.
        The outermost corners (White paper corners) are removed from the list. The next maximum indices
        are the corners of the AR Tag. Then a bounding box is created around those points.
        
        Parameters
        ----------
        ifft_image : numpy array
            edge image.
        gray_frame : numpy array
            gray frame.
        frame : numpy array
            BGR frame.
    
        Returns
        -------
        numpy nd array
            BGR Frame.
        """
        corners = cv2.goodFeaturesToTrack(ifft_image, 20, 0.15, 50) #Shi-Tomasi Corner Detection
        # print(type(corners))
        corner_x = []
        corner_y = []
        if corners is not None:
            corners = np.int0(corners)
            corners = corners.reshape(corners.shape[0], corners.shape[1]*corners.shape[2])
            for i in corners:
                x,y = i.ravel()
            #     corner_x.append(x)
            #     corner_y.append(y)
                cv2.circle(frame, (x,y), 6, (255,0,0),-1)
            
            #Finding and Deleting the white paper corners from the detected corners
            x_min = np.argmin(corners[:,0])
            x_max = np.argmax(corners[:,0])
            y_min = np.argmin(corners[:,1])
            y_max = np.argmax(corners[:,1])
            corners = np.delete(corners, (x_min,x_max,y_min,y_max), 0)
            
            
            corner_x = corners[:,0]
            corner_y = corners[:,1]
            #Extracting the next min max coordinates.
            if len(corner_x) != 0 and len(corner_y) != 0:
                min_x = min(corner_x)
                min_xy = corners[np.argmin(corner_x)][1]
            
                max_x = max(corner_x)
                max_xy = corners[np.argmax(corner_x)][1]
            
                min_y = min(corner_y)
                min_yx = corners[np.argmin(corner_y)][0]
                
                max_y = max(corner_y)
                max_yx = corners[np.argmax(corner_y)][0]
                
                p1 = (min_x,min_xy)
                p2 = (max_x,max_xy)
                p3 = (min_yx,min_y)
                p4 = (max_yx,max_y)
                
                
                # Drawing a bounding box around the tag        
                frame = cv2.line(frame, p1, p3, (13,255,207),4)
                frame = cv2.line(frame, p1, p4, (13,255,207),4)
                frame = cv2.line(frame, p2, p3, (13,255,207),4)
                frame = cv2.line(frame, p2, p4, (13,255,207),4)

                cv2.imshow("TAG",frame)
                return frame

                
def compute_mask(shape):
    """
    This function computes the circular mask for extracting the high frequency edges.

    Parameters
    ----------
    shape : tuple
        shape of the frame.

    Returns
    -------
    mask : circular mask

    """
    mask = np.ones(shape,dtype=np.uint8)
    radius = 200
    x, y = np.ogrid[:shape[0], :shape[1]]
    mask_area = (x - shape[0]/2) ** 2 + (y - shape[1]/2) ** 2 <= radius*radius
    mask[mask_area] = 0
    # print(mask_area.shape)
    return mask

def pipeline():
    
    """
    This function has the functional flow for the given problem.
    The flow goes as below:
    1) Frame is read (one of the frame from the video is stored as scenery.jpg and it is used to detect the AR Code)
    2) It is converted to gray scale image
    3) Threshold the gray image
    4) Apply Median Blur to remove Salt and Pepper Noise and smoothen the image
    5) Apply the FFT, FFT Shift, Mask, IFFT Shift and IFFT to get the edges from the image. Plot their magnitudes.
    6) Use the morphology close operator to get better output of the edges
    7) Detect the corners of AR Tag draw a bounding box

    Returns
    -------
    None.

    """
    
    frame = cv2.imread('scenery.jpg')
    
    
    kernel = np.ones((3, 3), np.uint8)
    
    # print(frame.shape)
    gray_frame = convert_to_binary(frame)
    
    _, gray_frame = cv2.threshold(gray_frame, 180, 255, cv2.THRESH_BINARY)
    
    gray_frame = cv2.medianBlur(gray_frame, 7)
    
    # gray_frame = cv2.dilate(gray_frame, kernel) #did not give better performance
    
    # gray_frame = cv2.GaussianBlur(gray_frame, (3,3),10) #Corners are clumpsy if gaussian blur is used
    
    fft_image = scipy.fft.fft2(gray_frame)
    fig,axes = plt.subplots()
    axes.imshow(20*np.log(np.abs(fft_image)),cmap='gray')
    axes.set_title("Magnitude of FFT")
    
    fft_shifted = scipy.fft.fftshift(fft_image)
    fig,axes = plt.subplots()
    axes.imshow(20*np.log(np.abs(fft_shifted)),cmap='gray')
    axes.set_title("Magnitude of Shifted FFT")
    
    mask = compute_mask(fft_shifted.shape)
    
    fft_masked = fft_shifted * mask
    fig,axes = plt.subplots()
    axes.imshow(20*np.log(np.abs(fft_masked)),cmap='gray')
    axes.set_title("Magnitude of Masked FFT")
    
    ifft_shifted = scipy.fft.ifftshift(fft_masked)
    fig,axes = plt.subplots()
    axes.imshow(20*np.log(np.abs(ifft_shifted)),cmap='gray')
    axes.set_title("Magnitude of Shifted IFFT")
    
    ifft_image = scipy.fft.ifft2(ifft_shifted)
    fig,axes = plt.subplots()
    axes.imshow(20*np.log(np.abs(ifft_image)),cmap='gray')
    axes.set_title("Magnitude of IFFT")
    
    ifft_image = np.uint8(np.abs(ifft_image))
    fig,axes = plt.subplots()
    axes.imshow(ifft_image,cmap='gray')
    axes.set_title("IFFT Image")
    
    closed_image = cv2.morphologyEx(ifft_image, cv2.MORPH_CLOSE, kernel)
    
    cv2.imshow("Edge FFT",closed_image)
    
    ret = detect_corners(closed_image, gray_frame,frame)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("Detected TAG.png", ret)
    
    print("Done")
    
def main():
    """
    This function calls the pipeline function to run the overall algorithm

    Returns
    -------
    None.

    """
    pipeline()
    plt.show()
    
if __name__ == '__main__':
    main()