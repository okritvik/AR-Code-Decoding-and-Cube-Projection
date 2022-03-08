"""
@author: Kumara Ritvik Oruganti
"""

import numpy as np
import cv2
import scipy
import os
import copy
import matplotlib.pyplot as plt


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

def decode_tag(tag_frame):
    """
    This function decodes the AR Tag orientation and information.

    Parameters
    ----------
    tag_frame : AR Tag image
    
    Returns
    -------
    None

    """
    # print(tag_frame.shape)
    tag_frame = cv2.resize(tag_frame, (160,160))
    # cv2.imshow("TAG",tag_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # grid = 8 Grid Size
    
    grid_pixels = 20 #8x8 grid. Hence each grid has 20 pixels
    
    orientation = [] #To calculate the orientation
    
    #Top left grid
    t_left = tag_frame[2*grid_pixels:3*grid_pixels,2*grid_pixels:3*grid_pixels]
    # print(np.mean(t_left))
    orientation.append(255 if np.mean(t_left)>200 else 0)
    
    #Top right grid
    t_right = tag_frame[2*grid_pixels:3*grid_pixels,5*grid_pixels:6*grid_pixels]
    # print(np.mean(t_right))
    orientation.append(255 if np.mean(t_right)>200 else 0)
    
    #Bottom left grid
    b_left = tag_frame[5*grid_pixels:6*grid_pixels,2*grid_pixels:3*grid_pixels]
    # print(np.mean(b_left))
    orientation.append(255 if np.mean(b_left)>200 else 0)
    
    #Bottom Right grid
    b_right = tag_frame[5*grid_pixels:6*grid_pixels,5*grid_pixels:6*grid_pixels]
    # print(np.mean(b_right))
    orientation.append(255 if np.mean(b_right)>200 else 0)
    
    
    index = orientation.index(255) #Get where the white corner is present
    # print(index)
    
    if index == 3:
        print("Normal Orientation")
    
    if index == 2:
        print("Rotated by 90")
        tag_frame = cv2.rotate(tag_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
    if index == 0:
        print("Rotated by 180")
        tag_frame = cv2.rotate(tag_frame, cv2.ROTATE_180)
        
    if index == 1:
        print("Rotated by 270")
        tag_frame = cv2.rotate(tag_frame, cv2.ROTATE_90_CLOCKWISE)
    
    #Extract the information from the inner 2x2 grid
    i1 = tag_frame[3*grid_pixels:4*grid_pixels,3*grid_pixels:4*grid_pixels]
    i2 = tag_frame[3*grid_pixels:4*grid_pixels,4*grid_pixels:5*grid_pixels]
    i3 = tag_frame[4*grid_pixels:5*grid_pixels,3*grid_pixels:4*grid_pixels]
    i4 = tag_frame[4*grid_pixels:5*grid_pixels,4*grid_pixels:5*grid_pixels]
    
    info = []
    
    info.append(1 if np.mean(i1)>240 else 0)
    info.append(1 if np.mean(i2)>240 else 0)
    info.append(1 if np.mean(i3)>240 else 0)
    info.append(1 if np.mean(i4)>240 else 0)
    # compute the binary information.
    sum = 0
    for i in range(0,4):
        sum = sum+info[i]*np.power(2,i)
        
    print("INFORMATION: ",sum)
    

def detect_corners(ifft_image,gray_frame,frame):
        """
        This function detects the corners from the edge image using Shi-Tomasi corner detector.
        The outermost corners are the corners of the reference AR Tag.
        This is used for the decoding the reference AR Tag. The same can be used for decoding the AR tag from Video with little modifications that is implemented in the project1_2a code.
        
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
        corners = cv2.goodFeaturesToTrack(ifft_image, 20, 0.15, 50)
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
            
            # Used to 
            # x_min = np.argmin(corners[:,0])
            # x_max = np.argmax(corners[:,0])
            # y_min = np.argmin(corners[:,1])
            # y_max = np.argmax(corners[:,1])
            # corners = np.delete(corners, (x_min,x_max,y_min,y_max), 0)
            
            
            corner_x = corners[:,0]
            corner_y = corners[:,1]
            if len(corner_x) != 0 and len(corner_y) != 0:
                min_x = min(corner_x)
                # min_xy = corners[np.argmin(corner_x)][1]
            
                max_x = max(corner_x)
                # max_xy = corners[np.argmax(corner_x)][1]
            
                min_y = min(corner_y)
                # min_yx = corners[np.argmin(corner_y)][0]
                
                max_y = max(corner_y)
                # max_yx = corners[np.argmax(corner_y)][0]
                
                p1 = (min_x,min_y)
                p2 = (max_x,min_y)
                p3 = (min_x,max_y)
                p4 = (max_x,max_y)
                
                
                        
                frame = cv2.line(frame, p1, p2, (13,255,207),4)
                frame = cv2.line(frame, p2, p4, (13,255,207),4)
                frame = cv2.line(frame, p1, p3, (13,255,207),4)
                frame = cv2.line(frame, p3, p4, (13,255,207),4)

                cv2.imshow("TAG",frame)
                
                tag_frame = gray_frame.copy()
                
                tag_frame = tag_frame[min_y:max_y,min_x:max_x]
        
                
                
                decode_tag(tag_frame)
                return frame


def pipeline():
    """
    This function has the functional flow for the given problem.
    The flow goes as below:
    1) Reference AR Tag Frame is read
    2) It is converted to gray scale image
    3) Threshold the gray image
    4) Apply Median Blur to remove Salt and Pepper Noise and smoothen the image
    5) Apply the FFT, FFT Shift, Mask, IFFT Shift and IFFT to get the edges from the image
    6) Use the morphology close operator to get better output of the edges
    7) Detect the corners of AR Tag and decode the information and orientation of the Tag
    8) Print them on the console
    Returns
    -------
    None.

    """
    
    frame = cv2.imread('ref_tag1.png')
    
    #Uncomment below line and pass three rotations cv2.ROTATE_90_CLOCKWISE cv2.ROTATE_180 in place of the second argument of the below function call to test with the different orientations of the reference AR Tag.
    
    # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    kernel = np.ones((3, 3), np.uint8)
    
    # print(frame.shape)
    gray_frame = convert_to_binary(frame)
    
    _, gray_frame = cv2.threshold(gray_frame, 180, 255, cv2.THRESH_BINARY)
    
    gray_frame = cv2.medianBlur(gray_frame, 7)
    
    # Uncomment below fig,axes and plt.show() in main function to see the outputs of FFT, FFT Shift, Masked FFT Shift, IFFT Shift, IFFT
    
    fft_image = scipy.fft.fft2(gray_frame)
    # fig,axes = plt.subplots()
    # axes.imshow(20*np.log(np.abs(fft_image)),cmap='gray')
    # axes.set_title("Magnitude of FFT")
    
    fft_shifted = scipy.fft.fftshift(fft_image)
    # fig,axes = plt.subplots()
    # axes.imshow(20*np.log(np.abs(fft_shifted)),cmap='gray')
    # axes.set_title("Magnitude of Shifted FFT")
    
    mask = compute_mask(fft_shifted.shape)
    
    fft_masked = fft_shifted * mask
    # fig,axes = plt.subplots()
    # axes.imshow(20*np.log(np.abs(fft_masked)),cmap='gray')
    # axes.set_title("Magnitude of Masked FFT")
    
    ifft_shifted = scipy.fft.ifftshift(fft_masked)
    # fig,axes = plt.subplots()
    # axes.imshow(20*np.log(np.abs(ifft_shifted)),cmap='gray')
    # axes.set_title("Magnitude of Shifted IFFT")
    
    ifft_image = scipy.fft.ifft2(ifft_shifted)
    # fig,axes = plt.subplots()
    # axes.imshow(20*np.log(np.abs(ifft_image)),cmap='gray')
    # axes.set_title("Magnitude of IFFT")
    
    ifft_image = np.uint8(np.abs(ifft_image))
    # fig,axes = plt.subplots()
    # axes.imshow(ifft_image,cmap='gray')
    # axes.set_title("IFFT Image")
    
    closed_image = cv2.morphologyEx(ifft_image, cv2.MORPH_CLOSE, kernel)
    
    cv2.imshow("Edge FFT",closed_image)
    
    ret = detect_corners(closed_image, gray_frame,frame)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("TAG_Decoding.png", ret)
    
    print("Done")


def main():
    """
    This function calls the pipeline function to run the overall algorithm

    Returns
    -------
    None.

    """
    pipeline() 
    # plt.show()
    
if __name__ == '__main__':
    main()
