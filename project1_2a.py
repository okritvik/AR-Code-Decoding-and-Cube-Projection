"""
@author: Kumara Ritvik Oruganti
"""
import numpy as np
import cv2
import scipy
import os
import copy



def find_homography(p1,p2):
    """
    This function computes the Homography between 4 points
    Parameters
    ----------
    p1 : List
        Corner points of the AR Tag.
        
    p2 : List
        Corner points of the 160x160 image.
    
    Note that the points should be in the same order

    Returns
    -------
    Homography Matrix.

    """
    try:
        A = []
        for i in range(0, len(p1)):
            x, y = p1[i][0], p1[i][1]
            u, v = p2[i][0], p2[i][1]
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
            A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v]) #Computing the Homography Matrix as given in the Homework1 problem statement
        A = np.asarray(A)
        U, S, Vh = np.linalg.svd(A) #Using SVD
        L = Vh[-1,:] / Vh[-1,-1] #Normalizing
        H = L.reshape(3, 3) #Reshaping to 3x3
        return(H)
    except:
        pass


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

def decode_tag(tag_frame,testudo):
    """
    This function decodes the AR Tag orientation and information.

    Parameters
    ----------
    tag_frame : AR Tag image
    
    testudo : Testudo image
        

    Returns
    -------
    integer
        index at which the white corner is present.
    integer
        information that is decoded from the AR tag.

    """
    
    # print(tag_frame.shape)
    tag_frame = cv2.resize(tag_frame, (160,160)) #Resize the tag to 160x160
    # cv2.imshow("TAG",tag_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    #grid = 8
    
    grid_pixels = 20 #8x8 grid. Hence each grid has 20 pixels
    
    orientation = [] #To calculate the orientation
    
    #Top left grid
    t_left = tag_frame[2*grid_pixels:3*grid_pixels,2*grid_pixels:3*grid_pixels]
    # print(np.mean(t_left))
    orientation.append(255 if np.mean(t_left)>200 else 0)
    
    #Top Right Grid
    t_right = tag_frame[2*grid_pixels:3*grid_pixels,5*grid_pixels:6*grid_pixels]
    # print(np.mean(t_right))
    orientation.append(255 if np.mean(t_right)>200 else 0)
    
    #Bottom Left Grid
    b_left = tag_frame[5*grid_pixels:6*grid_pixels,2*grid_pixels:3*grid_pixels]
    # print(np.mean(b_left))
    orientation.append(255 if np.mean(b_left)>200 else 0)
    
    #Bottom Right Grid
    b_right = tag_frame[5*grid_pixels:6*grid_pixels,5*grid_pixels:6*grid_pixels]
    # print(np.mean(b_right))
    orientation.append(255 if np.mean(b_right)>200 else 0)
    
    try:
        index = orientation.index(255) #Get where the white corner is present
        print(index)
        if index is not None:
        
            if index == 3:
                print("Normal Orientation")
            
            if index == 2:
                print("Rotated by 90")
                tag_frame = cv2.rotate(tag_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # testudo = cv2.rotate(testudo,cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            if index == 0:
                print("Rotated by 180")
                tag_frame = cv2.rotate(tag_frame, cv2.ROTATE_180)
                # testudo = cv2.rotate(testudo,cv2.ROTATE_180)
                
            if index == 1:
                print("Rotated by 270")
                tag_frame = cv2.rotate(tag_frame, cv2.ROTATE_90_CLOCKWISE)
                # testudo = cv2.rotate(testudo,cv2.ROTATE_90_CLOCKWISE)
            
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
        
        # print(info)
        # compute the binary information.
        binary = 0
        for i in range(0,4):
            binary = binary+info[i]*np.power(2,i)
            
        # print("INFORMATION: ",binary)
        return index,binary
    except:
        print("Index not found")
        return None,None

def detect_corners(ifft_image,gray_frame,frame):
        """
        This function detects the corners from the edge image using Shi-Tomasi corner detector.
        The outermost corners (White paper corners) are removed from the list. The next maximum indices
        are the corners of the AR Tag. They are extracted and given to compute the Homography Matrix.
        The Homography Matrix is with respect to the image and world coordinates instead of world and image coordinates
        Here image is the Frame and world is the 160x160 image.
    
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
        numpy array
            Homography Matrix.
    
        """
        corners = cv2.goodFeaturesToTrack(ifft_image, 20, 0.15, 50) #Shi Tomasi corner detection
        # print(type(corners))
        corner_x = [] #To store the x coordinates of detected corners
        corner_y = [] #To store the y coordinates of detected corners
        p=[] #To store all the x,y coordinates of  the AR Tag
        if corners is not None: #If the detected corners are not empty
            corners = np.int0(corners) #Convert to integers
            corners = corners.reshape(corners.shape[0], corners.shape[1]*corners.shape[2]) #Reshape them to get in x,y
            for i in corners:
                x,y = i.ravel()
            #     corner_x.append(x)
            #     corner_y.append(y)
                # cv2.circle(frame, (x,y), 6, (255,0,0),-1)
            
            x_min = np.argmin(corners[:,0]) #get the xmin index
            x_max = np.argmax(corners[:,0]) #get the xmax index
            y_min = np.argmin(corners[:,1]) #get the ymin index
            y_max = np.argmax(corners[:,1]) #get the ymax index
            corners = np.delete(corners, (x_min,x_max,y_min,y_max), 0) #Delete them from the corners
            
            
            corner_x = corners[:,0] #get all the x coordinates into list
            corner_y = corners[:,1] #get all the y coordinates into list
            if len(corner_x) != 0 and len(corner_y) != 0:
                #Get the coordinates of the AR Tag
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
                
                
                #Append the corners in order to the list
                p.append(p3)
                p.append(p2)
                p.append(p1)
                p.append(p4)
                
                #Draw Bounding Box
                frame = cv2.line(frame, p1, p3, (13,255,207),4)
                frame = cv2.line(frame, p1, p4, (13,255,207),4)
                frame = cv2.line(frame, p2, p3, (13,255,207),4)
                frame = cv2.line(frame, p2, p4, (13,255,207),4)
                
                #Compute the homography between the points
                
                H = find_homography(p, [(0,0),(0,160),(160,0),(160,160)])
                
                # print(H)

                #return frame[min_y:max_y, min_x:max_x],H
                return frame,H
            else:
                return None,None
        else:
            return None,None
            

def warp_perspective(H,img,max_height,max_width):
    """
    This function uses the Homography matrix to warp the AR image on to the projection  plane

    Parameters
    ----------
    H : array
        Homography Matrix.
    img : image frame
        DESCRIPTION.
    maxHeight : int
        size of the projection.
    maxWidth : int
        width of the projection.

    Returns
    -------
    Warped/Projected AR Tag

    """
    try:
        H_inv=np.linalg.inv(H) #Compute the H inverse 
        warped=np.zeros((max_height,max_width,3),np.uint8) #Generate a matrix with given sizes
        for a in range(max_height):
            for b in range(max_width):
                f = [a,b,1]
                f = np.reshape(f,(3,1))
                x, y, z = np.matmul(H_inv,f)
                warped[a][b] = img[int(y/z)][int(x/z)]
        return(warped)
    except:
        return None
                
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
    This function has the whole pipeline for solving the question.
    1) It takes the current working directory and opens the video.
    2) Read the image to be warped and resize it to 160x160
    3) Codec and output writer objects are initialized.
    4) Frame is read
    5) It is converted to gray scale image
    6) Threshold the gray image
    7) Apply Median Blur to remove Salt and Pepper Noise and smoothen the image
    8) Apply the FFT, FFT Shift, Mask, IFFT Shift and IFFT to get the edges from the image
    9) Use the morphology close operator to get better output of the edges
    10) Detect the corners of AR Tag and compute the homography
    11) Warp the detected AR Tag and decode the information, print the rotation and information to console
    12) According to the rotation of the AR Tag, rotate the image to be warped in the opposite direction to get proper orientation
    13) Warp the image on to the frame
    14) Write the frome to the video
    
    Returns
    -------
    None.

    """
    path = os.getcwd()
    video = cv2.VideoCapture(path+"/1tagvideo.mp4")
    
    if(video.isOpened()==False):
        print("Error opening the video. Check the path")
    else:
        print("Video opened")
    # frame = cv2.imread('scenery.jpg')
    testudo = cv2.imread('testudo.png')
    cv2.imwrite("testudo.jpg",testudo)
    testudo = cv2.imread('testudo.jpg')
    testudo = cv2.resize(testudo, (160,160))
    
    kernel = np.ones((3, 3), np.uint8)
    rot = 0
    H_Old = None
    
    video_writer = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Testudo_on_AR.avi',video_writer,16,(1920,1080))
    
    while video.isOpened():
        ret,frame = video.read()
        if ret:
            # print(frame.shape)
            new_testudo = testudo.copy() #copying the original upright position of the testudo
            
            gray_frame = convert_to_binary(frame)
            
            _, gray_frame = cv2.threshold(gray_frame, 180, 255, cv2.THRESH_BINARY)
            
            gray_frame = cv2.medianBlur(gray_frame, 7)
            
            fft_image = scipy.fft.fft2(gray_frame)
            
            fft_shifted = scipy.fft.fftshift(fft_image)
            
            mask = compute_mask(fft_shifted.shape)
            
            fft_masked = fft_shifted * mask
            
            ifft_shifted = scipy.fft.ifftshift(fft_masked)
            
            ifft_image = scipy.fft.ifft2(ifft_shifted)
            
            ifft_image = np.uint8(np.abs(ifft_image))
            
            closed_image = cv2.morphologyEx(ifft_image, cv2.MORPH_CLOSE, kernel)
            
            # cv2.imshow("Edge FFT",closed_image)
            
            ret,H = detect_corners(closed_image, gray_frame,frame)
            if ret is not None:
                cv2.imshow("TAG",ret)
                cv2.waitKey(1)
                H_Old = H  
                # print(ret.shape)
                
                img = warp_perspective(H, ret, 160,160)
                if img is not None:
                    cv2.imshow("WARP",img)
                    cv2.waitKey(1)
                    # cv2.destroyAllWindows()
                    
                    #Decode the tag
                    index,binary = decode_tag(convert_to_binary(img),testudo)
                    if index is not None:
                        if binary==11:
                            if rot != index:
                                rot = index
                    if rot == 2:
                        # print("Rotated by 90")
                        new_testudo = cv2.rotate(new_testudo, cv2.ROTATE_90_CLOCKWISE)
                        
                    elif rot == 0:
                        # print("Rotated by 180")
                        new_testudo = cv2.rotate(new_testudo, cv2.ROTATE_180)
                        
                    elif rot == 1:
                        # print("Rotated by 270")
                        new_testudo = cv2.rotate(new_testudo, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
                    #Warping the testudo on the frame            
                    h_inv = np.linalg.inv(H)
                    for a in range(0,img.shape[1]):
                        # print(img.shape)
                        for b in range(0,img.shape[0]):
                            x_test, y_test, z_test = np.dot(h_inv,[a,b,1])
                            frame[int(y_test/z_test)][int(x_test/z_test)] = new_testudo[a][b]
                    cv2.imshow('Testudo Warp',frame)
                    cv2.waitKey(1)
                    out.write(frame)
                    
            else:
                #Preserving the old Homography matrix in case the corners are not detected
                H = H_Old
                if rot == 2:
                    # print("Rotated by 90")
                    new_testudo = cv2.rotate(new_testudo, cv2.ROTATE_90_CLOCKWISE)
                    
                elif rot == 0:
                    # print("Rotated by 180")
                    new_testudo = cv2.rotate(new_testudo, cv2.ROTATE_180)
                    
                elif rot == 1:
                    # print("Rotated by 270")
                    new_testudo = cv2.rotate(new_testudo, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                #Warping the testudo on the frame
                h_inv = np.linalg.inv(H)
                try:
                    for a in range(0,160):
                        # print(img.shape)
                        for b in range(0,160):
                            x_test, y_test, z_test = np.dot(h_inv,[a,b,1])
                            frame[int(y_test/z_test)][int(x_test/z_test)] = new_testudo[a][b]
                    cv2.imshow('Testudo Warp',frame)
                    cv2.waitKey(1)
                    out.write(frame)
                except:
                    pass
        else:
            break
    print("Done")
    cv2.destroyAllWindows()
    
def main():
    """
    This function calls the pipeline function to run the overall algorithm

    Returns
    -------
    None.

    """
    pipeline() 


if __name__ == '__main__':
    main()