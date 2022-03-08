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

    Parameters
    ----------
    p1 : List
        Corner points of the 160x160 image.
    p2 : List
        Corner points of the AR Tag.
    
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
        U, S, Vh = np.linalg.svd(A) #Using SVD to find the Homography
        L = Vh[-1,:] / Vh[-1,-1] #Normalizing the homography with the last element 
        H = L.reshape(3, 3) # Reshaping to 3x3 matrix
        return(H)
    except:
        pass

def projection_matrix(H, K):
    """
    This function computes the projection matrix P using the equations given in the supplementary document for homography

    Parameters
    ----------
    H : 2 D Array
        Homography Matrix.
    K : 2 D Array
        Camera Intrinsic Matrix.

    Returns
    -------
    P : 2 D Array
        Projection Matrix.

    """
    
    B_tilda = np.matmul(np.linalg.inv(K),H)
    # check if determinant is greater than 0 or < 0. If it is negative, then multiply the matrix with -1
    det = np.linalg.det(B_tilda)
    # print("Det",det,"And",B_tilda)
    if det<0:
        B_tilda = -1*B_tilda
        # print("Negated")
    col1 = B_tilda[:,0]
    col2 = B_tilda[:,1]
    
    #calculating lamda 1/(||b1|| + ||b2^2||)
    lamda = 2 / (np.linalg.norm(col1)+np.linalg.norm(col2))
    
    B = lamda * B_tilda #Multiplying with lamda to get the B Matrix
    

    row1 = B[:, 0]
    row2 = B[:, 1]                      #extract rotation and translation vectors
    row3 = np.cross(row1, row2) #Because the 3rd row is orthogonal to row1 and row2
    
    T = B[:, 2] #Translation Vector
    Rt = np.array([row1, row2, row3, T]).T #Rotation + Translation 
    P = np.dot(K,Rt)   #Projection Matrix
    P = P/P[2,3] #Normalizing with the last element
    # print("P Matrix: ",P)
    return P

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
        List
            Corner points of the AR Tag.
    
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
                H = find_homography([(0,0),(0,160),(160,0),(160,160)],p)
                
                return frame,H,p
            else:
                return None,None,None
        else:
            return None,None,None
            

                
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
    2) Next the Intrinsic Matrix K is initialized
    3) Codec and output writer objects are initialized.
    4) Frame is read
    5) It is converted to gray scale image
    6) Threshold the gray image
    7) Apply Median Blur to remove Salt and Pepper Noise and smoothen the image
    8) Apply the FFT, FFT Shift, Mask, IFFT Shift and IFFT to get the edges from the image
    9) Use the morphology close operator to get better output of the edges
    10) Detect the corners and compute the homography
    11) Compute the projection Matrix
    12) Get the top coordinates of the cube (As the bottom coordinates are the AR Tag corners, there is no need to compute the bottom coordinates)
    13) Draw the corresponding lines on the frame
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
    
    K = np.array([[1346.100595,0,932.1633975],
                 [0,1355.933136,654.8986796],
                 [0,0,1]])
    
    kernel = np.ones((3, 3), np.uint8)
    
    H_Old = None
    
    video_writer = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Cube_on_AR.avi',video_writer,16,(1920,1080))
    
    while video.isOpened():
        ret,frame = video.read()
        if ret:
            # print(frame.shape)
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
            
            ret,H,p = detect_corners(closed_image, gray_frame,frame)
            if ret is not None:
                # cv2.imshow("TAG",ret)
                # cv2.waitKey(1)
                
                H_Old = H
                
                try:
                    P = projection_matrix(H, K)
                    x1,y1,z1 = np.dot(P,[0,0,-160,1])
                    x2,y2,z2 = np.dot(P,[0,160,-160,1])
                    x3,y3,z3 = np.dot(P,[160,0,-160,1])
                    x4,y4,z4 = np.dot(P,[160,160,-160,1])
                    
                    xc1,yc1,zc1 = np.dot(P,[0,0,0,1])
                    # print("AR Edge: ",np.sqrt(np.power(p[0][0]-p[1][0] ,2) + np.power(p[0][1]-p[1][1],2)))
                    # print("Cube Edge: ",np.sqrt(np.power(p[0][0]-int(x1/z1),2) + np.power(p[0][1]-int(y1/z1),2)))
                    # print(int(x1/z1),int(y1/z1))
                    cv2.line(frame,(int(x1/z1),int(y1/z1)),tuple(p[0]), (255,0,0), 4)
                    cv2.line(frame,(int(x2/z2),int(y2/z2)),tuple(p[1]), (255,0,0), 4)
                    cv2.line(frame,(int(x3/z3),int(y3/z3)),tuple(p[2]), (255,0,0), 4)
                    cv2.line(frame,(int(x4/z4),int(y4/z4)),tuple(p[3]), (255,0,0), 4)
                    
                    cv2.line(frame, (int(x3/z3),int(y3/z3)), (int(x1/z1),int(y1/z1)), (0,0,255),4)
                    cv2.line(frame, (int(x3/z3),int(y3/z3)), (int(x4/z4),int(y4/z4)), (0,0,255),4)
                    cv2.line(frame, (int(x2/z2),int(y2/z2)), (int(x1/z1),int(y1/z1)), (0,0,255),4)
                    cv2.line(frame, (int(x2/z2),int(y2/z2)), (int(x4/z4),int(y4/z4)), (0,0,255),4)
                    
                except:
                    pass
            else:
                H = H_Old
            
            cv2.imshow("CUBE VIDEO", frame)
            out.write(frame)
            cv2.waitKey(1)
        else:
            break
    print("Done")
    cv2.destroyAllWindows()
    
def main():
    """
    This function calls the pipeline function to run the overall pipeline

    Returns
    -------
    None.

    """
    pipeline() 


if __name__ == '__main__':
    main()
