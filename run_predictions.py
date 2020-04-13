import os
import numpy as np
import json
import cv2

def get_local_max(A):
    '''
    Obtains the coordinates of the local maxima in an image
    '''
    # Expand the size of A by 1 on each edge
    A_aug = np.copy(A)
    (y,x) = A.shape
    A_aug = np.hstack((A_aug, np.zeros((y,1))))
    A_aug = np.hstack((np.zeros((y,1)), A_aug))
    A_aug = np.vstack((np.zeros((1,x+2)), A_aug))
    A_aug = np.vstack((A_aug, np.zeros((1,x+2))))
    
    # Compare A values with adjacent 4 values
    maxima = A > A_aug[1:-1,2:]
    maxima = np.logical_and(maxima, A > A_aug[1:-1,:-2])
    maxima = np.logical_and(maxima, A > A_aug[2:,1:-1])
    maxima = np.logical_and(maxima, A > A_aug[:-2,1:-1])
    
    b = np.where(maxima)
        
    return b


def matched_filter_convolution_2d(kernel, img):
    '''
    Convolutes a given kernel kXxkY with a given image, size XxY
    Convolution only applied in regions where kernel fully fits in image
    @img: color image as numpy array, kXxkYx3
    @kernel: color image as numpy array, XxYx3
    
    @return: Outputs a (X-kX)x(Y-kY) numpy array. Each point corresponds to the result
    where the top left corner of the kernel was placed in the original image.
    
    '''
    # Find size of image and kernel
    kX = np.shape(kernel)[0]
    kY = np.shape(kernel)[1]
    X = np.shape(img)[0]
    Y = np.shape(img)[1]
    
    # Convert kernel to 1d array
    kernel_1d = kernel.reshape((kX*kY, 3))
    
    # initialize output
    filtered_img = np.zeros((X-kX+1,Y-kY+1))
    
    # do the convolution
    for x in range(X-kX+1):
        for y in range(Y-kY+1):
            # The relevant portion of the image
            img_extract = img[x:x+kX, y:y+kY, :]
            img_extract_1d = img_extract.reshape((kX*kY), 3)
            # normalize
            img_extract_1d = img_extract_1d/np.linalg.norm(img_extract_1d)
            for c in range(3): # rbg color
                filtered_img[x,y] += np.dot(kernel_1d[:,c], img_extract_1d[:,c])
    
    return filtered_img


def convolute_2d(kernel, img):
    '''
    Convolutes a given kernel kXxkY with a given image, size XxY
    Out of bound regions are given the average value of the in-bounds dot product
    Uses edge extension for edge handling
    @img: XxYx3 color image as numpy array
    @kernel: kXxkY numpy array. Lengthes must be odd
    
    @return: Outputs a (X)x(Y)x3 color image as a numpy array.
    
    '''
    # Edge and half-edge length of kernel
    kX = np.shape(kernel)[0]
    kY = np.shape(kernel)[1]
    kX2 = (np.shape(kernel)[0]-1)/2
    kY2 = (np.shape(kernel)[1]-1)/2
    # Size of image
    X = np.shape(img)[0]
    Y = np.shape(img)[1]
    
    # Extends image by repeating each edge by kX2 and kY2 pixels
    img_ext = np.copy(img)
    
    edge = np.repeat(img_ext[np.newaxis,X-1,:,:], kY2, axis=0)
    img_ext = np.vstack((img_ext, edge))
    
    edge = np.repeat(img_ext[np.newaxis,0,:,:], kY2, axis=0)
    img_ext = np.vstack((edge, img_ext))
    
    edge = np.repeat(img_ext[:,np.newaxis,Y-1,:], kX2, axis=1)
    img_ext = np.hstack((img_ext, edge))
    
    edge = np.repeat(img_ext[:,np.newaxis,0,:], kX2, axis=1)
    img_ext = np.hstack((edge, img_ext))
    
    # Do the convolution
    # Convert kernel to 1d array
    kernel_1d = kernel.reshape(kX*kY)
    
    # initialize output
    filtered_img = np.zeros((X,Y,3))
    
    for x in range(X):        
        for y in range(Y):
            # The relevant portion of the image
            img_extract = img_ext[x:x+kX, y:y+kY, :]
            img_extract_1d = img_extract.reshape((kX*kY), 3)
            for c in range(3): # rbg color
                filtered_img[x,y,c] = np.dot(kernel_1d, img_extract_1d[:,c])
    
    return filtered_img



def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    '''
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    # get kernels
    kernel = cv2.imread(os.path.join(kernel_path,kernel_names[6]))
    # convert to numpy array:
    kernel = np.asarray(kernel)
    
    box_height = kernel.shape[0]
    box_width = kernel.shape[1]
    
    # perform matched filter, single image
    matched_filter = matched_filter_convolution_2d(kernel, I)
    
    thresh = min(matched_filter.max()*0.99, matched_filter.mean()+1.9*matched_filter.std())
    
    # Filter out values that don't meet threshold
    matched_filter[matched_filter<thresh] = thresh
    
    # Find the local maxima
    inds = get_local_max(matched_filter)
    
    for i in range(len(inds[0])):
        tl_row = int(inds[0][i])
        tl_col = int(inds[1][i])
        br_row = tl_row + box_height
        br_col = tl_col + box_width
            
        bounding_boxes.append([tl_row,tl_col,br_row,br_col]) 
    

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes


''' 
Main Code
'''

# set the path to the downloaded data: 
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = '../data/hw01_preds' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

# set kernel path: 
global kernel_path
global kernel_names
version = 'blurred'
kernel_path = '../data/kernels/'+version
# get sorted list of files: 
kernel_names = sorted(os.listdir(kernel_path)) 
# remove any non-JPEG files: 
kernel_names = [f for f in kernel_names if '.jpg' in f] 

preds = {}

for i in range(len(file_names)):
    # read image using PIL:
    I = cv2.imread(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I)
    
    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds.json'),'w') as f:
        json.dump(preds,f)


#plot_boxes(i)

# test on 240, 247, 314, 2, 11



