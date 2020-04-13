import os
import numpy as np
import cv2

'''
Processes kernels
'''

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

# set the path to kernel: 
global kernel_path
global kernel_names
kernel_path = '../data/kernels'
# get sorted list of files: 
kernel_names = sorted(os.listdir(kernel_path)) 
# remove any non-JPEG files: 
kernel_names = [f for f in kernel_names if '.jpg' in f] 

# apply gaussian blur
kern2 = np.ones((5,5)) * 1./25

version = '/blurred'
os.makedirs(kernel_path+version,exist_ok=True) # create directory if needed 
for i in range(len(kernel_names)):
    # read image using PIL:
    k = cv2.imread(os.path.join(kernel_path,kernel_names[i]))
    # convert to numpy array:
    k = np.asarray(k)
    k = convolute_2d(kern2, k)
    
    cv2.imshow("edge", k);
    cv2.imwrite(kernel_path+version+version+'_'+kernel_names[i], k);
    cv2.waitKey(0);    
    cv2.destroyAllWindows()
    

