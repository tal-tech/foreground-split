import cv2
from skimage import io, transform, color
import torch
import math
import numpy as np
class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self,flag=0):
        self.flag = flag

    def __call__(self, image):

        tmpImg = np.zeros((image.shape[0],image.shape[1],3))
        image = image/np.max(image)

        if image.shape[2]==1:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
        else:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225


        # change the r,g,b to b,r,g from [0,255] to [0,1]
        #transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        tmpImg = tmpImg.transpose((2, 0, 1))

        return torch.from_numpy(tmpImg.copy()).type(torch.FloatTensor)

class RescaleT(object):

    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self,image):
        img = transform.resize(image,(self.output_size,self.output_size),mode='constant', preserve_range=True)
        return img

def remove_prefix_state_dict(state_dict, prefix="module"):
    """
    remove prefix from the key of pretrained state dict for Data-Parallel
    """
    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key[len(prefix)+1:]] = state_dict[key].float()
    return new_state_dict

def checkImage(image):
    """
    Args:
        image: input image to be checked
    Returns:
        binary image
    Raises:
        RGB image, grayscale image, all-black, and all-white image

    """
    if len(image.shape) > 2:
        print("ERROR: non-binary image (RGB)"); sys.exit();

    smallest = image.min(axis=0).min(axis=0) # lowest pixel value: 0 (black)
    largest  = image.max(axis=0).max(axis=0) # highest pixel value: 1 (white)
    if (smallest == 0 and largest == 0):
        print("ERROR: non-binary image (all black)"); sys.exit()
    elif (smallest == 255 and largest == 255):
        print("ERROR: non-binary image (all white)"); sys.exit()
    elif (smallest > 0 or largest < 255 ):
        print("ERROR: non-binary image (grayscale)"); sys.exit()
    else:
        return True

def generate_trimap(image):
    """
    This function creates a trimap based on simple dilation algorithm
    Inputs [1]: a binary image (black & white only)
    Output    : a trimap
    """
    erosion = 1
    image[image >= 127] = 255
    image[image < 127] = 0
    #Adaptive trimap unknown area generation based the object area in the image
    num_zeros = np.count_nonzero(image==0)
    num_non_zeros = np.count_nonzero(image!=0)
    area_ratio = num_non_zeros*1.0/(num_non_zeros + num_zeros)
    length_ratio = math.sqrt(area_ratio)
    max_edge = max(image.shape)
    length = length_ratio * max_edge
    dialtion_size = int(length/500.0) + 1
    erosion_size = int(length/20.0)
    checkImage(image)
    row    = image.shape[0]
    col    = image.shape[1]
    pixels = 2*dialtion_size + 1      ## Double and plus 1 to have an odd-sized kernel
    kernel = np.ones((pixels,pixels),np.uint8)   ## Pixel of extension I get
    dilation  = cv2.dilate(image, kernel, iterations = 1)
    dilation  = np.where(dilation == 255, 127, dilation)    ## WHITE to GRAY
    dilation  = np.where(dilation != 127, 0, dilation)     ## Smoothing
    erosion = int(erosion)
    erosion_kernel = np.ones((erosion_size,erosion_size), np.uint8)                     ## Design an odd-sized erosion kernel
    remake = cv2.erode(image, erosion_kernel, iterations=erosion)  ## How many erosion do you expect
    remake = np.where(remake > 0, 255, remake)                       ## Any gray-clored pixel becomes white (smoothing)
    # Error-handler to prevent entire foreground annihilation
    if cv2.countNonZero(remake) == 0:
        print("ERROR: foreground has been entirely eroded")
        sys.exit()

    remake = np.where(remake == 255, 255, dilation)
    for i in range(0,row):
        for j in range (0,col):
            if (remake[i,j] != 0 and remake[i,j] != 255):
                remake[i,j] = 127
    return remake