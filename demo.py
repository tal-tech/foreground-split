import base64
import cv2
import torch
import toml
import numpy as np

from torch.autograd import Variable
import torch.nn.functional as F
import os
import sys
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path))
from model import U2NET # full size version 173.6 MB
from model import RefinementModule
#from model import CONFIG
from skimage import io, transform
import util
import utils
from utils import CONFIG
from util import RescaleT
from util import ToTensorLab
from torchvision import transforms
import networks

class Sod_Model:
    """
    Initialize the salient object detection model.
    model_path specifies the folder in which the model is saved.
    """
    def __init__(self, input_size, prob, model_path=None):
        self.input_size = input_size
        self.prob = prob
        self.model = U2NET(3,1)
        if torch.cuda.is_available():
            self.model.cuda()
        model_dict = torch.load(model_path)
        # Remove module. from dataparallel
        model_dict = {key.replace("module.", ""): value for key, value in model_dict.items()}
        self.model.load_state_dict(model_dict)
        self.model.eval()

    def inference(self, image):
        with torch.no_grad():
            #Resize the input image to the size of the desired size of the model, and normalize the image
            im_transform = transforms.Compose([RescaleT(self.input_size),ToTensorLab(flag=0)])
            input_image = im_transform(image)
            input_image = input_image.type(torch.FloatTensor)
            #Adding batch size 1 for the input image
            input_image = input_image.unsqueeze(0)
            if torch.cuda.is_available():
                input_image = Variable(input_image.cuda())
            else:
                input_image = Variable(input_image)
            d1,d2,d3,d4,d5,d6,d7= self.model(input_image)
            # normalization
            pred = d1[:,0,:,:]
            pred = self.normPRED(pred)
            mask = self.get_sod_mask(image, pred, self.prob)
            del d1,d2,d3,d4,d5,d6,d7
            return mask


    @staticmethod
    # normalize the predicted SOD probability map
    def normPRED(d):
        ma = torch.max(d)
        mi = torch.min(d)
        dn = (d-mi)/(ma-mi)
        return dn

    @staticmethod
    # get sod output mask
    def get_sod_mask(image, pred, prob):
        pred = pred.squeeze().cpu().data.numpy()
        pred = cv2.resize(pred, (image.shape[1],image.shape[0]))
        pred = np.expand_dims(pred, axis=2)
        pred[pred > prob] = 1
        pred[pred <= prob] = 0
        pred = pred * 255
        return pred


class CascadePSP:
    """
    Initialize the Cascade PSP model.
    model_path specifies the folder in which the model is saved.
    """
    def __init__(self, model_path=None):
        self.model = RefinementModule()
        if torch.cuda.is_available():
            self.model.cuda()
        model_dict = torch.load(model_path)
        # Remove module. from dataparallel
        model_dict = {key.replace("module.", ""): value for key, value in model_dict.items()}
        self.model.load_state_dict(model_dict)
        self.model.eval()

        # Input transformation methods
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.seg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5],
                std=[0.5]
            ),
       ])

    def inference(self, image, mask, fast=False, L=900):
        with torch.no_grad():
            """
            Refines an input segmentation mask of the image.

            image should be of size [H, W, 3]. Range 0~255.
            Mask should be of size [H, W] or [H, W, 1]. Range 0~255. We will make the mask binary by thresholding at 127.
            Fast mode - Use the global step only. Default: False. The speedup is more significant for high resolution images.
            L - Hyperparameter. Setting a lower value reduces memory usage. In fast mode, a lower L will make it runs faster as well.
            """
            image = self.im_transform(image).unsqueeze(0).cuda()
            mask = self.seg_transform((mask>127).astype(np.uint8)*255).unsqueeze(0).cuda()
            if len(mask.shape) < 4:
                mask = mask.unsqueeze(0)

            if fast:
                output = self.process_im_single_pass(image, mask, L)
            else:
                output = self.process_high_res_im(image, mask, L)

            return (output[0,0].cpu().numpy()*255).astype('uint8')

    def process_high_res_im(self, im, seg, L=900):

        stride = L//2

        _, _, h, w = seg.shape

        """
        Global Step
        """
        if max(h, w) > L:
            im_small = self.resize_max_side(im, L, 'area')
            seg_small = self.resize_max_side(seg, L, 'area')
        elif max(h, w) < L:
            im_small = self.resize_max_side(im, L, 'bicubic')
            seg_small = self.resize_max_side(seg, L, 'bilinear')
        else:
            im_small = im
            seg_small = seg

        images = self.safe_forward(self.model, im_small, seg_small)

        pred_224 = images['pred_224']
        pred_56 = images['pred_56_2']
        
        """
        Local step
        """

        for new_size in [max(h, w)]:
            im_small = self.resize_max_side(im, new_size, 'area')
            seg_small = self.resize_max_side(seg, new_size, 'area')
            _, _, h, w = seg_small.shape

            combined_224 = torch.zeros_like(seg_small)
            combined_weight = torch.zeros_like(seg_small)

            r_pred_224 = (F.interpolate(pred_224, size=(h, w), mode='bilinear', align_corners=False)>0.5).float()*2-1
            r_pred_56 = F.interpolate(pred_56, size=(h, w), mode='bilinear', align_corners=False)*2-1

            padding = 16
            step_size = stride - padding*2
            step_len  = L

            used_start_idx = {}
            for x_idx in range((w)//step_size+1):
                for y_idx in range((h)//step_size+1):

                    start_x = x_idx * step_size
                    start_y = y_idx * step_size
                    end_x = start_x + step_len
                    end_y = start_y + step_len

                    # Shift when required
                    if end_y > h:
                        end_y = h
                        start_y = h - step_len
                    if end_x > w:
                        end_x = w
                        start_x = w - step_len

                    # Bound x/y range
                    start_x = max(0, start_x)
                    start_y = max(0, start_y)
                    end_x = min(w, end_x)
                    end_y = min(h, end_y)

                    # The same crop might appear twice due to bounding/shifting
                    start_idx = start_y*w + start_x
                    if start_idx in used_start_idx:
                        continue
                    else:
                        used_start_idx[start_idx] = True
                    
                    # Take crop
                    im_part = im_small[:,:,start_y:end_y, start_x:end_x]
                    seg_224_part = r_pred_224[:,:,start_y:end_y, start_x:end_x]
                    seg_56_part = r_pred_56[:,:,start_y:end_y, start_x:end_x]

                    # Skip when it is not an interesting crop anyway
                    seg_part_norm = (seg_224_part>0).float()
                    high_thres = 0.9
                    low_thres = 0.1
                    if (seg_part_norm.mean() > high_thres) or (seg_part_norm.mean() < low_thres):
                        continue
                    grid_images = self.safe_forward(self.model, im_part, seg_224_part, seg_56_part)
                    grid_pred_224 = grid_images['pred_224']

                    # Padding
                    pred_sx = pred_sy = 0
                    pred_ex = step_len
                    pred_ey = step_len

                    if start_x != 0:
                        start_x += padding
                        pred_sx += padding
                    if start_y != 0:
                        start_y += padding
                        pred_sy += padding
                    if end_x != w:
                        end_x -= padding
                        pred_ex -= padding
                    if end_y != h:
                        end_y -= padding
                        pred_ey -= padding

                    combined_224[:,:,start_y:end_y, start_x:end_x] += grid_pred_224[:,:,pred_sy:pred_ey,pred_sx:pred_ex]

                    del grid_pred_224

                    # Used for averaging
                    combined_weight[:,:,start_y:end_y, start_x:end_x] += 1

            # Final full resolution output
            seg_norm = (r_pred_224/2+0.5)
            pred_224 = combined_224 / combined_weight
            pred_224 = torch.where(combined_weight==0, seg_norm, pred_224)

        _, _, h, w = seg.shape
        images = {}
        images['pred_224'] = F.interpolate(pred_224, size=(h, w), mode='bilinear', align_corners=True)

        return images['pred_224']

    def process_im_single_pass(self, im, seg, L=900):
        """
        A single pass version, aka global step only.
        """

        _, _, h, w = im.shape
        if max(h, w) < L:
            im = self.resize_max_side(im, L, 'bicubic')
            seg = self.resize_max_side(seg, L, 'bilinear')

        if max(h, w) > L:
            im = self.resize_max_side(im, L, 'area')
            seg = self.resize_max_side(seg, L, 'area')

        images = self.safe_forward(self.model, im, seg)

        if max(h, w) < L:
            images['pred_224'] = F.interpolate(images['pred_224'], size=(h, w), mode='area')
        elif max(h, w) > L:
            images['pred_224'] = F.interpolate(images['pred_224'], size=(h, w), mode='bilinear', align_corners=True)

        return images['pred_224']

    @staticmethod
    def resize_max_side(im, size, method):
        h, w = im.shape[-2:]
        max_side = max(h, w)
        ratio = size / max_side
        if method in ['bilinear', 'bicubic']:
            return F.interpolate(im, scale_factor=ratio, mode=method, align_corners=False)
        else:
            return F.interpolate(im, scale_factor=ratio, mode=method)

    @staticmethod
    def safe_forward(model, im, seg, inter_s8=None, inter_s4=None):
        """
        Slightly pads the input image such that its length is a multiple of 8
        """
        b, _, ph, pw = seg.shape
        if (ph % 8 != 0) or (pw % 8 != 0):
            newH = ((ph//8+1)*8)
            newW = ((pw//8+1)*8)
            p_im = torch.zeros(b, 3, newH, newW, device=im.device)
            p_seg = torch.zeros(b, 1, newH, newW, device=im.device) - 1

            p_im[:,:,0:ph,0:pw] = im
            p_seg[:,:,0:ph,0:pw] = seg
            im = p_im
            seg = p_seg

            if inter_s8 is not None:
                p_inter_s8 = torch.zeros(b, 1, newH, newW, device=im.device) - 1
                p_inter_s8[:,:,0:ph,0:pw] = inter_s8
                inter_s8 = p_inter_s8
            if inter_s4 is not None:
                p_inter_s4 = torch.zeros(b, 1, newH, newW, device=im.device) - 1
                p_inter_s4[:,:,0:ph,0:pw] = inter_s4
                inter_s4 = p_inter_s4

        images = model(im, seg, inter_s8, inter_s4)
        return_im = {}

        for key in ['pred_224', 'pred_28_3', 'pred_56_2']:
            return_im[key] = images[key][:,:,0:ph,0:pw]
        del images

        return return_im


class Matting_Model:
    """
    Initialize the matting model.
    model_path specifies the folder in which the model is saved.
    """
    def __init__(self, model_path=None, config_path='config/gca-dist-all-data.toml'):
        config_path = os.path.join(base_path, 'config/gca-dist-all-data.toml')
        with open(config_path) as f:
            utils.load_config(toml.load(f))
            # Check if toml config file is loaded
        if CONFIG.is_default:
            raise ValueError("No .toml config loaded.")
        self.model = networks.get_generator(encoder=CONFIG.model.arch.encoder, \
                                            decoder=CONFIG.model.arch.decoder)
        if torch.cuda.is_available():
            self.model.cuda()
        model_dict = torch.load(model_path)
        # Remove module. from dataparallel
        self.model.load_state_dict(utils.remove_prefix_state_dict(model_dict['state_dict']), strict=True)
        self.model.eval()

    def inference(self, image, mask):
        hh, ww = image.shape[:2]
        trimap = util.generate_trimap(mask)
        #trimap = np.expand_dims(trimap,axis=2)
        #print(trimap.shape)
        #print(image.shape)
        image_dict = self.generator_tensor_dict(image, trimap)
        with torch.no_grad():
            image, trimap = image_dict['image'], image_dict['trimap']
            #print(image.shape)
            #print(trimap.shape)
            alpha_shape = image_dict['alpha_shape']
            image = image.cuda()
            trimap = trimap.cuda()
            alpha_pred, info_dict = self.model(image, trimap)

            if CONFIG.model.trimap_channel == 3:
                trimap_argmax = trimap.argmax(dim=1, keepdim=True)

            alpha_pred[trimap_argmax == 2] = 1
            alpha_pred[trimap_argmax == 0] = 0

            h, w = alpha_shape
            pred = alpha_pred[0, 0, ...].data.cpu().numpy()*255
            pred = pred.astype(np.uint8)
            pred = pred[32:h+32, 32:w+32]
            pred = cv2.resize(pred, dsize=(ww,hh), interpolation=cv2.INTER_LANCZOS4)
            pred = np.expand_dims(pred, axis = 2)
            return pred

    def generator_tensor_dict(self, image, trimap):
        # resize inputs
        image = self.resize_input(image)
        trimap = self.resize_input(trimap)
        sample = {'image': image, 'trimap': trimap, 'alpha_shape': trimap.shape}

        # reshape
        h, w = sample["alpha_shape"]
        if h % 32 == 0 and w % 32 == 0:
            padded_image = np.pad(sample['image'], ((32,32), (32, 32), (0,0)), mode="reflect")
            padded_trimap = np.pad(sample['trimap'], ((32,32), (32, 32)), mode="reflect")
            sample['image'] = padded_image
            sample['trimap'] = padded_trimap
        else:
            target_h = 32 * ((h - 1) // 32 + 1)
            target_w = 32 * ((w - 1) // 32 + 1)
            pad_h = target_h - h
            pad_w = target_w - w
            padded_image = np.pad(sample['image'], ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
            padded_trimap = np.pad(sample['trimap'], ((32,pad_h+32), (32, pad_w+32)), mode="reflect")
            sample['image'] = padded_image
            sample['trimap'] = padded_trimap

        # ImageNet mean & std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        # convert BGR images to RGB
        image, trimap = sample['image'][:,:,::-1], sample['trimap']
        # swap color axis
        image = image.transpose((2, 0, 1)).astype(np.float32)
        trimap[trimap < 85] = 0
        trimap[trimap >= 170] = 2
        trimap[trimap >= 85] = 1
        # normalize image
        image /= 255.


        # to tensor
        sample['image'], sample['trimap'] = torch.from_numpy(image), torch.from_numpy(trimap).to(torch.long)
        sample['image'] = sample['image'].sub_(mean).div_(std)

        if CONFIG.model.trimap_channel == 3:
            sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2, 0, 1).float()
        elif CONFIG.model.trimap_channel == 1:
            sample['trimap'] = sample['trimap'][None, ...].float()
        else:
            raise NotImplementedError("CONFIG.model.trimap_channel can only be 3 or 1")

        # add first channel
        sample['image'], sample['trimap'] = sample['image'][None, ...], sample['trimap'][None, ...]

        return sample

    @staticmethod
    def resize_input(image, target_size=1500, interpolation_method=cv2.INTER_LANCZOS4):
        h_ori, w_ori = image.shape[:2]
        if h_ori >= w_ori:
            resize_scale = target_size / h_ori
        if w_ori > h_ori:
            resize_scale = target_size / w_ori
        image = cv2.resize(image, dsize=(int(w_ori * resize_scale), int(h_ori * resize_scale)), \
                          interpolation=interpolation_method)
        return image


class RemoveBackground:
    #Loading various models into GPU
    def __init__(self, input_size, prob, sod_model_path, cascadepsp_model_path, matting_model_path):
        #load sod model
        self.sod_model = Sod_Model(input_size, prob, sod_model_path)
        #load cascade psp model
        self.cascadepsp= CascadePSP(cascadepsp_model_path)
        #load matting model
        self.matting_model = Matting_Model(matting_model_path)

    def inference(self, image_bgr):
        image_rgb = image_bgr[...,::-1]
        mask = self.sod_model.inference(image_rgb)
        refined_mask = self.cascadepsp.inference(image_bgr, mask)
        alpha = self.matting_model.inference(image_bgr, refined_mask)
        b,g,r = cv2.split(image_bgr)
        new_image = cv2.merge([b, g, r, alpha])
        _, im_arr = cv2.imencode('.png', new_image)  # im_arr: image in Numpy one-dim array format.
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes)
        return im_b64


if __name__ == '__main__':
    print('Starting removing the background from the input image')
    # The input image should be in RGB order and if it is a greyscale image it should have 3 dimensions
    image_cv2 = cv2.imread('test.jpg')
    #image = io.imread('test.jpg')
    # image = image_cv2[...,::-1]
    # sod_model = Sod_Model(320,0.5,'u2net_old.pth')
    # mask = sod_model.inference(image)
    # #cv2.imwrite('test_mask_new_io.png', mask)

    # cascadepsp_model = CascadePSP('cascadepsp')
    # refined_mask = cascadepsp_model.inference(image_cv2, mask)

    # matting_model = Matting_Model('gca-dist-all-data.pth')
    # alpha = matting_model.inference(image_cv2, refined_mask)
    # cv2.imwrite('alpha.png', alpha)

    removebg = RemoveBackground(320, 0.5, 'u2net_old.pth', 'cascadepsp', 'gca-dist-all-data.pth')
    result_img = removebg.inference(image_cv2)
    #cv2.imwrite('result.png', result_img)
    # grey_bg = np.ones_like(image_cv2)*127
    # alpha = alpha/255.
    # result = image_cv2 * alpha + (1-alpha)*grey_bg
    # cv2.imwrite('result_greybg.png', result)