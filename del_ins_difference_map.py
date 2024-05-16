import torch
import numpy as np

def make_grids(h, w):
    shifts_x = torch.arange(
        0, w, 1)
    shifts_y = torch.arange(
        0, h, 1)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    grids = torch.stack((shift_x, shift_y), dim=1)
    return grids

def random_pixel(image, poses):
    # adjust the size of perturbation each step based on the size of object size 
    h,w, _ = image.shape 
    random_patch = torch.rand(len(poses), 3) * 255.
    xs, ys = zip(*poses)
    image[ys, xs, :] = random_patch
    return image

def delection_process(image, heatmap, area, L, step):
    '''
    delete the image pixels based on values of heat map, high attention first
    L: total deletion steps
    step: the current step
    area: the area of the object bbox
    '''
    image_array = np.array(image).copy()
    image_array.setflags(write=1)
    h, w = heatmap.shape
    grids = make_grids(h, w)
    order = np.argsort(-heatmap.reshape(-1))
    pixel_once = max(1, int(area/L))
    image_array = random_pixel(image_array, grids[order[:(step+1)*pixel_once]])
    return torch.tensor(image_array)

def add_pixel(image, input_img, poses):
    xs, ys = zip(*poses)
    input_img[ys, xs,:] = image[ys, xs, :]
    return input_img

def insertion_process(image, heatmap, area, L, step):
    '''
    insert the image pixels into an empty image based on values of heat map, high attention first
    L: total deletion steps
    step: the current step
    area: the area of the object bbox
    '''
    image_array = np.array(image).copy()
    h, w = heatmap.shape
    grids = make_grids(h, w)
    order = np.argsort(-heatmap.reshape(-1))
    pixel_once = max(1, int(area/L))
    ins_img = np.zeros(image_array.shape)
    ins_img = add_pixel(image_array, ins_img, grids[order[:(step+1)*pixel_once]])
    return torch.tensor(ins_img)


# L = 100
# gap = 10
# for step in range(10, L+1, gap):
#     del_img = delection_process(input_img, heatmap, area, L, step)
#     det_results_del = detector(del_img)
    
#     ins_img = insertion_process(input_img, heatmap, area, L, step)
#     det_results_ins = detector(ins_img)

