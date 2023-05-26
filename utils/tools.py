import random
import os
import torch
import numpy as np
import cv2


Building = np.array([128, 0, 0])  # label 0 Static
Road = np.array([128, 64, 128]) # label 1 Road
Tree = np.array([0, 128, 0]) # label 0 Static
LowVeg = np.array([128, 128, 0]) # label 0 Static
Moving_Car = np.array([64, 0, 128]) # label 2 Dynamic
Static_Car = np.array([192, 0, 192]) # label 0 Static
Human = np.array([64, 64, 0]) # label 2 Dynamic
Clutter = np.array([0, 0, 0]) # label 3 Clutter
Boundary = np.array([255, 255, 255]) # label 255

num_classes = 4

def seed(seed_num):
    random.seed(seed_num)
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def rgb2label(label):
    label_seg = np.zeros(label.shape[:2], dtype=np.uint8)
    label_seg[np.all(label == Building, axis=-1)] = 0 #Static
    label_seg[np.all(label == Road, axis=-1)] = 1 #Road
    label_seg[np.all(label == Tree, axis=-1)] = 0 #Static
    label_seg[np.all(label == LowVeg, axis=-1)] = 0 #Static
    label_seg[np.all(label == Moving_Car, axis=-1)] = 2 #Dynamic
    label_seg[np.all(label == Static_Car, axis=-1)] = 0 #Static
    label_seg[np.all(label == Human, axis=-1)] = 2 #Dynamic
    label_seg[np.all(label == Clutter, axis=-1)] = 3 #Clutter
    label_seg[np.all(label == Boundary, axis=-1)] = 255 #Boundary
    return label_seg

def uavid2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 0, 0] #Static
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 128, 0] #Road
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 0, 128] #Dynamic
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 0] #Clutter
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb