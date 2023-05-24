import glob
import os
import numpy as np
import cv2
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="./uavid/uavid_val")
    parser.add_argument("--output-img-dir", default="./val/images")
    parser.add_argument("--output-mask-dir", default="./val/masks")
    parser.add_argument("--split-size-h", type=int, default=1024)
    parser.add_argument("--split-size-w", type=int, default=1024)
    parser.add_argument("--stride-h", type=int, default=1024)
    parser.add_argument("--stride-w", type=int, default=1024)
    return parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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


def image_augment(image, mask):
    image_list = []
    mask_list = []
    image_width, image_height = image.shape[1], image.shape[0]
    mask_width, mask_height = mask.shape[1], mask.shape[0]
    assert image_height == mask_height and image_width == mask_width

    image_list_train = [image]
    mask_list_train = [mask]
    for i in range(len(image_list_train)):
        mask_tmp = rgb2label(mask_list_train[i])
        image_list.append(image_list_train[i])
        mask_list.append(mask_tmp)
    return image_list, mask_list


def padifneeded(image, mask):
    pad = albu.PadIfNeeded(min_height=2160, min_width=4096, position='bottom_right',
                           border_mode=0, value=[0, 0, 0], mask_value=[255, 255, 255])(image=image, mask=mask)
    img_pad, mask_pad = pad['image'], pad['mask']
    assert img_pad.shape[0] == 2048 or img_pad.shape[1] == 4096, print(img_pad.shape)

    return img_pad, mask_pad


def patch_format(inp):
    (input_dir, seq, imgs_output_dir, masks_output_dir, split_size, stride) = inp
    img_paths = glob.glob(os.path.join(input_dir, str(seq), 'Images',  "*.png"))
    mask_paths = glob.glob(os.path.join(input_dir, str(seq), 'Labels', "*.png"))
    for img_path, mask_path in zip(img_paths, mask_paths):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        id = os.path.splitext(os.path.basename(img_path))[0]
        assert img.shape == mask.shape and img.shape[0] == 2160, print(img.shape)
        assert img.shape[1] == 3840 or img.shape[1] == 4096, print(img.shape)
        img, mask = padifneeded(img.copy(), mask.copy())


        # img and mask shape: WxHxC
        image_list, mask_list = image_augment(image=img.copy(), mask=mask.copy())
        assert len(image_list) == len(mask_list)
        for m in range(len(image_list)):
            k = 0
            img = image_list[m]
            mask = mask_list[m]
            img, mask = img[-2048:, -4096:, :], mask[-2048:, -4096:]
            assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]
            for y in range(0, img.shape[0], stride[0]):
                for x in range(0, img.shape[1], stride[1]):
                    img_tile_cut = img[y:y + split_size[0], x:x + split_size[1]]
                    mask_tile_cut = mask[y:y + split_size[0], x:x + split_size[1]]
                    img_tile, mask_tile = img_tile_cut, mask_tile_cut

                    if img_tile.shape[0] == split_size[0] and img_tile.shape[1] == split_size[1] \
                            and mask_tile.shape[0] == split_size[0] and mask_tile.shape[1] == split_size[1]:
                        out_img_path = os.path.join(imgs_output_dir, "{}_{}_{}_{}.png".format(seq, id, m, k))
                        img_tile = cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(out_img_path, img_tile)

                        out_mask_path = os.path.join(masks_output_dir,
                                                     "{}_{}_{}_{}.png".format(seq, id, m, k))
                        cv2.imwrite(out_mask_path, mask_tile)


                    k += 1


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    input_dir = args.input_dir
    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir
    split_size_h = args.split_size_h
    split_size_w = args.split_size_w
    split_size = (split_size_h, split_size_w)
    stride_h = args.stride_h
    stride_w = args.stride_w
    stride = (stride_h, stride_w)
    seqs = os.listdir(input_dir)

    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)

    inp = [(input_dir, seq, imgs_output_dir, masks_output_dir, split_size, stride) for seq in seqs]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(patch_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))


