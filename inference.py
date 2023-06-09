import argparse
from pathlib import Path
import glob
from src.lightning.lightning_fpn import FPNetModule
from src.lightning.lightning_segformer import SegFormerModule
from src.lightning.lightning_unetformer import UnetFormerModule
import ttach as tta
import albumentations as albu
from utils.cfg import py2cfg
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.metric import Evaluator
from utils.tools import *


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--image_path", type=str, help="Path to  UAVid test/val",required=True)
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config",nargs='+')
    arg("-w", "--weights", required=True, help="List of weights for ensemble learning ",  nargs='+')
    arg("-o", "--output_path", type=Path, help="Path to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default="lr", choices=[None, "lr"])
    arg("-ph", "--patch-height", help="height of patch size", type=int, default=1024)
    arg("-pw", "--patch-width", help="width of patch size", type=int, default=1024)
    arg("-b", "--batch-size", help="batch size", type=int, default=1)
    parser.add_argument('-m', '--mask',help="Existence of labels ",action='store_true')
    return parser.parse_args()


def get_img_padded(image, patch_size):
    oh, ow = image.shape[0], image.shape[1]
    rh, rw = oh % patch_size[0], ow % patch_size[1]

    width_pad = 0 if rw == 0 else patch_size[1] - rw
    height_pad = 0 if rh == 0 else patch_size[0] - rh
    h, w = oh + height_pad, ow + width_pad

    pad = albu.PadIfNeeded(min_height=h, min_width=w, border_mode=0,
                           position='bottom_right', value=[0, 0, 0])(image=image)
    img_pad = pad['image']
    return img_pad, height_pad, width_pad


class InferenceDataset(Dataset):
    def __init__(self, tile_list=None, transform=albu.Normalize(),img_dir='images', mask_dir='masks'):
        self.tile_list = tile_list
        self.transform = transform
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __getitem__(self, index):
        img = self.tile_list[index]
        img_id = index
        aug = self.transform(image=img)
        img = aug['image']
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        results = dict(img_id=img_id, img=img)
        return results

    def __len__(self):
        return len(self.tile_list)


def make_dataset_for_one_huge_image(img_path,mask_path, patch_size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = rgb2label(mask)
    tile_list = []
    image_pad, height_pad, width_pad = get_img_padded(img.copy(), patch_size)


    output_height, output_width = image_pad.shape[0], image_pad.shape[1]

    for x in range(0, output_height, patch_size[0]):
        for y in range(0, output_width, patch_size[1]):
            image_tile = image_pad[x:x+patch_size[0], y:y+patch_size[1]]
            tile_list.append(image_tile)

    dataset = InferenceDataset(tile_list=tile_list)
    return dataset, width_pad, height_pad, output_width, output_height, image_pad, img.shape,mask,img

def get_raw_pred(model,input):
    if isinstance(model,SegFormerModule):
        output = model(input['img'].cuda("cuda:0"), None)
        raw_prediction = nn.functional.interpolate(output.logits,
                                                     size=input['img'].size(dim=2),  # (height, width)
                                                     mode='bilinear',
                                                     align_corners=False)

    elif isinstance(model.model,UnetFormerModule):
        raw_prediction = model(input['img'].cuda("cuda:0"))
        raw_prediction = nn.Softmax(dim=1)(raw_prediction)
    elif isinstance(model.model,FPNetModule):
        raw_prediction = model(input['img'].cuda("cuda:0"))
        raw_prediction = nn.Softmax(dim=1)(raw_prediction)
    return raw_prediction

def main():
    args = get_args()
    seed(42)
    seqs = os.listdir(args.image_path)

    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
    # print(img_paths)
    patch_size = (args.patch_height, args.patch_width)
    models = []
    for config in args.config_path:
        cfg = py2cfg(config)
        if cfg.net_name=="Unetformer":
            model = UnetFormerModule.load_from_checkpoint(os.path.join(cfg.weights_path, cfg.test_weights_name+'.ckpt'), config=cfg)
            model = tta.SegmentationTTAWrapper(model, transforms)
        elif cfg.net_name=="Segformer":
            model = SegFormerModule.load_from_checkpoint(os.path.join(cfg.weights_path, cfg.test_weights_name+'.ckpt'), config=cfg)
            #model = tta.SegmentationTTAWrapper(model, transforms)
        elif cfg.net_name=="FPN":
            model = FPNetModule.load_from_checkpoint(os.path.join(cfg.weights_path, cfg.test_weights_name+'.ckpt'), config=cfg)
            model = tta.SegmentationTTAWrapper(model, transforms)


        model.cuda("cuda:0")
        model.eval()
        models.append(model)
    for seq in seqs:
        img_paths = []
        mask_paths = []
        output_path = os.path.join(args.output_path, str(seq), 'Labels')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for ext in ('*.tif', '*.png', '*.jpg'):
            img_paths.extend(glob.glob(os.path.join(args.image_path, str(seq), 'Images', ext)))
            if args.mask:
                mask_paths.extend(glob.glob(os.path.join(args.image_path, str(seq), 'Labels', ext)))
        img_paths.sort()
        if args.mask:
            mask_paths.sort()
            eval = Evaluator(num_class=num_classes)

        if len(mask_paths)==0:
            mask_paths = img_paths


        for img_path,mask_path in zip(img_paths,mask_paths):
            img_name = img_path.split('/')[-1]

            dataset, width_pad, height_pad, output_width, output_height, img_pad, img_shape,gt_mask,img = \
                make_dataset_for_one_huge_image(img_path,mask_path, patch_size)

            output_mask = np.zeros(shape=(output_height, output_width), dtype=np.uint8)
            output_tiles = []
            preds = []
            weights= [float(i) for i in args.weights]
            k = 0
            with torch.no_grad():
                dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                        drop_last=False, shuffle=False)
                for input in tqdm(dataloader):
                    # raw_prediction NxCxHxW
                    for model in models:
                        raw_prediction = get_raw_pred(model,input)
                        preds.append(raw_prediction.cpu().numpy())
                    weighted_preds = np.tensordot(preds, weights, axes=((0), (0)))
                    # input_images['features'] NxCxHxW C=3
                    prediction = torch.from_numpy(weighted_preds).argmax(dim=1)

                    image_ids = input['img_id']

                    for i in range(prediction.shape[0]):
                        raw_mask = prediction[i].cpu().numpy()
                        mask = raw_mask
                        output_tiles.append((mask, image_ids[i].cpu().numpy()))

                for m in range(0, output_height, patch_size[0]):
                    for n in range(0, output_width, patch_size[1]):
                        output_mask[m:m + patch_size[0], n:n + patch_size[1]] = output_tiles[k][0]
                        k = k + 1

                output_mask = output_mask[-img_shape[0]:, -img_shape[1]:]

                output_mask_rgb = uavid2rgb(output_mask)

                output_mask_rgb = 0.5 * output_mask_rgb + 0.5 * img
                cv2.imwrite(os.path.join(output_path, img_name), output_mask_rgb)
                if args.mask:
                    eval.add_batch(gt_mask, output_mask)

        if args.mask:
            mIoU = np.nanmean(eval.Intersection_over_Union())
            F1 = np.nanmean(eval.F1())

            OA = np.nanmean(eval.OA())
            iou_per_class = eval.Intersection_over_Union()
            eval_value = {'mIoU': mIoU,
                          'F1': F1,
                          'OA': OA}


            iou_value = {}
            for class_name, iou in zip(cfg.classes, iou_per_class):
                iou_value[class_name] = iou
            print('Metrics for model is :', eval_value)
            print('And for classes', iou_value)


if __name__ == "__main__":
    main()
