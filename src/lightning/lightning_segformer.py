import torch
from torch import nn
import pytorch_lightning as pl
from utils.metric import Evaluator
import numpy as np
from src.dataset.uavid_dataset import *
from src.dataset.image_transforms import get_transforms
from torch.utils.data import DataLoader



class SegFormerModule(pl.LightningModule):
    def __init__(
            self, config
    ):
        super().__init__()

        self.config = config
        self.net = config.net
        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)


    def forward(self, pixel_values, labels):
        outputs = self.net(pixel_values=pixel_values, labels=labels)
        return outputs

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def training_step(self, train_batch, batch_idx):
        images, targets = train_batch['img'], train_batch['gt_semantic_seg']
        images, targets = images.type(torch.float32), targets.type(torch.long)

        outputs = self.net(pixel_values=images, labels=targets)
        loss, logits = outputs.loss, outputs.logits
        upsampled_logits = nn.functional.interpolate(logits,
                                                     size=targets.size(dim=1),  # (height, width)
                                                     mode='bilinear',
                                                     align_corners=False)
        pre_mask = upsampled_logits.argmax(dim=1)
        for i in range(targets.shape[0]):
            self.metrics_train.add_batch(targets[i].cpu().numpy(), pre_mask[i].cpu().numpy())
        return {"loss": loss}

    def on_train_epoch_end(self):
        mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
        F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('train:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_train.reset()
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, val_batch, batch_idx):
        images, targets = val_batch['img'], val_batch['gt_semantic_seg']
        images, targets = images.type(torch.float32), targets.type(torch.long)

        outputs = self.net(pixel_values=images, labels=targets)
        loss_val, logits_val = outputs.loss, outputs.logits
        upsampled_logits = nn.functional.interpolate(logits_val,
                                                     size=targets.size(dim=1),  # (height, width)
                                                     mode='bilinear',
                                                     align_corners=False)
        pre_mask = upsampled_logits.argmax(dim=1)
        for i in range(targets.shape[0]):
            self.metrics_val.add_batch(targets[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
        F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('val:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)

        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)
        self.log_dict(iou_value)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def setup(self, stage=None):

        train_dataset = UAVID(data_root=self.config.DATA_DIR + '/train', img_dir='images', mask_dir='masks',
                              mode='train', mosaic_ratio=0.25, transform=get_transforms(train=True),
                              img_size=(1024, 1024))

        val_dataset = UAVID(data_root=self.config.DATA_DIR + '/val', img_dir='images', mask_dir='masks', mode='val',
                            mosaic_ratio=0.0, transform=get_transforms(train=False), img_size=(1024, 1024))

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.config.train_batch_size,
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=True,
                                  drop_last=True)

        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=self.config.val_batch_size,
                                num_workers=4,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False)
        self.train_dataset = train_dataset
        self.valid_dataset = val_dataset
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.val_loader


