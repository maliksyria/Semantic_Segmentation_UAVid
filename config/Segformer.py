from catalyst import utils
from transformers import SegformerForSemanticSegmentation
import os
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from src.dataset.image_transforms import get_transforms
from src.dataset.uavid_dataset import *
from datetime import datetime


num_classes = len(CLASSES)
classes = CLASSES

BASE_DIR = os.getcwd()
MODELS_DIR = f"{BASE_DIR}/results"
LOGS_DIR = f"{BASE_DIR}/logs"
DATA_DIR = f"{BASE_DIR}/uavid"

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H:%M")
experiment_name = "Segformer_{}".format(dt_string)


# training hparam
max_epoch = 40
train_batch_size = 6
val_batch_size = 6
accumulate_bs = 8

lr = 6e-5 * 8
# scheduler
step = [20, 45, 90]
gamma = 0.2

net_name = "Segformer"
weights_name = "segformer-e{}".format(max_epoch)
weights_path = "model_weights/uavid/{}".format(weights_name)
test_weights_name = "last"

# For callbacks
log_name = 'uavid/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None #In case of resume training - the path for the pretrained models weight
gpus = 'auto'  # auto means default setting or gpu ids:[0] or gpu nums: 2
enable_checkpointing = None  # whether continue training with the checkpoint, default None


id2label = {0: 0, 1: 1, 2: 2, 3: 3}
label2id = {v: k for k, v in id2label.items()}

net = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
        )


# define the dataloader

train_dataset = UAVID(data_root=DATA_DIR+'/train', img_dir='images', mask_dir='masks',
                             mode='train', mosaic_ratio=0.25, transform=get_transforms(train=True), img_size=(1024, 1024))

val_dataset = UAVID(data_root=DATA_DIR+'/val', img_dir='images', mask_dir='masks', mode='val',
                           mosaic_ratio=0.0, transform=get_transforms(train=False), img_size=(1024, 1024))


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)


# define the optimizer and scheduler
net_params = utils.process_model_params(net)
optimizer = torch.optim.AdamW(net_params, lr=lr, amsgrad=False)
lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=step, gamma=gamma)
