import torch
from src.loss.UNetFormer_loss import UnetFormerLoss
from src.models.UNetFormer import UNetFormer
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from datetime import datetime
import os

CLASSES = ('Static','Road','Dynamic',"Clutter")
num_classes = len(CLASSES)
classes = CLASSES

BASE_DIR = os.getcwd()
MODELS_DIR = f"{BASE_DIR}/results"
LOGS_DIR = f"{BASE_DIR}/logs"
DATA_DIR = f"{BASE_DIR}/uavid"

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H:%M")
experiment_name = "Unetformer_{}".format(dt_string)


# training hparam
max_epoch = 40
ignore_index = 255 #class to ignore
train_batch_size = 8
val_batch_size = 8

lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01

net_name = "Unetformer"
weights_name = "unetformer-r18-1024-tiles-e{}".format(max_epoch)
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

#  define the network
net = UNetFormer(num_classes=num_classes)
# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)

# define the optimizer and scheduler

layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

