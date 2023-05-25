import os
import torch
from catalyst import utils
import segmentation_models_pytorch as smp
from src.loss.soft_ce import SoftCrossEntropyLoss
from datetime import datetime

CLASSES = ('Static','Road','Dynamic',"Clutter")
num_classes = len(CLASSES)
classes = CLASSES

BASE_DIR = os.getcwd()
MODELS_DIR = f"{BASE_DIR}/results"
LOGS_DIR = f"{BASE_DIR}/logs"
DATA_DIR = f"{BASE_DIR}/uavid"


now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H:%M")
experiment_name = "Feature_pyramid_net{}".format(dt_string)


# training hparam
max_epoch = 40
ignore_index = 255 #class to ignore
train_batch_size = 4
val_batch_size = 4
lr = 0.001
weight_decay = 0.0003
backbone_lr = 0.0005
backbone_weight_decay = 0.00003

net_name = "FPN"
weights_name = "fpn-resnext50_32x4d-e{}".format(max_epoch)
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
net = smp.FPN(encoder_name="resnext50_32x4d", classes=num_classes)
# define the loss
loss = SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index)


# define the optimizer and scheduler
# Since we use a pre-trained encoder, we will reduce the learning rate on it.
layerwise_params = {"encoder*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}

# This function removes weight_decay for biases and applies our layerwise_params
model_params = utils.process_model_params(net, layerwise_params=layerwise_params)

optimizer = torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)

lr_scheduler = {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25,patience=2, mode=monitor_mode)
                ,'name': 'learning_rate'
                ,'monitor': monitor}
