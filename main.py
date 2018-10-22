import os
from skimage import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import transforms as ext_transforms
from models.enet import ENet
from train import Train
from test import Test
from metric.iou import IoU
from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
import utils
from PIL import Image
from collections import OrderedDict
import math
import time

# Get the arguments
args = get_arguments()

use_cuda = args.cuda and torch.cuda.is_available()




# Run only if this module is being run directly
if __name__ == '__main__':

    # Fail fast if the saving directory doesn't exist
    assert os.path.isdir(
        args.save_dir), "The directory \"{0}\" doesn't exist.".format(
            args.save_dir)

    class_encoding = color_encoding = OrderedDict([
            ('unlabeled', (0, 0, 0)),
            ('road', (128, 64, 128)),
            ('sidewalk', (244, 35, 232)),
            ('building', (70, 70, 70)),
            ('wall', (102, 102, 156)),
            ('fence', (190, 153, 153)),
            ('pole', (153, 153, 153)),
            ('traffic_light', (250, 170, 30)),
            ('traffic_sign', (220, 220, 0)),
            ('vegetation', (107, 142, 35)),
            ('terrain', (152, 251, 152)),
            ('sky', (70, 130, 180)),
            ('person', (220, 20, 60)),
            ('rider', (255, 0, 0)),
            ('car', (0, 0, 142)),
            ('truck', (0, 0, 70)),
            ('bus', (0, 60, 100)),
            ('train', (0, 80, 100)),
            ('motorcycle', (0, 0, 230)),
            ('bicycle', (119, 11, 32))
    ])

    
    # Intialize a new ENet model
    num_classes = len(class_encoding)
    model = ENet(num_classes)
    if use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters())
    # Load the previoulsy saved model state to the ENet model
    model = utils.load_checkpoint(model, optimizer, args.save_dir, args.name)[0]

    img = Image.open("spring_sentinel_cloudy_0.jpg")
    # width, height = img.size
    # m = -0.5
    # xshift = height+20#abs(m) * width
    # new_height = height + int(round(height))
    # img = img.transform((width, height), Image.AFFINE,(math.cos(0),math.sin(0),0,-math.sin(0),math.cos(math.pi/3),0,0,0,1), Image.BICUBIC)
    start = time.time()
    # img = img.resize((1024, 512), Image.ANTIALIAS)
    images = transforms.ToTensor()(img)
    torch.reshape(images, (1, 3, 360, 600))
    images= images.unsqueeze(0)
    with torch.no_grad():
        images = Variable(images)
        images = images.cuda()
        
        predictions = model(images) 
        end = time.time()
        print(1/(end - start),"FPS")
        _, predictions = torch.max(predictions.data, 1)
        label_to_rgb = transforms.Compose([ext_transforms.LongTensorToRGBPIL(class_encoding),transforms.ToTensor()])
        color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
        end = time.time()
        print(1/(end - start),"FPS")
        # utils.imshow_batch(images.data.cpu(), color_predictions)
    
