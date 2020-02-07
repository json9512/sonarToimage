'''
    It is important to note that the vehicle is moving as each laser scan is measured 
    Hence, the full 360 degree output is not a representation of a static full sweep.
'''

import os
import numpy as np 
import sys
from datasets import *
from models import *
import argparse
from function_lib import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

paths = sorted([x for x in os.listdir("data/test")], key=len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="GAN_data64", help="Name of dataset")
    parser.add_argument("--epoch", type=int, default=199, help="Last epoch of the model")
    parser.add_argument("--path_to_data", type=str, default="result/GAN_data64/full_scan/", help="path to GAN generated dataset")
    parser.add_argument("--pixel_to_crop", type=int, default=7, help="pixel value to crop each tile of image")
    opt = parser.parse_args()

    # Make dir 
    os.makedirs("result/%s/full_scan/" % opt.dataset_name , exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    generator = GeneratorUNet(kernel_size=4)

    if cuda:
        generator = generator.cuda()
    
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))

    test_loaders = []

    transforms_ = [
        transforms.Resize((64, 64), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    for i in paths:
        # Test loader
        testLoader = DataLoader(
            TestDataset("data/test/%s" % i , transforms_=transforms_, mode="val"),
            batch_size=1,
            num_workers=1,
        )
    
        test_loaders.append(testLoader)
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Test Model
    
    with torch.no_grad():
        # Iterate through the testloaders and create an array or images 
        j = 0 # Testloader counter
        sys.stdout.write("\n[INFO] Generating Images ...\n")
        for loader in test_loaders:
            iterator = iter(loader)
            for i in range(485):
                j = 0 if j >=12 else j
                list_of_imgs = []
                list_of_scans = []

                # Extract scan image from testloader
                imgs = next(iterator) 

                # receive scan and generate image
                real_A = Variable(imgs["scan"].type(Tensor))
                fake_B = generator(real_A)
                list_of_scans.append(real_A.data)
                list_of_imgs.append(fake_B.data)

                # Save output 
                scan_sample = torch.cat(list_of_scans)
                img_sample = torch.cat(list_of_imgs)

                save_image(img_sample, "result/%s/full_scan/%s_%s.png" % (opt.dataset_name, paths[j], i), nrow=1, normalize=True)
                j += 1

    # Generate the data for the final image
    GenerateFinalImage(opt.path_to_data, opt.pixel_to_crop)