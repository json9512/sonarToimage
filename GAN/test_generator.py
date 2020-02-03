import os
import numpy as np
from datasets import *
from models import *
import argparse

import torch.nn as nn
import torch.nn.functional as F
import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="GAN_data64", help="Name of dataset")
    parser.add_argument("--epoch", type=int, default=199, help="Last epoch of the model")
    parser.add_argument("--rename_files", type=bool, default=False, help="True if the files have to be renamed")
    parser.add_argument("--angles", type=str, default=None, help="Range of the angles")
    opt = parser.parse_args()

    if opt.rename_files == True:
        count = 0
        for f in os.listdir("data/test/%s" % opt.angles):
            os.rename("data/test/%s/%s" % (opt.angles, f) , "test/%s/%d.png" % (opt.angles, count))
            count +=1

    # Make dir 
    os.makedirs("result/gen_test_result/%s/" % opt.angles, exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    generator = GeneratorUNet(kernel_size=4)

    if cuda:
        generator = generator.cuda()
    
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))

    # Configure dataloaders
    transforms_ = [
        transforms.Resize((64, 64), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # Test loader
    testLoader = DataLoader(
        TestDataset("data/test/%s" % opt.angles , transforms_=transforms_, mode="val"),
        batch_size=12,
        shuffle=True,
        num_workers=1,
    )

    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Test Model
    
    with torch.no_grad():
        for i in range(len(testLoader)):
            imgs = next(iter(testLoader))
            real_A = Variable(imgs["scan"].type(Tensor))
            fake_B = generator(real_A)
            
            img_sample = torch.cat((real_A.data, fake_B.data), -2)
            # Save output 
            save_image(img_sample, "result/gen_test_result/%s/%s.png" % (opt.angles, i), nrow=6, normalize=True)