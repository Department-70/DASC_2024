"""
Paper: Toward the Understanding of Camouflaged Object Detection by Lv et al. 2023

Modified by Debra Hogue
Added callable function, process_image_with_generator
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from scipy import misc
from RankNet.model.ResNet_models import Generator
from data import test_dataset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

def process_image_with_generator(image_path, generator_model, output_path):
    # Load the generator model
    generator = Generator(channel=32)  # Assuming you use the same settings as in the original script
    generator.load_state_dict(torch.load(generator_model))
    generator.cuda()
    generator.eval()

    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (480, 480))  # Assuming you use the same test size as in the original script
    image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float().unsqueeze(0).cuda()

    # Generate predictions with the generator
    _, _, generator_pred = generator.forward(image_tensor)
    res = generator_pred
    res = F.upsample(res, size=[480, 480], mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = 255 * (res - res.min()) / (res.max() - res.min() + 1e-8)

    # Save the generated image
    cv2.imwrite(output_path, res)

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=480, help='testing size')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
opt = parser.parse_args()

dataset_path = './dataset/test/'

generator = Generator(channel=opt.feat_channel)
generator.load_state_dict(torch.load('./models/Resnet/Model_50_gen.pth'))

generator.cuda()
generator.eval()

test_datasets = ['CAMO','CHAMELEON','COD10K','NC4K']

for dataset in test_datasets:
    save_path = './results/ResNet50/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset + '/Imgs/'
    test_loader = test_dataset(image_root, opt.testsize)
    for i in range(test_loader.size):
        print(i)
        image, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        _,_,generator_pred = generator.forward(image)
        res = generator_pred
        res = F.upsample(res, size=[WW,HH], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = 255*(res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res)