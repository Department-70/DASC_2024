# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:02:01 2024

@author: Debra Hogue

Tracers - XAI
"""

import cv2
from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Folder containing images
gt_root = './GT/'
image_root = './images/'
output_root = './resnet50_output/'

"""
===================================================================================================
    XAI ResNet50 - Modified to accept extra tracer channel
===================================================================================================
"""
class XAIResNet50(torch.nn.Module):
    def __init__(self):
        super(XAIResNet50, self).__init__()
        
        # Load pre-trained ResNet50 model
        self.resnet50 = models.resnet50(pretrained=True)   
        
        # Initialize feature maps (off-ramps output collection)
        self.feature_maps = {}
        
        # Initialize prediction object
        self.predictions = None

    def forward(self, x, save_folder=None, file_name=None):
        # Initial layers
        features = self.resnet50.bn1(self.resnet50.relu(self.resnet50.conv1(x)))
        features = self.resnet50.maxpool(features)
        self.feature_maps['initial_conv'] = features.clone().detach()

        # Layer 1
        features = self.resnet50.layer1(features)
        self.feature_maps['stage1'] = features.clone().detach()
        
        # Layer 2
        features = self.resnet50.layer2(features)
        self.feature_maps['stage2'] = features.clone().detach()
        
        # Layer 3
        features = self.resnet50.layer3(features)
        self.feature_maps['stage3'] = features.clone().detach()
        
        # Layer 4
        features = self.resnet50.layer4(features)
        self.feature_maps['stage4'] = features.clone().detach()
        
        # Global average pooling and final fully connected layer
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        self.predictions = self.resnet50.fc(features)

        return self.predictions
    
    def get_feature_maps(self):
        return self.feature_maps

"""
===================================================================================================
    Define a function to preprocess the image
===================================================================================================
"""
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

"""
===================================================================================================
    Image Preprocessing Helper Functions
===================================================================================================
"""
# Define a basic transform function for preprocessing
def transform(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Adjust the mean and std for four channels (RGB + alpha)
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.0], std=[0.229, 0.224, 0.225, 1.0]),
    ])
    return preprocess(image)

# Define a function to predict using the ResNet50 model
def predict_function(images, xai_resnet50_model):
    # Make a copy of the NumPy array
    images_copy = np.copy(images)

    # Images should be in the shape (num_samples, height, width, num_channels)
    images_tensor = torch.Tensor(images_copy).permute(0, 3, 1, 2)
    outputs = xai_resnet50_model(images_tensor)
    return outputs.detach().numpy()

# Additional transform function to convert PyTorch tensor to PIL Image
def to_pil_image(tensor):
    return transforms.ToPILImage()(tensor)

"""
===================================================================================================
    Callable Function for C# front-end
===================================================================================================
"""
def process_image_with_resnet50(image_path):
    # Create an instance of the XAIResNet50 model
    xai_resnet50_model = XAIResNet50()
    
    # Load the image
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    input_tensor = preprocess_image(image_path)
    xai_resnet50_model(input_tensor)
    xai_resnet50_model.eval()
    target_layer = xai_resnet50_model.resnet50.layer4[2].conv3  # target the final convolutional layer
    
    # Convert the input tensor to a numpy array for visualization
    input_image = input_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())  # Normalize to [0, 1]

    # Construct the CAM object once, and then re-use it on many images:
    gradcam = GradCAM(model=model, target_layers=target_layer)
    targets = [ClassifierOutputTarget(281)]
    
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = gradcam(input_tensor=input_tensor, targets=targets)
    
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)

    # Get the feature maps
    feature_maps = xai_resnet50_model.get_feature_maps()
    
    prediction_output_location = os.path.join(output_root, file_name)
    os.mkdir(prediction_output_location, exist_ok=True)  # Create the directory if it does not exist
    
    if os.path.exists(prediction_output_location) and os.listdir(prediction_output_location):
        print(f'Skipping {file_name} as prediction output already exists.')
    
    print(f'Prediction Output Location: {prediction_output_location}')

    # Save "off-ramp" outputs to appropriate folder 
    for key, feature_map in tqdm(feature_maps.items(), desc='Saving Feature Maps'):
        fig, ax = plt.subplots(figsize=(feature_map.shape[2] / 100, feature_map.shape[1] / 100), dpi=100)
        ax.imshow(feature_map[0, 0].detach().numpy(), cmap='magma')  # Adjust the channel and color map as needed
        ax.axis('off')
    
        # Save the plot without title
        output_path = f'{prediction_output_location}/{file_name}_{key}_feature_map.png'
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure to free up memory

"""
===================================================================================================
    Main
===================================================================================================
"""
if __name__ == "__main__":
    # Counter
    counter = 1

    # Folder containing images
    image_folder_path = 'C:\\Users\\Windows\\Downloads\\images'
    
    # List all images in the folder
    image_files = [os.path.join(image_folder_path, file) for file in os.listdir(image_folder_path) if file.endswith(('jpg', 'jpeg', 'png'))]
    
    # Loop through each image in the folder
    for image_file in image_files:
        process_image_with_resnet50(image_file)
        counter += 1
        
        if counter >= 2:
            break