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
    Hook for the gradients of the target layer
===================================================================================================
"""
class SaveFeatures:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.features = None
        self.gradients = None

    def hook_fn(self, module, input, output):
        self.features = output
        self.hook_grad = output.register_hook(self.hook_grad_fn)

    def hook_grad_fn(self, grad):
        self.gradients = grad

    def close(self):
        self.hook.remove()

"""
===================================================================================================
    Generate Grad_CAM heatmap
===================================================================================================
"""
def generate_gradcam(model, img_tensor, target_layer, target_class=None):
    sf = SaveFeatures(target_layer)
    output = model(img_tensor)
    
    # Access the predictions directly from the model
    output = model.predictions

    if target_class is None:
        target_class = output.argmax().item()

    model.zero_grad()
    class_loss = F.cross_entropy(output, torch.tensor([target_class]))
    class_loss.backward()

    gradients = sf.gradients.data.numpy()[0]
    activations = sf.features.data.numpy()[0]
    weights = np.mean(gradients, axis=(1, 2))
    gradcam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        gradcam += w * activations[i, :, :]

    gradcam = np.maximum(gradcam, 0)
    gradcam = gradcam / gradcam.max()
    gradcam = np.uint8(gradcam * 255)
    gradcam = Image.fromarray(gradcam).resize((img_tensor.size(2), img_tensor.size(3)), Image.LANCZOS)
    return gradcam

# Superimpose Grad-CAM heatmap on original image
def superimpose_gradcam(img_path, gradcam, output_location, file_name):
    img = Image.open(img_path).convert('RGB')
    plt.imshow(img)
    plt.imshow(gradcam, cmap='jet', alpha=0.5)
    plt.axis('off')
    
    # Save the plot without title
    output_path = f'{output_location}/{file_name}_prediction.png'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free up memory
    
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
            
    # Get the feature maps
    feature_maps = xai_resnet50_model.get_feature_maps()
    
    prediction_output_location = os.path.join(output_root, file_name)
    if os.path.exists(prediction_output_location) and os.listdir(prediction_output_location):
        print(f'Skipping {file_name} as prediction output already exists.')
    
    print(f'Prediction Output Location: {prediction_output_location}')

    # Get Grad-CAM XAI output of model prediction
    gradcam = generate_gradcam(xai_resnet50_model, input_tensor, target_layer)
    superimpose_gradcam(image_path, gradcam, prediction_output_location, file_name)

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
    image_folder_path = './images/'
    
    # List all images in the folder
    image_files = [os.path.join(image_folder_path, file) for file in os.listdir(image_folder_path) if file.endswith(('jpg', 'jpeg', 'png'))]
    
    # Loop through each image in the folder
    for image_file in image_files:
        process_image_with_resnet50(image_file)
        counter += 1
        
        if counter >= 2:
            break