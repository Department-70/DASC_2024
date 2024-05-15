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

from lime import lime_image
from skimage.segmentation import mark_boundaries

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
        self.feature_maps = {}
        self.predictions = None

    def forward(self, x):
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
        self.feature_maps['prediction'] = self.predictions.clone().detach()   
    
    def apply_lime_explanation(self, feature_map, key, predictions):
            # Initialize LIME explainer for image classification
            explainer = lime_image.LimeImageExplainer()
    
            # Convert feature map tensor to numpy array
            feature_map_numpy = feature_map.detach().numpy()
    
            # Convert predictions (assuming self.predictions is a PyTorch tensor) to numpy array
            predictions_numpy = predictions.detach().numpy()
    
            # Create a synthetic RGB image by repeating the feature map across channels
            synthetic_image = np.repeat(np.squeeze(feature_map_numpy), 3, axis=0)  # Squeeze to remove singleton dimensions
    
            # Define a classifier function for LIME using the model's predictions
            def classifier_fn(images):
                # Assuming images is a numpy array
                # Convert images to PyTorch tensor if needed
                images_tensor = torch.from_numpy(images).float()
                # Forward pass through the model to get predictions
                with torch.no_grad():
                    outputs = self.model(images_tensor)
                return outputs.numpy()  # Convert predictions back to numpy array
    
            # Explain predictions for the synthetic RGB image
            explanation = explainer.explain_instance(
                synthetic_image,
                classifier_fn,  # Pass the classifier function instead of predictions
                top_labels=1,
                hide_color=0,
                num_samples=1000
            )
    
            # Display or save the LIME explanation
            lime_heatmap = mark_boundaries(
                np.array(explanation.segments),
                explanation.local_exp[explanation.top_labels[0]]
            )
            plt.imshow(lime_heatmap)
            plt.axis('off')
    
            # Save the plot
            output_path = f'{output_root}/{file_name}_{key}_LIME_prediction.png'
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
    
            return explanation
    
    def get_feature_maps(self):
        return self.feature_maps

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
def predict_function(images):
    # Make a copy of the NumPy array
    images_copy = np.copy(images)

    # Images should be in the shape (num_samples, height, width, num_channels)
    images_tensor = torch.Tensor(images_copy).permute(0, 3, 1, 2)
    outputs = xai_resnet50_model(images_tensor)
    return outputs.detach().numpy()

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

def process_image_with_resnet50(image_path):
    # Create an instance of the XAIResNet50 model
    xai_resnet50_model = XAIResNet50()
    
    # Load the image
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    original_image = Image.open(image_path).convert('RGB')
    #original_image = np.array(original_image_pil)
    original_image = original_image[:, :, :3]  # Ensure RGB format
    
    prediction_output_location = os.path.join(output_root, file_name)
    os.makedirs(prediction_output_location, exist_ok=True)

    # Forward pass with ResNet50
    features = xai_resnet50_model(torch.unsqueeze(transform(original_image), 0),
                                  save_folder=prediction_output_location,
                                  file_name=file_name)
    
    # Get the feature maps
    feature_maps = xai_resnet50_model.get_feature_maps()
    
    # Save feature maps
    for key, feature_map in tqdm(feature_maps.items(), desc='Saving Feature Maps'):
        plt.imshow(feature_map[0, 0].detach().numpy(), cmap='magma')  
        plt.title(f"Feature Map - {key}")
        plt.axis('off')
        output_path = f'{prediction_output_location}/{file_name}_{key}_feature_map.png'
        plt.savefig(output_path)
        plt.close()  # Close the plot

    # Get predictions
    #predictions = predict_function(np.expand_dims(original_image_pil, 0))
    #heatmap = predictions[0]
    
    # Resize and normalize heatmap
    #heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    #heatmap_normalized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized))
    
    # Convert heatmap to RGB image
    #heatmap_rgb = plt.cm.viridis(heatmap_normalized)[:, :, :3]
    
    # Overlay heatmap on original image
    #overlay_image = (heatmap_rgb * 255).astype(np.uint8)
    #final_image = cv2.addWeighted(original_image, 0.4, overlay_image, 0.6, 0)
