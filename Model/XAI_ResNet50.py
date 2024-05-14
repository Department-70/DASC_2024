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
        # Load pre-trained ResNet50 model with 3 channels
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Modify the first layer to accept input with 4 channels
        # Adjust the weight accordingly by adding an extra channel
        self.resnet50.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Initialize as a dictionary object
        self.feature_maps = {}

    def forward(self, x, save_folder=None, file_name=None):
        # Original forward pass of ResNet50
        features = self.resnet50.bn1(self.resnet50.relu(self.resnet50.conv1(x)))
        
        # Save the initial conv layer features
        # 1st "Off-ramp"
        self.feature_maps['initial_conv'] = features.clone().detach()
        
        features = self.resnet50.maxpool(features)

        features = self.resnet50.layer1(features)
        
        # Save images after each block in layer1
        # 2nd "Off-ramp"
        for i in range(3):
            block_name = f'layer1_block{i+1}'
            self.feature_maps[block_name] = features.clone().detach()

        features = self.resnet50.layer2(features)
        
        # Save images after each block in layer2
        # 3rd "Off-ramp"
        for i in range(4):
            block_name = f'layer2_block{i+1}'
            self.feature_maps[block_name] = features.clone().detach()

        features = self.resnet50.layer3(features)
        
        # Save images after each block in layer3
        # 4th "Off-ramp"
        for i in range(6):
            block_name = f'layer3_block{i+1}'
            self.feature_maps[block_name] = features.clone().detach()

        features = self.resnet50.layer4(features)
        
        # Save images after each block in layer4
        # 5th "Off-ramp"
        for i in range(3):
            block_name = f'layer4_block{i+1}'
            self.feature_maps[block_name] = features.clone().detach()

        # Global average pooling layer
        features = F.adaptive_avg_pool2d(features, (1, 1))
        
        # Fully connected layer (Output layer)
        features = torch.flatten(features, 1)
        output = self.resnet50.fc(features)

        return output
    
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
            plt.title(f"{key}_LIME_Explanation")
            plt.axis('off')
    
            # Save the plot
            output_path = f'{output_root}/{file_name}_{key}_LIME_Explanation.png'
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
    original_image_pil = Image.open(image_path).convert('RGBA')
    original_image = np.array(original_image_pil)
    original_image = original_image[:, :, :3]  # Ensure RGB format
    
    prediction_output_location = os.path.join(output_root, file_name)
    os.makedirs(prediction_output_location, exist_ok=True)

    # Forward pass with ResNet50
    features = xai_resnet50_model(torch.unsqueeze(transform(original_image_pil), 0),
                                  save_folder=prediction_output_location,
                                  file_name=file_name)
    
    # Get the feature maps
    feature_maps = xai_resnet50_model.get_feature_maps()
    
    # Save feature maps
    for key, feature_map in tqdm(feature_maps.items(), desc='Saving Feature Maps'):
        fig, ax = plt.subplots(figsize=(feature_map.shape[2] / 100, feature_map.shape[1] / 100), dpi=100)
        ax.imshow(feature_map[0, 0].detach().numpy(), cmap='magma')  # Adjust the channel and color map as needed
        ax.axis('off')

        # Save the plot without title
        output_path = f'{prediction_output_location}/{file_name}_{key}_feature_map.png'
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Close the figure to free up memory
        
        # Apply LIME explanation using feature maps and predictions
        #lime_explanation = xai_resnet50_model.apply_lime_explanation(feature_map[0,0], key, prediction_output_location)

    # Get predictions
    #predict1ions = predict_function(np.expand_dims(original_image_pil, 0), xai_resnet50_model)
    # heatmap = predictions[0]   
    
    # Resize and normalize heatmap
    #heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    #heatmap_normalized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized))
    
    # Convert heatmap to RGB image
    #heatmap_rgb = plt.cm.viridis(heatmap_normalized)[:, :, :3]
    
    # Overlay heatmap on original image
    #overlay_image = (heatmap_rgb * 255).astype(np.uint8)
    #final_image = cv2.addWeighted(original_image, 0.4, overlay_image, 0.6, 0)
    
    # Save the final prediction image
    #plt.imshow(final_image)
    #plt.axis('off')
    #output_path = f'{prediction_output_location}/{file_name}_prediction.png'
    #plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    #plt.close()  # Close the plot
    
    #return lime_explanation #final_image

"""
===================================================================================================
    Main
===================================================================================================
"""
if __name__ == "__main__":
    # Counter
    counter = 1
    
    # Create an instance of the XAIResNet50 model
    xai_resnet50_model = XAIResNet50()
            
    """
    ===================================================================================================
        Loop to iterate through the dataset
    ===================================================================================================
    """
    for files in os.scandir(image_root):
        print("Counter = " + str(counter) + '.')
    
        # Filename
        file_name = os.path.splitext(files.name)[0]
        print("File Name:" + file_name)
        
        # Load an RGB image using PIL
        original_image_pil = Image.open(image_root + file_name + '.jpg').convert('RGBA')
        original_image = np.array(original_image_pil)
        
        # Load the GT image using PIL
        gt_image_pil = Image.open(gt_root + file_name + '.png')
        gt_image = np.array(gt_image_pil)
                
        # Ensure the image is in RGB format
        original_image = original_image[:, :, :3]  # Discard the alpha channel if it exists
        
        prediction_output_location = os.path.join(output_root, file_name)
        if os.path.exists(prediction_output_location) and os.listdir(prediction_output_location):
            print(f'Skipping {file_name} as prediction output already exists.')
            counter += 1
            continue
        
        print(f'Prediction Output Location: {prediction_output_location}')
        
        # Create the folder dynamically
        os.makedirs(prediction_output_location, exist_ok=True)

        # Forward pass with ResNet50
        features = xai_resnet50_model(torch.unsqueeze(transform(original_image_pil), 0),
                                      save_folder=prediction_output_location,
                                      file_name=file_name)
        
        # Get the feature maps
        feature_maps = xai_resnet50_model.get_feature_maps()
        
        # Visualize or save the feature maps as needed
        # for key, feature_map in feature_maps.items():
        for key, feature_map in tqdm(feature_maps.items(), desc='Saving Feature Maps'):
            fig, ax = plt.subplots(figsize=(feature_map.shape[2] / 100, feature_map.shape[1] / 100), dpi=100)
            ax.imshow(feature_map[0, 0].detach().numpy(), cmap='magma')  # Adjust the channel and color map as needed
            ax.axis('off')

            # Save the plot without title
            output_path = f'{prediction_output_location}/{file_name}_{key}_feature_map.png'
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)  # Close the figure to free up memory
                                
        # Use the predict_function to get predictions
        predictions = predict_function(np.expand_dims(original_image_pil, 0))
        
        # Assuming predictions is a NumPy array with shape (1, 1000)
        heatmap = predictions[0]
        
        # Resize the heatmap to match the size of the original image
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Normalize the heatmap values to be between 0 and 1
        heatmap_normalized = (heatmap_resized - np.min(heatmap_resized)) / (np.max(heatmap_resized) - np.min(heatmap_resized))
        
        # Convert the heatmap to an RGB image
        heatmap_rgb = plt.cm.viridis(heatmap_normalized)[:, :, :3]
        
        # Overlay the heatmap on the original image
        overlay_image = (heatmap_rgb * 255).astype(np.uint8)
        final_image = cv2.addWeighted(original_image, 0.4, overlay_image, 0.6, 0)
        
        # Display the final image
        plt.imshow(final_image)
        plt.title("Prediction Overlay on Original Image")
        plt.axis('off')
        
        # Save the plot
        output_path = f'{prediction_output_location}/{file_name}_prediction.png'
        plt.savefig(output_path)
        
        # Show the plot
        # plt.show()
        
        # Close the Plot to clear up space
        plt.close()
                
        counter += 1
        
        if counter > 3040:
            break
