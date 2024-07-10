import cv2
from PIL import Image
import numpy as np
import os
import torch
from torchvision.models.detection import efficientdet
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the model
class XAI_EfficientDetD7(torch.nn.Module):
    def __init__(self):
        super(XAI_EfficientDetD7, self).__init__()
        self.model = efficientdet.detection.backbone_utils.resnet_fpn_backbone('resnet101', pretrained_backbone=True)
        self.model = efficientdet.EfficientDet(90, num_classes=1, compound_coef=7, backbone=self.model.backbone)

    def forward(self, x):
        return self.model(x)

def transform(image):
    preprocess = transforms.Compose([
        transforms.Resize(896),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)

def process_image_with_refficientdet_d7(image_path):
    # Load the model
    xai_refficientdet_d7_model = XAI_EfficientDetD7()
    xai_refficientdet_d7_model.eval()
    
    # Load the image
    original_image_pil = Image.open(image_path).convert('RGB')
    original_image = np.array(original_image_pil)
    
    prediction_output_location = './refficientdet_output/'
    os.makedirs(prediction_output_location, exist_ok=True)

    # Forward pass with rEfficientDet-D7
    with torch.no_grad():
        inputs = transform(original_image_pil)
        inputs = inputs.unsqueeze(0)
        outputs = xai_refficientdet_d7_model(inputs)
    
    # Process and save the predictions
    # Assuming outputs contain the bounding boxes and scores
    # You can customize this part based on rEfficientDet-D7's output format
    # Here, we'll just save the image with bounding boxes
    for i in range(len(outputs)):
        image_with_boxes = draw_boxes_on_image(original_image, outputs[i]['boxes'], outputs[i]['scores'])
        plt.imshow(image_with_boxes)
        plt.axis('off')
        output_path = f'{prediction_output_location}/{os.path.splitext(os.path.basename(image_path))[0]}_prediction_{i}.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    return outputs  # Return the outputs if needed

def draw_boxes_on_image(image, boxes, scores):
    # Draw bounding boxes on the image
    for box in boxes:
        box = box.tolist()
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    return image

# Example usage
image_path = 'path_to_your_image.jpg'
process_image_with_refficientdet_d7(image_path)
