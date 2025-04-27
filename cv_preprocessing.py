import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import os

class PreprocessingPipeline:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize face detection model (MTCNN)
        self.face_detector = MTCNN(
            keep_all=True,
            device=self.device,
            post_process=False
        )
        print("Face detector (MTCNN) initialized")
        
        # Initialize edge detection model (HED-based)
        self.edge_detector = self.initialize_edge_detector()
        print("Edge detector initialized")
        
        # Initialize semantic segmentation model (DeepLabV3)
        self.segmentation_model = self.initialize_segmentation_model()
        print("Semantic segmentation model initialized")
        
        # Image transformation for neural networks
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def initialize_edge_detector(self):
        # For simplicity, we'll use a pre-trained model that approximates HED behavior
        model = models.resnet18(pretrained=True)
        model.fc = nn.Conv2d(512, 1, kernel_size=1)  # Modify to output edge maps
        model.to(self.device)
        model.eval()
        return model
    
    def initialize_segmentation_model(self):
        # Load DeepLabV3 with ResNet-50 backbone
        model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        model.to(self.device)
        model.eval()
        return model
    
    def detect_faces(self, image):
        """Detect faces in the image and return bounding boxes"""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Convert to PIL Image for MTCNN
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detect faces
        boxes, _ = self.face_detector.detect(pil_image)
        
        # If no faces detected, return None
        if boxes is None:
            return None
        
        return boxes
    
    def generate_edge_map(self, image):
        """Generate edge map using the edge detection model"""
        # Prepare image for the model
        img_tensor = self.transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Get edge map
        with torch.no_grad():
            # For our simplified model
            features = self.edge_detector.conv1(img_tensor)
            features = self.edge_detector.bn1(features)
            features = self.edge_detector.relu(features)
            features = self.edge_detector.maxpool(features)
            
            features = self.edge_detector.layer1(features)
            features = self.edge_detector.layer2(features)
            features = self.edge_detector.layer3(features)
            features = self.edge_detector.layer4(features)
            
            edge_map = F.interpolate(features, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)
            edge_map = torch.sigmoid(edge_map)
        
        # Convert to numpy array
        edge_map = edge_map.squeeze().cpu().numpy()
        
        # Normalize to 0-255 range
        edge_map = (edge_map * 255).astype(np.uint8)
        
        return edge_map
    
    def generate_segmentation_mask(self, image):
        """Generate semantic segmentation mask"""
        # Prepare image for the model
        img_tensor = self.transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Get segmentation mask
        with torch.no_grad():
            output = self.segmentation_model(img_tensor)['out']
            output = F.interpolate(output, size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False)
            mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Convert to colored segmentation mask for visualization
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        # Define colors for different semantic classes (simplified)
        colors = [
            [0, 0, 0],       # background
            [128, 0, 0],     # person
            [0, 128, 0],     # animal
            [128, 128, 0],   # vehicle
            [0, 0, 128],     # indoor object
            [128, 0, 128],   # outdoor object
        ]
        
        # Apply colors to mask
        for i, color in enumerate(colors):
            if i < 21:  # DeepLabV3 has 21 classes
                colored_mask[mask == i] = color
        
        return colored_mask
    
    def preprocess_image(self, image):
        """Apply all preprocessing steps to the image"""
        # Make a copy of the original image
        processed_image = image.copy()
        
        # 1. Face detection
        face_boxes = self.detect_faces(image)
        
        # 2. Edge detection
        edge_map = self.generate_edge_map(image)
        
        # 3. Semantic segmentation
        segmentation_mask = self.generate_segmentation_mask(image)
        
        # Create image with face boxes for visualization
        face_detection_vis = processed_image.copy()
        if face_boxes is not None:
            for box in face_boxes:
                box = box.astype(int)
                cv2.rectangle(face_detection_vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Return all preprocessing results
        results = {
            'original': image,
            'edge_map': edge_map,
            'segmentation': segmentation_mask,
            'face_detection': face_detection_vis,
            'face_boxes': face_boxes
        }
        
        return results

def process_cv(input_image):
    """Process function for CV preprocessing in Gradio interface"""
    # Convert from PIL to OpenCV format if needed
    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # Initialize preprocessing pipeline
    preprocessor = PreprocessingPipeline()
    
    # Preprocess image
    results = preprocessor.preprocess_image(input_image)
    
    # Convert back to PIL for Gradio
    original_pil = Image.fromarray(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
    edge_map_pil = Image.fromarray(results['edge_map'])
    segmentation_pil = Image.fromarray(results['segmentation'])
    face_detection_pil = Image.fromarray(cv2.cvtColor(results['face_detection'], cv2.COLOR_BGR2RGB))
    
    return original_pil, edge_map_pil, segmentation_pil, face_detection_pil

if __name__ == "__main__":
    # Test the preprocessing pipeline on a sample image
    image_path = "sample.jpg"
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        preprocessor = PreprocessingPipeline()
        results = preprocessor.preprocess_image(image)
        
        # Display results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(results['edge_map'], cmap='gray')
        plt.title('Edge Map')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(results['segmentation'])
        plt.title('Segmentation Mask')
        plt.axis('off')
        
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(results['face_detection'], cv2.COLOR_BGR2RGB))
        plt.title('Face Detection')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("preprocessing_results.png")
        plt.show()
    else:
        print(f"Sample image {image_path} not found.")