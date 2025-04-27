import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device)
        self.vgg = nn.Sequential()
        self.vgg_layers = [0, 5, 10, 19, 28]
        
        for i in range(max(self.vgg_layers) + 1):
            self.vgg.add_module(str(i), vgg[i])
        
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
    
    def forward(self, x, y):
        x_vgg, y_vgg = x, y
        loss = 0
        
        for i, layer in enumerate(self.vgg_layers):
            x_vgg = self.vgg[layer](x_vgg)
            y_vgg = self.vgg[layer](y_vgg)
            loss += self.weights[i] * self.criterion(x_vgg, y_vgg)
        
        return loss

def save_sample_images(real_images, fake_images, epoch, output_dir):
    """Save sample images during training"""
    # Create output directory if it doesn't exist
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays
    real_images = real_images.cpu().numpy()
    fake_images = fake_images.cpu().numpy()
    
    # Create a grid of images
    n_samples = min(5, real_images.shape[0])
    plt.figure(figsize=(12, 6))
    
    for i in range(n_samples):
        # Original image
        plt.subplot(2, n_samples, i + 1)
        real_img = np.transpose(real_images[i], (1, 2, 0))
        real_img = (real_img * 0.5 + 0.5).clip(0, 1)
        plt.imshow(real_img)
        plt.axis('off')
        
        if i == 0:
            plt.title('Input Images')
        
        # Generated image
        plt.subplot(2, n_samples, i + 1 + n_samples)
        fake_img = np.transpose(fake_images[i], (1, 2, 0))
        fake_img = (fake_img * 0.5 + 0.5).clip(0, 1)
        plt.imshow(fake_img)
        plt.axis('off')
        
        if i == 0:
            plt.title('Anime-Style Images')
    
    plt.tight_layout()
    plt.savefig(os.path.join(samples_dir, f'samples_epoch_{epoch}.png'))
    plt.close()

def process_anime_dataset(anime_dir, output_dir, size=(256, 256)):
    """Process and prepare anime dataset for training"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(anime_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Processing {len(image_files)} anime images...")
    
    # Process each image
    for i, img_file in enumerate(image_files):
        if i % 100 == 0:
            print(f"Processing image {i+1}/{len(image_files)}")
            
        # Load image
        img_path = os.path.join(anime_dir, img_file)
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Resize image
            img = img.resize(size, Image.LANCZOS)
            
            # Save processed image
            output_path = os.path.join(output_dir, img_file)
            img.save(output_path)
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
    
    print(f"Processed anime dataset saved to {output_dir}")

def visualize_results(original, cv_results, anime_image):
    """Create a visualization of all results"""
    # Create a figure for visualization
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axs[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    # Edge map
    axs[0, 1].imshow(cv_results['edge_map'], cmap='gray')
    axs[0, 1].set_title('Edge Map')
    axs[0, 1].axis('off')
    
    # Segmentation mask
    axs[0, 2].imshow(cv_results['segmentation'])
    axs[0, 2].set_title('Segmentation Mask')
    axs[0, 2].axis('off')
    
    # Face detection
    axs[1, 0].imshow(cv2.cvtColor(cv_results['face_detection'], cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('Face Detection')
    axs[1, 0].axis('off')
    
    # Anime-style image
    axs[1, 1].imshow(cv2.cvtColor(anime_image, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title('Anime-Style Image')
    axs[1, 1].axis('off')
    
    # Side-by-side comparison
    comparison = np.hstack((cv2.cvtColor(original, cv2.COLOR_BGR2RGB), 
                           cv2.cvtColor(anime_image, cv2.COLOR_BGR2RGB)))
    axs[1, 2].imshow(comparison)
    axs[1, 2].set_title('Before / After')
    axs[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Convert the figure to an image
    fig.canvas.draw()
    vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return vis_image