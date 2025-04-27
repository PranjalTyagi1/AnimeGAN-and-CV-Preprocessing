import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import os

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)

class AnimeGAN(nn.Module):
    def __init__(self):
        super(AnimeGAN, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(4)]
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Input: 3 x 256 x 256
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 x 128 x 128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 x 64 x 64
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 x 32 x 32
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 512 x 31 x 31
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
            # Output: 1 x 30 x 30
        )
    
    def forward(self, x):
        return self.model(x)

class AnimeStyleTransformer:
    def __init__(self, model_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize AnimeGAN model
        self.model = AnimeGAN().to(self.device)
        
        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        
        self.model.eval()
        
        # Image transformation for the GAN
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def transform_image(self, image):
        """Transform a real image to anime style"""
        # Prepare image for the GAN
        img_tensor = self.transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Generate anime-style image
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Convert output tensor to image
        output = output.squeeze().cpu().detach().numpy()
        output = output.transpose(1, 2, 0)
        output = ((output * 0.5 + 0.5) * 255).astype(np.uint8)
        
        # Convert from RGB to BGR for OpenCV
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        return output

def process_gan(input_image, model_path=None):
    """Process function for GAN transformation in Gradio interface"""
    # Convert from PIL to OpenCV format if needed
    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # Initialize transformer with model if provided
    transformer = AnimeStyleTransformer(model_path=model_path)
    
    # Transform image
    anime_image = transformer.transform_image(input_image)
    
    # Convert back to PIL for Gradio
    anime_image_pil = Image.fromarray(cv2.cvtColor(anime_image, cv2.COLOR_BGR2RGB))
    
    return anime_image_pil

if __name__ == "__main__":
    # Test the GAN on a sample image
    image_path = "sample.jpg"
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        transformer = AnimeStyleTransformer()
        anime_image = transformer.transform_image(image)
        
        # Save result
        cv2.imwrite("anime_result.jpg", anime_image)
        print("Anime-style image saved as anime_result.jpg")
    else:
        print(f"Sample image {image_path} not found.")