import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from anime_gan import AnimeGAN, Discriminator
from cv_preprocessing import PreprocessingPipeline
from utils import VGGLoss, save_sample_images, process_anime_dataset

# Custom dataset for anime images
class AnimeDataset(Dataset):
    def __init__(self, anime_dir, transform=None, paired=False, real_dir=None):
        self.anime_dir = anime_dir
        self.transform = transform
        self.paired = paired
        self.real_dir = real_dir
        
        # Get all image filenames
        self.anime_images = sorted([f for f in os.listdir(anime_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        if paired and real_dir:
            self.real_images = sorted([f for f in os.listdir(real_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            # Ensure we have matching pairs
            assert len(self.real_images) == len(self.anime_images), "Number of real and anime images must match"
    
    def __len__(self):
        return len(self.anime_images)
    
    def __getitem__(self, idx):
        # Load anime image
        anime_path = os.path.join(self.anime_dir, self.anime_images[idx])
        anime_image = Image.open(anime_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            anime_image = self.transform(anime_image)
        
        if self.paired and self.real_dir:
            # Load real image
            real_path = os.path.join(self.real_dir, self.real_images[idx])
            real_image = Image.open(real_path).convert('RGB')
            
            if self.transform:
                real_image = self.transform(real_image)
            
            return real_image, anime_image
        
        return anime_image

def train_anime_gan(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training started on device: {device}")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create dataset and dataloader
    anime_dirs = args.anime_dir.split(',')  # Split the input string into directories
    print("Loading anime images from:", anime_dirs)
    
    if args.mode == 'paired':
        # For paired training (real-anime pairs)
        dataset = AnimeDataset(
            anime_dir=anime_dirs[0],  # First directory for anime images
            real_dir=args.real_dir, 
            transform=transform,
            paired=True
        )
    else:
        # For unpaired training (anime images only)
        dataset = AnimeDataset(
            anime_dir=anime_dirs[0],  # First directory for anime images
            transform=transform,
            paired=False
        )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Initialize models
    generator = AnimeGAN().to(device)
    discriminator = Discriminator().to(device)
    
    # Initialize preprocessing pipeline for edge and segmentation features if needed
    if args.use_cv_features:
        preprocessor = PreprocessingPipeline(device=device)
    
    # Initialize optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Initialize losses
    adversarial_loss = nn.BCEWithLogitsLoss()
    content_loss = VGGLoss(device)
    l1_loss = nn.L1Loss()
    
    # Training loop
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, data in enumerate(pbar):
            if args.mode == 'paired':
                real_images, anime_images = data
                real_images = real_images.to(device)
                anime_images = anime_images.to(device)
            else:
                # For unpaired training, we only have anime images
                anime_images = data
                anime_images = anime_images.to(device)
                # Generate random noise as input to the generator
                real_images = torch.randn(anime_images.size()).to(device)
            
            # Create labels
            real_labels = torch.ones(anime_images.size(0), 1, 30, 30).to(device)
            fake_labels = torch.zeros(anime_images.size(0), 1, 30, 30).to(device)
            
            # -----------------------
            # Train Discriminator
            # -----------------------
            d_optimizer.zero_grad()
            
            # Real images
            real_outputs = discriminator(anime_images)
            d_real_loss = adversarial_loss(real_outputs, real_labels)
            
            # Fake images
            fake_images = generator(real_images)
            fake_outputs = discriminator(fake_images.detach())
            d_fake_loss = adversarial_loss(fake_outputs, fake_labels)
            
            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # -----------------------
            # Train Generator
            # -----------------------
            g_optimizer.zero_grad()
            
            # Adversarial loss
            fake_outputs = discriminator(fake_images)
            g_adv_loss = adversarial_loss(fake_outputs, real_labels)
            
            # Content loss
            g_content_loss = content_loss(fake_images, real_images)
            
            # L1 loss (only for paired training)
            if args.mode == 'paired':
                g_l1_loss = l1_loss(fake_images, anime_images)
                g_loss = g_adv_loss + 10 * g_content_loss + 5 * g_l1_loss
            else:
                g_loss = g_adv_loss + 10 * g_content_loss
            
            g_loss.backward()
            g_optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({
                'D Loss': d_loss.item(),
                'G Loss': g_loss.item(),
                'Adv': g_adv_loss.item(),
                'Content': g_content_loss.item()
            })
        
        # Save model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            torch.save(generator.state_dict(), os.path.join(args.output_dir, f'generator_epoch_{epoch+1}.pth'))
            
            # Generate and save sample images
            generator.eval()
            with torch.no_grad():
                # Get a batch of images from the dataloader
                if args.mode == 'paired':
                    real_batch, _ = next(iter(dataloader))
                    real_batch = real_batch.to(device)
                else:
                    # For unpaired training, generate random noise
                    real_batch = torch.randn(min(5, args.batch_size), 3, 256, 256).to(device)
                
                # Generate fake images
                fake_batch = generator(real_batch)
                
                # Save sample images
                save_sample_images(real_batch, fake_batch, epoch + 1, args.output_dir)
        
        # Log progress
        print(f"Epoch {epoch+1}/{args.epochs} completed. Saving model...")

    # Save final model
    torch.save(generator.state_dict(), os.path.join(args.output_dir, 'generator_final.pth'))
    print(f"Training completed. Final model saved to {os.path.join(args.output_dir, 'generator_final.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AnimeGAN model")
    parser.add_argument('--anime_dir', type=str, required=True, help='Comma-separated list of directories containing anime images')
    parser.add_argument('--real_dir', type=str, default=None, help='Directory containing real-world images (for paired training)')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for model checkpoints and samples')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval for saving model checkpoints')
    parser.add_argument('--mode', type=str, default='unpaired', choices=['paired', 'unpaired'], help='Training mode')
    parser.add_argument('--process_dataset', action='store_true', help='Process and prepare the anime dataset before training')
    parser.add_argument('--processed_dir', type=str, default='./processed_anime', help='Directory to save processed anime images')
    parser.add_argument('--use_cv_features', action='store_true', help='Use computer vision features during training')
    
    args = parser.parse_args()
    
    # Process dataset if requested
    if args.process_dataset:
        process_anime_dataset(args.anime_dir, args.processed_dir)
        args.anime_dir = args.processed_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    train_anime_gan(args)
