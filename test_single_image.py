import cv2
import argparse
import os
import numpy as np
from PIL import Image

from cv_preprocessing import PreprocessingPipeline
from anime_gan import AnimeStyleTransformer
from utils import visualize_results

def process_single_image(image_path, model_path=None, output_dir='./output'):
    """Process a single image and save the result"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Step 1: CV Preprocessing
    preprocessor = PreprocessingPipeline()
    cv_results = preprocessor.preprocess_image(image)
    
    # Save CV preprocessing results
    cv2.imwrite(os.path.join(output_dir, f"edge_{os.path.basename(image_path)}"), cv_results['edge_map'])
    cv2.imwrite(os.path.join(output_dir, f"segmentation_{os.path.basename(image_path)}"), cv_results['segmentation'])
    cv2.imwrite(os.path.join(output_dir, f"face_{os.path.basename(image_path)}"), cv_results['face_detection'])
    
    print(f"CV preprocessing results saved to {output_dir}")
    
    # Step 2: GAN Transformation
    transformer = AnimeStyleTransformer(model_path=model_path)
    anime_image = transformer.transform_image(image)
    
    # Save anime-style image
    output_path = os.path.join(output_dir, f"anime_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, anime_image)
    
    print(f"Anime-style image saved to {output_path}")
    
    # Step 3: Create and save visualization
    visualization = visualize_results(image, cv_results, anime_image)
    vis_path = os.path.join(output_dir, f"vis_{os.path.basename(image_path)}")
    cv2.imwrite(vis_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    
    print(f"Visualization saved to {vis_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Anime Transformer on a single image")
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained generator model')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for transformed images')
    
    args = parser.parse_args()
    
    process_single_image(args.image_path, args.model_path, args.output_dir)