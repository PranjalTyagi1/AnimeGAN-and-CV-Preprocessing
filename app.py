import gradio as gr
import argparse
import numpy as np
import cv2
from PIL import Image
import os

from cv_preprocessing import PreprocessingPipeline
from anime_gan import AnimeStyleTransformer
from utils import visualize_results

def process_image(input_image, model_path=None):
    """Process function for Gradio interface"""
    # Convert from PIL to OpenCV format if needed
    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    
    # Step 1: CV Preprocessing
    preprocessor = PreprocessingPipeline()
    cv_results = preprocessor.preprocess_image(input_image)
    
    # Step 2: GAN Transformation
    transformer = AnimeStyleTransformer(model_path=model_path)
    anime_image = transformer.transform_image(input_image)
    
    # Step 3: Create visualization of all results
    visualization = visualize_results(input_image, cv_results, anime_image)
    
    # Convert results to PIL for Gradio
    original_pil = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
    edge_map_pil = Image.fromarray(cv_results['edge_map'])
    segmentation_pil = Image.fromarray(cv_results['segmentation'])
    face_detection_pil = Image.fromarray(cv2.cvtColor(cv_results['face_detection'], cv2.COLOR_BGR2RGB))
    anime_image_pil = Image.fromarray(cv2.cvtColor(anime_image, cv2.COLOR_BGR2RGB))
    visualization_pil = Image.fromarray(visualization)
    
    return original_pil, edge_map_pil, segmentation_pil, face_detection_pil, anime_image_pil, visualization_pil

def create_gradio_interface(model_path=None):
    with gr.Blocks(title="Advanced Anime-Style Image Transformation System") as demo:
        gr.Markdown("# Advanced Anime-Style Image Transformation System")
        gr.Markdown("Upload a photo to transform it into anime style with our advanced CV+GAN pipeline.")
        
        with gr.Row():
            input_image = gr.Image(label="Input Image", type="pil")
        
        with gr.Row():
            transform_btn = gr.Button("Transform Image")
        
        with gr.Row():
            with gr.Column():
                edge_map = gr.Image(label="Edge Map")
                segmentation = gr.Image(label="Segmentation Mask")
            
            with gr.Column():
                face_detection = gr.Image(label="Face Detection")
                anime_output = gr.Image(label="Anime-Style Output")
        
        with gr.Row():
            visualization = gr.Image(label="Complete Transformation Process")
        
        transform_btn.click(
            fn=lambda img: process_image(img, model_path),
            inputs=[input_image],
            outputs=[input_image, edge_map, segmentation, face_detection, anime_output, visualization]
        )
        
        gr.Markdown("## How it works")
        gr.Markdown("""
        This system uses a hybrid approach combining traditional computer vision techniques with GAN-based stylization:
        
        1. **Face Detection**: MTCNN identifies and preserves facial features
        2. **Edge Detection**: HED-based model extracts important edges
        3. **Semantic Segmentation**: DeepLabV3 identifies different objects in the image
        4. **AnimeGAN Transformation**: Custom-trained GAN converts the image to anime style
        
        The visualization shows all intermediate steps in the transformation process.
        """)
    
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Anime Transformer Web Interface")
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained generator model')
    parser.add_argument('--share', action='store_true', help='Create a public link for the interface')
    
    args = parser.parse_args()
    
    # Create and launch Gradio interface
    demo = create_gradio_interface(model_path=args.model_path)
    demo.launch(share=args.share)