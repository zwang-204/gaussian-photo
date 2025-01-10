import os
import json
import shutil
import math
import numpy as np
from pathlib import Path
import argparse
import subprocess
from PIL import Image
from diffusers import MarigoldDepthPipeline

def create_depth_map(image, output_path):
    """
    Generates a depth map using the Marigold model given a PIL Image object in memory.
    Saves the depth map to disk, returning its path.
    """
    print("Generating depth map using Marigold model...")
    pipe = MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-lcm-v1-0")

    # Run the pipeline directly on the PIL image (no need to open it again)
    depth = pipe(image)

    # Save the depth map
    depth_16bit = pipe.image_processor.export_depth_to_16bit_png(depth.prediction)
    depth_path = Path(output_path) / "generated_depth.png"
    depth_16bit[0].save(depth_path)
    print(f"Depth map saved to {depth_path}")
    return depth_path

def resize_image(img_path):
    max_size = args.resize_size

    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    
    # Determine the resizing scale
    scale = min(max_size / width, max_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize the image in memory
    resized_image = image.resize((new_width, new_height))
    
    return resized_image, new_width, new_height

def process_image(args):
    if not args.output_path:
        output_path = Path(args.img_path).parent
    else:
        output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True)

    images_path = output_path / "images"
    images_path.mkdir(exist_ok=True)

    sparse_path = output_path / "sparse" / "0"
    sparse_path.mkdir(exist_ok=True, parents=True)

    resized_image, width, height = resize_image(args.img_path)

    img_path = Path(args.img_path)
    shutil.copy2(img_path, images_path) 
    
    # Prevent scale from being too large
    scale = 100
    focal = math.sqrt((width/scale)**2 + (height/scale)**2)
    scaled_width = width / scale
    scaled_height = height / scale
    
    depth_map_path = args.depth_path
    if not depth_map_path:
        depth_map_path = create_depth_map(resized_image, images_path)

    depth_map = Image.open(depth_map_path) if depth_map_path else None

    # create black image
    black_img = Image.new("RGB", (width, height), (0, 0, 0))
    black_img.save(images_path / "black.jpg")

    with open(sparse_path / "cameras.txt", "w") as f:
        f.write(f"1 PINHOLE {scaled_width} {scaled_height} {focal} {focal} {scaled_width/2} {scaled_height/2}\n")
        f.write(f"2 PINHOLE {scaled_width} {scaled_height} {focal} {focal} {scaled_width/2} {scaled_height/2}\n")

    with open(sparse_path / "images.txt", "w") as f:
        f.write(f"1 1 0 0 0 0 0 {focal} 1 {img_path.name}\n\n")
        f.write(f"2 {math.sqrt(2)/2} 0 {math.sqrt(2)/2} 0 0 0 {focal} 2 black.jpg\n\n")

    img = Image.open(args.img_path)
    idx = 0
    grid_size = args.grid_size

    with open(sparse_path / "points3D.txt", "w") as f:
        for row in range(grid_size):
            for col in range(grid_size):
                idx += 1
                pixel_x = int(row / grid_size * width)
                pixel_y = int(col / grid_size * height)
                
                pixel_color = img.getpixel((
                    int(row / grid_size * img.width),
                    int(col / grid_size * img.height)
                ))

                if depth_map:
                    depth_value = depth_map.getpixel((pixel_x, pixel_y)) / 255 * args.depth_scale
                else:
                    depth_value = 0

                xyz = np.array([-width / 2, -height / 2, 0]) + np.array([pixel_x, pixel_y, depth_value])
                xyz /= scale
                f.write(f"{idx} {' '.join(map(str, xyz.tolist()))} {' '.join(map(str, pixel_color))} 0\n")

    try:
        subprocess.run(
            [
                "colmap", 
                "model_converter", 
                "--input_path", str(sparse_path),
                "--output_path", str(sparse_path),
                "--output_type", "BIN"
            ],
            check=True,
            shell=True
        )
    except FileNotFoundError as e:
        print("File not found error:", e)
    except subprocess.CalledProcessError as e:
        print("Subprocess error:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--depth_path", type=str, required=False, 
                        help="Optional path to depth map")
    parser.add_argument("--output_path", type=str, required=False, 
                        help="The default output directory is at the same level as the input image")
    parser.add_argument("--resize_size", type=int, required=False, default=600, 
                        help="Resizing to this pixel size")
    parser.add_argument("--depth_scale", type=float, required=False, default=1.0)
    parser.add_argument("--grid_size", type=int, required=False, default=20,
                        help="Size of the grids that divide the image")

    args = parser.parse_args()
    process_image(args)
