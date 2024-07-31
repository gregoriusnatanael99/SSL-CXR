import os
import cv2
import argparse
from tqdm import tqdm
import numpy as np

def apply_clahe(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(image)
    return clahe_image

def augment_image(image, output_folder, base_filename):
    cv2.imwrite(os.path.join(output_folder, f"{base_filename}_clahe.jpg"), image)
    
    h_flip = cv2.flip(image, 1)
    cv2.imwrite(os.path.join(output_folder, f"{base_filename}_hflip.jpg"), h_flip)
    
    rows, cols = image.shape
    angles = [-50, -40, -30, -20, -10, 10, 20, 30, 40, 50]
    for angle in angles:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        cv2.imwrite(os.path.join(output_folder, f"{base_filename}_rot{angle}.jpg"), rotated)

def process_directory(input_folder, output_folder):
    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        
        if os.path.isdir(class_path):
            output_class_path = os.path.join(output_folder, class_name)
            os.makedirs(output_class_path, exist_ok=True)
            
            file_names = [f for f in os.listdir(class_path) if f.endswith(('jpg', 'jpeg', 'png'))]
            
            for file_name in tqdm(file_names, desc=f"Processing {class_name}", unit="file"):
                input_path = os.path.join(class_path, file_name)
                output_base_filename = os.path.splitext(file_name)[0]
                
                clahe_image = apply_clahe(input_path)
                augment_image(clahe_image, output_class_path, output_base_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str)
    parser.add_argument("--dest", type=str)
    args = parser.parse_args()

    input_base = args.dir
    output_base = args.dest

    for folder_name in ['train', 'val', 'test']:
        input_folder = os.path.join(input_base, folder_name)
        output_folder = os.path.join(output_base, folder_name)

        print(f"Processing {folder_name} directory...")
        process_directory(input_folder, output_folder)
        print(f"Finished processing {folder_name} directory.")

    print("CLAHE and Augmentation DONE :D")