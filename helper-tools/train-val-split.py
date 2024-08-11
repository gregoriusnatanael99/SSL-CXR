import os
import shutil
import random
from datetime import datetime as dt
import argparse
import pickle
import csv
import numpy as np
random.seed(42)

source_folders = [r"./images"]

test_list_txt = r"test_list.txt"
train_val_list_txt = r"train_val_list.txt"

train_folder = r"train"
val_folder = r"val"
test_folder = r"test"

csv_file_path = r"Data_Entry_2017_v2020.csv"

train_output_file = "train_output.txt"
val_output_file = "val_output.txt"
test_output_file = "test_output.txt"


def move_test(test_list, source_folders, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    with open(test_list, 'r') as f:
        image_names = f.read().splitlines()

    for source_folder in source_folders:
        for image_name in image_names:
            source_path = os.path.join(source_folder, image_name)
            if os.path.exists(source_path):
                destination_path = os.path.join(destination_folder, image_name)
                shutil.move(source_path, destination_path)
                print(f"Moved {image_name} to {destination_folder}")

    print("All images moved to test.")

def move_train(train_val_list, source_folders, train_folder, val_folder):
    random.seed()
    
    # Membuat folder train dan val jika belum ada
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    # Membaca nama file dari train_val_list.txt
    with open(train_val_list, 'r') as f:
        train_val_names = f.read().splitlines()

    # Memindahkan sisa gambar ke folder train
    for source_folder in source_folders:
        for image_name in os.listdir(source_folder):
            if image_name in train_val_names:
                source_path = os.path.join(source_folder, image_name)
                if os.path.exists(source_path):
                    destination_path = os.path.join(train_folder, image_name)
                    shutil.move(source_path, destination_path)
                    print(f"Moved {image_name} to {train_folder}")

    print("All images moved to train and val.")

def move_subfolder_class(source_folder, destination_folder, csv_file_path):
    # Membaca file CSV
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        
        # Membaca header
        header = next(csv_reader)
        image_index_idx = header.index('Image Index')
        finding_labels_idx = header.index('Finding Labels')
        
        # Membaca setiap baris data
        for row in csv_reader:
            image_name = row[image_index_idx]
            labels = row[finding_labels_idx].split('|')
            
            # Jika label hanya satu, maka pindahkan gambar
            if len(labels) == 1:
                label = labels[0]
                label_folder = os.path.join(destination_folder, label)
                if not os.path.exists(label_folder):
                    os.makedirs(label_folder)
                
                source_path = os.path.join(source_folder, image_name)
                destination_path = os.path.join(destination_folder, label, image_name)
                if os.path.exists(source_path):
                    shutil.move(source_path, destination_path)
                    print(f"Moved {image_name} to {destination_folder}/{label}")

def train_val_split_from_folder(tgt_folder,dest_folder,sample_rate:float=0.2):
    for curr_class in os.listdir(tgt_folder):
        images = os.listdir(os.path.join(tgt_folder,curr_class))
        val_data = random.sample(images,int(np.floor(len(images)*sample_rate)))
        # print(val_data)
        os.makedirs(os.path.join(dest_folder,curr_class),exist_ok=True)
        for i in val_data:
            shutil.move(os.path.join(tgt_folder,curr_class,i), os.path.join(dest_folder,curr_class,i))

def count_images_per_class(folder):
    class_counts = {}
    # Melakukan iterasi melalui setiap subfolder (class)
    for root, dirs, files in os.walk(folder):
        for dir in dirs:
            class_folder = os.path.join(root, dir)
            # Menghitung jumlah gambar dalam subfolder (class)
            class_count = len(os.listdir(class_folder))
            class_counts[dir] = class_count
    return class_counts

# Function untuk write output to txt
def write_output_to_txt(output_file, data):
    with open(output_file, 'w') as f:
        for label, count in data.items():
            f.write(f"{label}: {count}\n")

def save_image_list(tgt_folder):
    img_list = []
    for curr_class in os.listdir(tgt_folder):
        img_list.extend(os.listdir(os.path.join(tgt_folder,curr_class)))
    # print(img_list)
    # print(len(img_list))
    with open('val_imgs.pkl', 'wb') as f:
        pickle.dump(img_list, f)

def begin_splitting():
    move_test(test_list_txt, source_folders, test_folder)
    move_train(train_val_list_txt, source_folders, train_folder, val_folder)
    move_subfolder_class(test_folder, test_folder, csv_file_path)
    move_subfolder_class(train_folder, train_folder, csv_file_path)
    train_val_split_from_folder(tgt_folder=train_folder,dest_folder=val_folder)

    # train_class_counts = count_images_per_class(train_folder)
    # val_class_counts = count_images_per_class(val_folder)
    # test_class_counts = count_images_per_class(test_folder)

    # print("\nTrain:")
    # for label, count in train_class_counts.items(): print(f"- {label}: {count}")
    # print("\nVal:")
    # for label, count in val_class_counts.items(): print(f"- {label}: {count}")
    # print("\nTest:")
    # for label, count in test_class_counts.items(): print(f"- {label}: {count}")

    # # Write output to text files
    # write_output_to_txt(train_output_file, train_class_counts)
    # write_output_to_txt(val_output_file, val_class_counts)
    # write_output_to_txt(test_output_file, test_class_counts)

    # print("\nOutput has been written to a txt file:")
    # print(f"- {train_output_file}")
    # print(f"- {val_output_file}")
    # print(f"- {test_output_file}")
    # save_image_list(val_folder)


if __name__ == "__main__":
    
    begin_splitting()
