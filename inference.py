from transformers import ConvNextV2Model
import torch

# load pretrained model
convnextv2 = ConvNextV2Model.from_pretrained("facebook/convnextv2-tiny-1k-224")
# load pretrained weight
# weight = torch.load('convnextv2_atto_1k_224_fcmae.pt')

# process input data
from PIL import Image
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Dataset

# transform_train = transforms.Compose([
#             transforms.RandomResizedCrop((224,224), scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def transform_image(image):
    resized_image = image.resize((224, 224))
    tensor_image = transforms.ToTensor()(resized_image)
    normalized_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_image)
    return normalized_image

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_paths = [os.path.join(self.data_dir, img) for img in os.listdir(self.data_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        transformed_image = transform_image(image)
        return transformed_image

# Load images from train and test folders
train_data_dir = "dataset/train"
test_data_dir = "dataset/test"

train_dataset = CustomDataset(train_data_dir)
test_dataset = CustomDataset(test_data_dir)

# Function to extract features from images
def extract_features(dataset, model):
    features = []
    for image in dataset:
        with torch.no_grad():
            outputs = model(image.unsqueeze(0))  # Add batch dimension
        features.append(outputs)
    return features

# Extract features from train and test datasets
train_features = extract_features(train_dataset, convnextv2)
test_features = extract_features(test_dataset, convnextv2)

# Save the extracted features
torch.save(train_features, "train_features.pt")
torch.save(test_features, "test_features.pt")


import pandas as pd
import os

df = pd.read_csv('nih-cxr-lt_single-label_train.csv')
classes = df.columns
classes = classes[1:-1]
print(classes)

# make dir
for i in classes:
    os.mkdir(i)

# shutil
