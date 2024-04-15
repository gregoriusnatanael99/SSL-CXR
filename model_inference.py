import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Load pre-trained model
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)

# Define transformation function for input images
def transform_image(image):
    # Resize image to (224, 224) using PIL
    resized_image = image.resize((224, 224))
    # Convert image to PyTorch tensor
    tensor_image = transforms.ToTensor()(resized_image)
    # Normalize image
    normalized_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_image)
    return normalized_image

# Define custom dataset class
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
train_features = extract_features(train_dataset, dinov2_vits14)
test_features = extract_features(test_dataset, dinov2_vits14)

# Save the extracted features
torch.save(train_features, "train_features.pt")
torch.save(test_features, "test_features.pt")
