import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# from train import *
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import torchvision.transforms.functional as TF
import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.io import imread


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()
        
        # Define encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_model(model_path):
    model = DeblurNet()
    # Load the model parameters from the .pth file
    if(torch.cuda.is_available()):
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    # print(model)
    return model


# Function to apply the model to an image
# Function to apply the model to an image
# Function to apply the model to an image

# Function to apply the model to an image
def apply_model(model, image_path):
    image = Image.open(image_path)
    img = image.resize((448, 256), Image.LANCZOS)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        model = model.to(device)
        output = model(image_tensor.to(device))
        output = output.squeeze(0).permute(1, 2, 0)  # Change tensor layout to HWC
        output = output.cpu()
        output = output.mul(255).byte().numpy()  # Convert to numpy array
    return output



# Function to process images in a folder
def process_images(model, input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in tqdm(os.listdir(input_folder), desc="Testing..."):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            output = apply_model(model, input_path)
            Image.fromarray(output).save(output_path)


def psnr_between_folders(folder1, folder2):
    psnr_values = []
    
    # Get list of filenames in folder1
    filenames = os.listdir(folder1)
    
    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read corresponding images from both folders
            img_path1 = os.path.join(folder1, filename)
            img_path2 = os.path.join(folder2, filename)
            img1 = imread(img_path1)
            img2 = imread(img_path2)
            
            # Compute PSNR between corresponding images
            # print(img1.shape)
            # print(img2.shape)
            psnr = peak_signal_noise_ratio(img1, img2)
            psnr_values.append(psnr)
    
    # Compute average PSNR across all images
    avg_psnr = sum(psnr_values) / len(psnr_values)

    return avg_psnr



if __name__ == '__main__':
    model_path = "final_checkpoint.pth"
    input_folder = 'custom_test/blur'
    output_folder = 'custom_test/sharp0'

    # Load model
    model = load_model(model_path)

    # Process images
    process_images(model, input_folder, output_folder)
    folder1 = "custom_test/sharp/"
    folder2 = "custom_test/sharp0/"
    avg_psnr = psnr_between_folders(folder1, folder2)
    print(f"Average PSNR between corresponding images: {avg_psnr} dB")

