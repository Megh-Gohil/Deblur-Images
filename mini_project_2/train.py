import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
import os
from tqdm import tqdm
from torchsummary import summary
import argparse
from piqa import SSIM

class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1.0 - (super().forward(x,y))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

def loss_func(img1, img2, loss_fn):
    if(loss_fn == "psnr"):
        return psnr(img1, img2)
    elif(loss_fn == "mse"):
        return nn.functional.mse_loss(img1, img2).mean()
    elif(loss_fn == "ssim"):
        return ssim(img1, img2)

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def psnr(img1, img2, max_val=1.0):
    mse = nn.functional.mse_loss(img1, img2)  # Calculate MSE for the entire batch
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    mean_psnr = psnr.mean()  # Compute the mean PSNR across the batch
    return -mean_psnr


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, preprocessed_data_path=None):
        self.root_dir = root_dir
        self.transform_a = transforms.Compose([
            transforms.ToPILImage(),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 0.3)), # Configuration a
            transforms.ToTensor(),
        ])
        self.transform_b = transforms.Compose([
            transforms.ToPILImage(),
            transforms.GaussianBlur(kernel_size=7, sigma=(1.0, 1.0)),  # Configuration b
            transforms.ToTensor(),
        ])
        self.transform_c = transforms.Compose([
            transforms.ToPILImage(),
            transforms.GaussianBlur(kernel_size=11, sigma=(1.6, 1.6)),  # Configuration c
            transforms.ToTensor(),
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Add any other transformations you need
        ])
        self.target_files = [self.transform(Image.open(os.path.join(self.root_dir, 'target', img_name))).to(device) for img_name in tqdm(os.listdir(os.path.join(root_dir, 'target')))]
        self.input_files = [self.transform_a(img) for img in tqdm(self.target_files)]
        self.input_files += [self.transform_b(img) for img in tqdm(self.target_files)]
        self.input_files += [self.transform_c(img) for img in tqdm(self.target_files)]

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        return self.input_files[idx], self.target_files[idx % 24000]

class DeblurNet2(nn.Module):
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
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        # print(x.shape)
        return x

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

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64 * 28 * 16, latent_dim)
        self.fc_logvar = nn.Linear(64 * 28 * 16, latent_dim)
        
        # Decoder layers
        self.decoder_fc = nn.Linear(latent_dim, 64 * 28 * 16)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 64, 16, 28)
        x = self.decoder_conv(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def main(learning_rate, batch_size):
    print(f'learning_rate : {learning_rate}         batch_size : {batch_size}')
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add more transformations if needed (e.g., normalization)
    ])
    dataset = CustomDataset(root_dir='Dataset', preprocessed_data_path="preprocessed_data.pt")
    learning_rates = [0.005]
    batch_sizes = [8]
    loss_list = []
    for learning_rate in learning_rates:
        for batch_size in batch_sizes:

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            torch.cuda.empty_cache()
            # Initialize model
            model = DeblurNet()
            model = model.to(device)

            # Inside your main function
            model = DeblurNet().to(device)
            summary(model, input_size=(3, 448, 256))  # sAdjust input size as per your input dimensions

            # Loss and optimizer
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            # criterion = SSIMLoss()
            # Training loop
            total_step = len(dataloader)
            for epoch in range(100):
                total_loss = 0
                for batch_idx, (gaussian_images, original_images) in enumerate(tqdm(dataloader), 0):
                    # Move data to GPU
                    gaussian_images, original_images = gaussian_images.to(device), original_images.to(device)

                    # Forward pass
                    outputs = model(gaussian_images)
                    loss = psnr(outputs, original_images)
                    total_loss+=loss.item()
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                print(f"Epoch {epoch}, Loss: {total_loss}")
                loss_list.append(total_loss)
                torch.save(loss_list, "loss_list.pt")
                torch.save(model.state_dict(), f'models_8/deblurring_model_{epoch}_{learning_rate}_{batch_size}.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deblurring Model Training')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pre-trained model checkpoint')
    args = parser.parse_args()

    main(args.learning_rate, args.batch_size)
