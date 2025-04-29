# # import torch
# # import torch.optim as optim
# # from torch.utils.data import DataLoader
# # import torchvision
# # from torchvision import transforms
# # from torch.utils.data import Dataset
# # import os
# # from tqdm import tqdm
# # from vae import VAE  # Import your custom VAE model
# # from torchvision.utils import save_image

# # # Hyperparameters
# # latent_dim = 256  # Latent space dimension
# # batch_size = 64  # Batch size for training
# # epochs = 20  # Number of epochs for training
# # learning_rate = 1e-3  # Learning rate for optimizer
# # image_size = 32  # Image size (CIFAR-10 images are 32x32)

# # # Loss function (MSE + KL Divergence)
# # def loss_function(recon_x, x, mu, log_var):
# #     MSE = torch.nn.functional.mse_loss(recon_x.view(-1, 3*image_size*image_size), x.view(-1, 3*image_size*image_size), reduction='sum')
# #     # KL Divergence
# #     return MSE + -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

# # # Dataset and DataLoader (CIFAR-10)
# # transform = transforms.Compose([
# #     transforms.ToTensor(),
# #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
# # ])

# # train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# # # Initialize the VAE model and optimizer
# # vae = VAE(latent_dim).cuda()
# # optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# # # Output directory to save model checkpoints and generated images
# # output_dir = './output'
# # os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
# # os.makedirs(os.path.join(output_dir, 'generated_images'), exist_ok=True)

# # # Training loop
# # for epoch in range(epochs):
# #     vae.train()
# #     running_loss = 0.0
# #     for batch_idx, (data, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
# #         data = data.cuda()  # Send data to GPU
# #         optimizer.zero_grad()

# #         # Forward pass: compute reconstruction and loss
# #         recon_data, mu, log_var = vae(data)
# #         loss = loss_function(recon_data, data, mu, log_var)

# #         # Backward pass: compute gradients and update weights
# #         loss.backward()
# #         optimizer.step()

# #         running_loss += loss.item()

# #     avg_loss = running_loss / len(train_loader.dataset)
# #     print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# #     # Save the model checkpoint after every epoch
# #     torch.save(vae.state_dict(), os.path.join(output_dir, 'checkpoints', f'vae_epoch_{epoch+1}.pth'))

# #     # Optionally, save generated images to monitor progress
# #     if (epoch + 1) % 5 == 0:  # Save images every 5 epochs
# #         with torch.no_grad():
# #             vae.eval()
# #             sample = torch.randn(64, latent_dim).cuda()  # Sample from the latent space
# #             generated_images = vae.decoder(sample).cpu()
# #             save_image(generated_images, os.path.join(output_dir, 'generated_images', f'generated_epoch_{epoch+1}.png'))

# #     vae.train()  # Reset model to train mode after evaluation

# # print("Training completed!")



# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import torchvision
# from torchvision import transforms
# from torch.utils.data import Dataset
# import os
# from tqdm import tqdm
# from vae import VAE  # Import your custom VAE model
# from torchvision.utils import save_image

# # Hyperparameters
# latent_dim = 256  # Latent space dimension
# batch_size = 64  # Batch size for training
# epochs = 20  # Number of epochs for training
# learning_rate = 1e-3  # Learning rate for optimizer
# image_size = 32  # Image size (CIFAR-10 images are 32x32)

# # Loss function (MSE + KL Divergence)
# def loss_function(recon_x, x, mu, log_var):
#     MSE = torch.nn.functional.mse_loss(recon_x.view(-1, 3*image_size*image_size), x.view(-1, 3*image_size*image_size), reduction='sum')
#     # KL Divergence
#     return MSE + -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

# # Dataset and DataLoader (CIFAR-10)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
# ])

# train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# # Initialize the VAE model and optimizer
# vae = VAE(latent_dim).cuda()
# optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# # Output directory to save model checkpoints and generated images
# output_dir = './output'
# os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
# os.makedirs(os.path.join(output_dir, 'generated_images'), exist_ok=True)

# # Training loop
# for epoch in range(epochs):
#     vae.train()
#     running_loss = 0.0
#     for batch_idx, (data, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
#         data = data.cuda()  # Send data to GPU
#         optimizer.zero_grad()

#         # Forward pass: compute reconstruction and loss
#         recon_data, mu, log_var = vae(data)
#         loss = loss_function(recon_data, data, mu, log_var)

#         # Backward pass: compute gradients and update weights
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     avg_loss = running_loss / len(train_loader.dataset)
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

#     # Save the model checkpoint after every epoch
#     torch.save(vae.state_dict(), os.path.join(output_dir, 'checkpoints', f'vae_epoch_{epoch+1}.pth'))

#     # Optionally, save generated images to monitor progress
#     if (epoch + 1) % 5 == 0:  # Save images every 5 epochs
#         with torch.no_grad():
#             vae.eval()
#             sample = torch.randn(64, latent_dim).cuda()  # Sample from the latent space
#             generated_images = vae.decoder(sample).cpu()
#             save_image(generated_images, os.path.join(output_dir, 'generated_images', f'generated_epoch_{epoch+1}.png'))

#     vae.train()  # Reset model to train mode after evaluation

# print("Training completed!")


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import os
from tqdm import tqdm
from vae import VAE  # Ensure this file is in the same directory or in your PYTHONPATH
from torchvision.utils import save_image

# Hyperparameters
latent_dim = 256
batch_size = 64
epochs = 20
learning_rate = 1e-3
image_size = 32

# Loss function (MSE + KL divergence)
def loss_function(recon_x, x, mu, log_var):
    mse = torch.nn.functional.mse_loss(recon_x.view(-1, 3 * image_size * image_size),
                                       x.view(-1, 3 * image_size * image_size),
                                       reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return mse + kld

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Load CIFAR-10 (with download enabled)
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Output directories
output_dir = './output'
os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'generated_images'), exist_ok=True)

# Training loop
for epoch in range(epochs):
    vae.train()
    running_loss = 0.0

    for batch_idx, (data, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"):
        data = data.to(device)
        optimizer.zero_grad()
        recon_data, mu, log_var = vae(data)
        loss = loss_function(recon_data, data, mu, log_var)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    # Save model checkpoint
    torch.save(vae.state_dict(), os.path.join(output_dir, 'checkpoints', f'vae_epoch_{epoch+1}.pth'))

    # Save sample generated images
    if (epoch + 1) % 5 == 0:
        vae.eval()
        with torch.no_grad():
            sample = torch.randn(64, latent_dim).to(device)
            generated = vae.decoder(sample).cpu()
            save_image(generated, os.path.join(output_dir, 'generated_images', f'generated_epoch_{epoch+1}.png'), normalize=True)
        vae.train()

print("✅ VAE training completed and saved in './output'")
