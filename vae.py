# # # # # # import torch
# # # # # # from torch import nn
# # # # # # from torch.nn import functional as F

# # # # # # class VAEEncoder(nn.Module):
# # # # # #     def __init__(self, latent_dim=256):
# # # # # #         super(VAEEncoder, self).__init__()
# # # # # #         self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
# # # # # #         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
# # # # # #         self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
# # # # # #         self.fc1 = nn.Linear(128 * 4 * 4, latent_dim)  # Flatten and create latent space
# # # # # #         self.fc2 = nn.Linear(128 * 4 * 4, latent_dim)

# # # # # #     def forward(self, x):
# # # # # #         x = F.relu(self.conv1(x))
# # # # # #         x = F.relu(self.conv2(x))
# # # # # #         x = F.relu(self.conv3(x))
# # # # # #         x = x.view(x.size(0), -1)  # Flatten
# # # # # #         return self.fc1(x), self.fc2(x)  # mu, log_var

# # # # # # # class VAEDecoder(nn.Module):
# # # # # # #     def __init__(self, latent_dim=256):
# # # # # # #         super(VAEDecoder, self).__init__()
# # # # # # #         self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
# # # # # # #         self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
# # # # # # #         self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
# # # # # # #         self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

# # # # # # #     def forward(self, z):
# # # # # # #         z = self.fc(z).view(z.size(0), 128, 4, 4)
# # # # # # #         z = F.relu(self.deconv1(z))
# # # # # # #         z = F.relu(self.deconv2(z))
# # # # # # #         return torch.sigmoid(self.deconv3(z))  # Final output (RGB image)
# # # # # # class VAEDecoder(nn.Module):
# # # # # #     def __init__(self, latent_dim=256):
# # # # # #         super(VAEDecoder, self).__init__()
# # # # # #         # Update the fully connected layer to output 8192 elements
# # # # # #         self.fc = nn.Linear(latent_dim, 128 * 8 * 8)  # Output size should match the reshaped size
# # # # # #         self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
# # # # # #         self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
# # # # # #         self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

# # # # # #     def forward(self, z):
# # # # # #         # Pass the latent vector through the fully connected layer
# # # # # #         z = self.fc(z).view(z.size(0), 128, 8, 8)  # Reshape to [batch_size, 128, 8, 8]
# # # # # #         z = F.relu(self.deconv1(z))
# # # # # #         z = F.relu(self.deconv2(z))
# # # # # #         return torch.sigmoid(self.deconv3(z))  # Final output (RGB image)



# # # # # # class VAE(nn.Module):
# # # # # #     def __init__(self, latent_dim=256):
# # # # # #         super(VAE, self).__init__()
# # # # # #         self.encoder = VAEEncoder(latent_dim)
# # # # # #         self.decoder = VAEDecoder(latent_dim)

# # # # # #     def forward(self, x):
# # # # # #         mu, log_var = self.encoder(x)
# # # # # #         z = self.reparameterize(mu, log_var)
# # # # # #         return self.decoder(z), mu, log_var

# # # # # #     def reparameterize(self, mu, log_var):
# # # # # #         std = torch.exp(0.5 * log_var)
# # # # # #         eps = torch.randn_like(std)
# # # # # #         return mu + eps * std

# # # # # #     def loss_function(self, recon_x, x, mu, log_var):
# # # # # #         BCE = F.binary_cross_entropy(recon_x.view(-1, 3*32*32), x.view(-1, 3*32*32), reduction='sum')
# # # # # #         # KL Divergence
# # # # # #         # See VAE paper for explanation (https://arxiv.org/abs/1312.6114)
# # # # # #         # D_KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
# # # # # #         # where mu is mean and log_var is log of variance
# # # # # #         # KL divergence between q(z|x) and p(z) (prior distribution of z)
# # # # # #         # q(z|x) is Gaussian with mean mu and variance exp(log_var)
# # # # # #         # p(z) is a standard normal distribution N(0, I)
# # # # # #         # The term is normalized by the batch size at the end of the loss function
# # # # # #         # (this is because BCE is already summed over all pixels)
# # # # # #         # In practice, we balance the contribution of the two terms with the factor beta
# # # # # #         # (in case you want to prioritize one term over the other).
# # # # # #         MSE = F.mse_loss(recon_x.view(-1, 3*32*32), x.view(-1, 3*32*32), reduction='sum')
# # # # # #         return MSE + -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


# # # # # import torch
# # # # # from torch import nn
# # # # # from torch.nn import functional as F

# # # # # class VAEEncoder(nn.Module):
# # # # #     def __init__(self, latent_dim=256):
# # # # #         super(VAEEncoder, self).__init__()
# # # # #         self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
# # # # #         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
# # # # #         self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
# # # # #         self.fc1 = nn.Linear(128 * 4 * 4, latent_dim)  # Flatten and create latent space
# # # # #         self.fc2 = nn.Linear(128 * 4 * 4, latent_dim)

# # # # #     def forward(self, x):
# # # # #         x = F.relu(self.conv1(x))
# # # # #         x = F.relu(self.conv2(x))
# # # # #         x = F.relu(self.conv3(x))
# # # # #         x = x.view(x.size(0), -1)  # Flatten
# # # # #         return self.fc1(x), self.fc2(x)  # mu, log_var

# # # # # class VAEDecoder(nn.Module):
# # # # #     def __init__(self, latent_dim=256):
# # # # #         super(VAEDecoder, self).__init__()
# # # # #         self.fc = nn.Linear(latent_dim, 128 * 16 * 16)  # Match the size for reshaping
# # # # #         self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
# # # # #         self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
# # # # #         self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

# # # # #     def forward(self, z):
# # # # #         z = self.fc(z).view(z.size(0), 128, 16, 16) #nsure correct reshaping
# # # # #         z = F.relu(self.deconv1(z))
# # # # #         z = F.relu(self.deconv2(z))
# # # # #         return torch.sigmoid(self.deconv3(z))  # Final output (RGB image)

# # # # # class VAE(nn.Module):
# # # # #     def __init__(self, latent_dim=256):
# # # # #         super(VAE, self).__init__()
# # # # #         self.encoder = VAEEncoder(latent_dim)
# # # # #         self.decoder = VAEDecoder(latent_dim)

# # # # #     def forward(self, x):
# # # # #         mu, log_var = self.encoder(x)
# # # # #         z = self.reparameterize(mu, log_var)
# # # # #         return self.decoder(z), mu, log_var

# # # # #     def reparameterize(self, mu, log_var):
# # # # #         std = torch.exp(0.5 * log_var)
# # # # #         eps = torch.randn_like(std)
# # # # #         return mu + eps * std

# # # # #     def loss_function(self, recon_x, x, mu, log_var):
# # # # #         MSE = F.mse_loss(recon_x.view(-1, 3*32*32), x.view(-1, 3*32*32), reduction='sum')
# # # # #         return MSE + -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


# # # # import torch
# # # # from torch import nn
# # # # from torch.nn import functional as F

# # # # class VAEEncoder(nn.Module):
# # # #     def __init__(self, latent_dim=256):
# # # #         super(VAEEncoder, self).__init__()
# # # #         self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
# # # #         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
# # # #         self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
# # # #         self.fc1 = nn.Linear(128 * 4 * 4, latent_dim)
# # # #         self.fc2 = nn.Linear(128 * 4 * 4, latent_dim)

# # # #     def forward(self, x):
# # # #         x = F.relu(self.conv1(x))
# # # #         x = F.relu(self.conv2(x))
# # # #         x = F.relu(self.conv3(x))
# # # #         x = x.view(x.size(0), -1)
# # # #         return self.fc1(x), self.fc2(x)

# # # # # class VAEDecoder(nn.Module):
# # # # #     def __init__(self, latent_dim=256):
# # # # #         super(VAEDecoder, self).__init__()
# # # # #         self.fc = nn.Linear(latent_dim, 128 * 16 * 16)  # 32768
# # # # #         self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
# # # # #         self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
# # # # #         self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

# # # # #     def forward(self, z):
# # # # #         z = self.fc(z).view(z.size(0), 128, 16, 16)
# # # # #         z = F.relu(self.deconv1(z))
# # # # #         z = F.relu(self.deconv2(z))
# # # # # #         return torch.sigmoid(self.deconv3(z))
# # # # # class VAEDecoder(nn.Module):
# # # # #     def __init__(self, latent_dim=256):
# # # # #         super(VAEDecoder, self).__init__()
# # # # #         self.fc = nn.Linear(latent_dim, 128 * 16 * 16)  # Ensure matching size
# # # # #         self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
# # # # #         self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
# # # # #         self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

# # # # #     def forward(self, z):
# # # # #         z = self.fc(z).view(z.size(0), 128, 16, 16)  # Ensure correct reshaping
# # # # #         z = F.relu(self.deconv1(z))
# # # # #         z = F.relu(self.deconv2(z))
# # # # #         return torch.sigmoid(self.deconv3(z))
# # # # class VAEDecoder(nn.Module):
# # # #     def __init__(self, latent_dim=256):
# # # #         super(VAEDecoder, self).__init__()
# # # #         # If latent vector size is different, make sure this layer matches
# # # #         self.fc = nn.Linear(latent_dim, 2048)  # Match the shape from the checkpoint
# # # #         self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
# # # #         self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
# # # #         self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)

# # # #     def forward(self, z):
# # # #         z = self.fc(z).view(z.size(0), 128, 16, 16)  # Ensure correct reshaping
# # # #         z = F.relu(self.deconv1(z))
# # # #         z = F.relu(self.deconv2(z))
# # # #         return torch.sigmoid(self.deconv3(z))


# # # # class VAE(nn.Module):
# # # #     def __init__(self, latent_dim=256):
# # # #         super(VAE, self).__init__()
# # # #         self.encoder = VAEEncoder(latent_dim)
# # # #         self.decoder = VAEDecoder(latent_dim)

# # # #     def forward(self, x):
# # # #         mu, log_var = self.encoder(x)
# # # #         z = self.reparameterize(mu, log_var)
# # # #         return self.decoder(z), mu, log_var

# # # #     def reparameterize(self, mu, log_var):
# # # #         std = torch.exp(0.5 * log_var)
# # # #         eps = torch.randn_like(std)
# # # #         return mu + eps * std

# # # #     def loss_function(self, recon_x, x, mu, log_var):
# # # #         MSE = F.mse_loss(recon_x.view(-1, 3*32*32), x.view(-1, 3*32*32), reduction='sum')
# # # #         return MSE + -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


# # # import torch
# # # import torch.nn as nn
# # # import torch.nn.functional as F

# # # class Encoder(nn.Module):
# # #     def __init__(self, latent_dim=256):
# # #         super(Encoder, self).__init__()
# # #         self.latent_dim = latent_dim
# # #         self.encoder = nn.Sequential(
# # #             nn.Conv2d(3, 32, 4, 2, 1),  # 32x16x16
# # #             nn.ReLU(),
# # #             nn.Conv2d(32, 64, 4, 2, 1),  # 64x8x8
# # #             nn.ReLU(),
# # #             nn.Conv2d(64, 128, 4, 2, 1),  # 128x4x4
# # #             nn.ReLU(),
# # #             nn.Conv2d(128, latent_dim, 1)  # [B, latent_dim, 4, 4]
# # #         )

# # #     def forward(self, x):
# # #         return self.encoder(x)

# # # class Decoder(nn.Module):
# # #     def __init__(self, latent_dim=256):
# # #         super(Decoder, self).__init__()
# # #         self.decoder = nn.Sequential(
# # #             nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1),  # 8x8
# # #             nn.ReLU(),
# # #             nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16
# # #             nn.ReLU(),
# # #             nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32x32
# # #             nn.ReLU(),
# # #             nn.Conv2d(32, 3, 3, 1, 1),  # Final RGB image
# # #             nn.Sigmoid()  # [0, 1] output
# # #         )

# # #     def forward(self, z):
# # #         return self.decoder(z)

# # # class VAE(nn.Module):
# # #     def __init__(self, latent_dim=256):
# # #         super(VAE, self).__init__()
# # #         self.encoder = Encoder(latent_dim)
# # #         self.decoder = Decoder(latent_dim)

# # #     def forward(self, x):
# # #         z = self.encoder(x)
# # #         recon = self.decoder(z)
# # #         return recon


# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # class Encoder(nn.Module):
# #     def __init__(self, latent_dim=256):
# #         super(Encoder, self).__init__()
# #         self.encoder = nn.Sequential(
# #             nn.Conv2d(3, 32, 4, 2, 1),  # 16x16
# #             nn.BatchNorm2d(32),
# #             nn.ReLU(),
# #             nn.Conv2d(32, 64, 4, 2, 1),  # 8x8
# #             nn.BatchNorm2d(64),
# #             nn.ReLU(),
# #             nn.Conv2d(64, 128, 4, 2, 1),  # 4x4
# #             nn.BatchNorm2d(128),
# #             nn.ReLU(),
# #             nn.Conv2d(128, latent_dim, 1)  # -> [B, latent_dim, 4, 4]
# #         )

# #     def forward(self, x):
# #         return self.encoder(x)

# # class Decoder(nn.Module):
# #     def __init__(self, latent_dim=256):
# #         super(Decoder, self).__init__()
# #         self.decoder = nn.Sequential(
# #             nn.ConvTranspose2d(latent_dim, 128, 4, 2, 1),  # 8x8
# #             nn.BatchNorm2d(128),
# #             nn.ReLU(),
# #             nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16x16
# #             nn.BatchNorm2d(64),
# #             nn.ReLU(),
# #             nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 32x32
# #             nn.BatchNorm2d(32),
# #             nn.ReLU(),
# #             nn.Conv2d(32, 3, 3, 1, 1),  # Final RGB
# #             nn.Sigmoid()  # Outputs in [0, 1]
# #         )

# #     def forward(self, z):
# #         return self.decoder(z)

# # class VAE(nn.Module):
# #     def __init__(self, latent_dim=256):
# #         super(VAE, self).__init__()
# #         self.encoder = Encoder(latent_dim)
# #         self.decoder = Decoder(latent_dim)

# #     def forward(self, x):
# #         z = self.encoder(x)
# #         recon = self.decoder(z)
# #         return recon
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Encoder(nn.Module):
#     def __init__(self, latent_dim=256):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)   # 16x16
#         self.conv2 = nn.Conv2d(64, 128, 4, 2, 1) # 8x8
#         self.conv3 = nn.Conv2d(128, 256, 4, 2, 1) # 4x4
#         self.fc1 = nn.Linear(256 * 4 * 4, latent_dim)
#         self.fc2 = nn.Linear(latent_dim, latent_dim)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         z = self.fc2(x)
#         return z.view(z.size(0), z.size(1), 1, 1)  # reshape for decoder

# class Decoder(nn.Module):
#     def __init__(self, latent_dim=256):
#         super(Decoder, self).__init__()
#         self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
#         self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 8x8
#         self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 16x16
#         self.deconv3 = nn.ConvTranspose2d(64, 3, 4, 2, 1)     # 32x32

#     def forward(self, z):
#         z = z.view(z.size(0), -1)
#         z = F.relu(self.fc(z))
#         z = z.view(z.size(0), 256, 4, 4)
#         z = F.relu(self.deconv1(z))
#         z = F.relu(self.deconv2(z))
#         z = torch.sigmoid(self.deconv3(z))
#         return z

# class VAE(nn.Module):
#     def __init__(self, latent_dim=256):
#         super(VAE, self).__init__()
#         self.encoder = Encoder(latent_dim)
#         self.decoder = Decoder(latent_dim)

#     def forward(self, x):
#         z = self.encoder(x)
#         recon = self.decoder(z)
#         return recon
import torch
import torch.nn as nn
import torch.nn.functional as F
class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.fc1 = nn.Linear(128 * 4 * 4, 2048)  # Updated to match the checkpoint
        self.fc2 = nn.Linear(2048, latent_dim)  # Matching the checkpoint architecture

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        z = F.relu(self.fc1(x))  # Pass through fc1 first
        z = self.fc2(z)          # Then pass through fc2
        return z




class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 2048)            # [B, 2048]
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)   # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(32, 3, 4, 2, 1)    # 16x16 -> 32x32

    def forward(self, z):
        z = z.view(z.size(0), -1)     # [B, latent_dim]
        z = F.relu(self.fc(z))        # [B, 2048]
        z = z.view(z.size(0), 128, 4, 4)  # reshape to conv
        z = F.relu(self.deconv1(z))   # [B, 64, 8, 8]
        z = F.relu(self.deconv2(z))   # [B, 32, 16, 16]
        z = torch.sigmoid(self.deconv3(z))  # [B, 3, 32, 32]
        return z

class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
