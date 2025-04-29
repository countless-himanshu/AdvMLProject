import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import resnet34
from torch.utils.data import Dataset
import os
from tqdm import tqdm

# Hyperparameters
batch_size = 64  # Batch size for training
epochs = 20  # Number of epochs for training
learning_rate = 1e-3  # Learning rate for optimizer
num_classes = 10  # CIFAR-10 has 10 classes

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Dataset and DataLoader (CIFAR-10)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize the ResNet-34 model for CIFAR-10
model = resnet34(pretrained=False, num_classes=num_classes).cuda()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Output directory to save model checkpoints
output_dir = './output'
os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.cuda(), target.cuda()  # Send data to GPU
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass: compute gradients and update weights
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

    avg_loss = running_loss / len(train_loader.dataset)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Save the model checkpoint after every epoch
    torch.save(model.state_dict(), os.path.join(output_dir, 'checkpoints', f'resnet34_epoch_{epoch+1}.pth'))

print("ResNet training completed!")
