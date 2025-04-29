# import torch
# import torch.nn as nn
# import torchvision.models as models

# class ResNet34(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNet34, self).__init__()
#         # Load pretrained model
#         self.model = models.resnet34(pretrained=False)
#         # Adjust first conv layer for CIFAR-10 (3x32x32)
#         self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.model.maxpool = nn.Identity()  # Remove maxpool to preserve more spatial info
#         self.model.fc = nn.Linear(512, num_classes)  # Final layer for 10 classes

#     def forward(self, x):
#         return self.model(x)

# def resnet34(num_classes=10):
#     return ResNet34(num_classes=num_classes)



#FOR STAGE2 RUNNING WE HAVE MODIFED THIS FILE OTHERWISE ITS OK

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet34(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        # Load pretrained model
        self.model = models.resnet34(pretrained=False)
        # Adjust first conv layer for CIFAR-10 (3x32x32)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool to preserve more spatial info
        self.model.fc = nn.Linear(512, num_classes)  # Final layer for 10 classes

    def forward(self, x):
        return self.model(x)

def resnet34(num_classes=10):
    return ResNet34(num_classes=num_classes)

# # Load the model weights correctly
# def load_resnet_checkpoint(model, checkpoint_path):
#     # Load checkpoint
#     state_dict = torch.load(checkpoint_path)
#     # Modify the state_dict keys to match the model's expected keys
#     new_state_dict = {'model.' + k: v for k, v in state_dict.items()}
#     # Load the modified state_dict into the model
#     model.load_state_dict(new_state_dict)

# In resnet.py
def load_resnet_checkpoint(model, checkpoint_path):
    # Load checkpoint
    state_dict = torch.load(checkpoint_path)
    
    # Adjust the state_dict keys to match the model's expected keys
    new_state_dict = {k: v for k, v in state_dict.items()}
    
    # If the model's conv1 layer has a size mismatch (7x7 vs 3x3), we handle it here:
    if 'conv1.weight' in new_state_dict:
        # Resize the first layer if necessary to fit the model's conv1 layer shape (3x3)
        conv1_weight = new_state_dict['conv1.weight']
        if conv1_weight.shape[2] == 7 and conv1_weight.shape[3] == 7:
            new_state_dict['conv1.weight'] = conv1_weight[:, :, 0:3, 0:3]  # Crop to 3x3
    
    # Now load the adjusted state_dict into the model
    model.load_state_dict(new_state_dict, strict=False)
