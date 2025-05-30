# import numpy as np
# from PIL import Image
# import torch
# import random

# class HandV_translation(object):

#     def __init__(self, image_gap=0):
#         self.image_gap = image_gap

#     def __call__(self, img):
#         HorV = random.randint(0, 2)
#         '''HandV translation '''
#         if HorV != 1:
#             left = img[:, :, :, :self.image_gap]
#             right = img[:, :, :, self.image_gap:]
#             img = torch.cat([right, left], dim=-1)
#         if HorV != 0:
#             top = img[:, :, :self.image_gap, :]
#             bottom = img[:, :, self.image_gap:, :]
#             img = torch.cat([bottom, top], dim=-2)
#         return img

# class AddSaltPepperNoise(object):

#     def __init__(self, density=0):
#         self.density = density

#     def __call__(self, img):
#         bs,c , h, w = img.shape
#         Pepper = torch.min(img).item()
#         Salt = torch.max(img).item()
#         Nd = self.density
#         Sd = 1 - Nd
#         mask = np.random.choice((0, 1, 2), size=(bs, 1, h, w), p=[Nd / 2.0, Nd / 2.0, Sd])  # 生成一个通道的mask
#         mask = np.repeat(mask, c, axis=1)  # 在通道的维度复制，生成彩色的mask
#         img[torch.from_numpy(mask == 0)] = Pepper/2  # 椒
#         img[torch.from_numpy(mask == 1)] = Salt/2  # 盐
#         return img

# class AddGaussianNoise(object):

#     def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
#         self.mean = mean
#         self.variance = variance
#         self.amplitude = amplitude

#     def __call__(self, img):
#         bs,c , h, w = img.shape
#         N = np.random.normal(loc=self.mean, scale=self.variance, size=(bs, 1, h, w))
#         N = np.repeat(N, c, axis=1).astype(np.float32)
#         N = self.amplitude*N
#         img = torch.from_numpy(N).cuda() + img
#         return img

# class CutMix(object):
#     def __init__(self, beta=0.0):
#         self.beta = beta

#     def rand_bbox(self, size, lam):
#         W = size[2]
#         H = size[3]
#         cut_rat = np.sqrt(1. - lam)
#         cut_w = int(W * cut_rat)
#         cut_h = int(H * cut_rat)
#         # uniform
#         cx = np.random.randint(W)
#         cy = np.random.randint(H)
#         bbx1 = np.clip(cx - cut_w // 2, 0, W)
#         bby1 = np.clip(cy - cut_h // 2, 0, H)
#         bbx2 = np.clip(cx + cut_w // 2, 0, W)
#         bby2 = np.clip(cy + cut_h // 2, 0, H)

#         return bbx1, bby1, bbx2, bby2

#     def __call__(self, image0,image1):
#         # generate mixed sample
#         lam = np.random.beta(self.beta, self.beta)
#         bbx1, bby1, bbx2, bby2 = self.rand_bbox(image0.size(), lam)
#         image0[:, :, bbx1:bbx2, bby1:bby2] = image1[:, :, bbx1:bbx2, bby1:bby2]
#         return image0,lam

# class RICAP(object):
#     def __init__(self, beta=0.0):
#         self.beta = beta

#     def __call__(self, image0,image1,image2,image3):
#         W = image0.size()[2]
#         H = image0.size()[3]
#         cx = np.random.randint(W)
#         cy = np.random.randint(H)
#         image0[:, :, 0:cx, cy:] = image1[:, :, 0:cx, cy:end]
#         image0[:, :, cx:, 0:cy] = image2[:, :, cx:, 0:cy]
#         image0[:, :, cx:, cy:] = image3[:, :, cx:, cy:]

#         return image0


import numpy as np
import torch
import random

class HandV_translation(object):
    def __init__(self, image_gap=0):
        self.image_gap = image_gap

    def __call__(self, img):
        HorV = random.randint(0, 2)
        '''HandV translation '''
        if HorV != 1:
            left = img[:, :, :, :self.image_gap]
            right = img[:, :, :, self.image_gap:]
            img = torch.cat([right, left], dim=-1)
        if HorV != 0:
            top = img[:, :, :self.image_gap, :]
            bottom = img[:, :, self.image_gap:, :]
            img = torch.cat([bottom, top], dim=-2)
        return img

class AddSaltPepperNoise(object):
    def __init__(self, density=0):
        self.density = density

    def __call__(self, img):
        bs, c, h, w = img.shape
        Pepper = torch.min(img).item()
        Salt = torch.max(img).item()
        Nd = self.density
        Sd = 1 - Nd
        mask = np.random.choice((0, 1, 2), size=(bs, 1, h, w), p=[Nd / 2.0, Nd / 2.0, Sd])  # Generate a mask for one channel
        mask = np.repeat(mask, c, axis=1)  # Replicate the mask along the channel dimension
        img[torch.from_numpy(mask == 0)] = Pepper / 2  # Apply pepper noise
        img[torch.from_numpy(mask == 1)] = Salt / 2  # Apply salt noise
        return img

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):
        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        bs, c, h, w = img.shape
        N = np.random.normal(loc=self.mean, scale=self.variance, size=(bs, 1, h, w))
        N = np.repeat(N, c, axis=1).astype(np.float32)
        N = self.amplitude * N
        img = torch.from_numpy(N).cuda() + img  # Adding Gaussian noise to the image
        return img

class CombinedTransforms(object):
    def __init__(self, image_gap=0, density=0, mean=0.0, variance=1.0, amplitude=1.0):
        self.handv_translation = HandV_translation(image_gap)
        self.salt_pepper = AddSaltPepperNoise(density)
        self.gaussian_noise = AddGaussianNoise(mean, variance, amplitude)

    def __call__(self, img):
        img = self.handv_translation(img)  # Apply HandV translation
        img = self.salt_pepper(img)  # Apply salt and pepper noise
        img = self.gaussian_noise(img)  # Apply Gaussian noise
        return img

class CutMix(object):
    def __init__(self, beta=0.0):
        self.beta = beta

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, image0, image1):
        lam = np.random.beta(self.beta, self.beta)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(image0.size(), lam)
        image0[:, :, bbx1:bbx2, bby1:bby2] = image1[:, :, bbx1:bbx2, bby1:bby2]
        return image0, lam

class RICAP(object):
    def __init__(self, beta=0.0):
        self.beta = beta

    def __call__(self, image0, image1, image2, image3):
        W = image0.size()[2]
        H = image0.size()[3]
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        image0[:, :, 0:cx, cy:] = image1[:, :, 0:cx, cy:]
        image0[:, :, cx:, 0:cy] = image2[:, :, cx:, 0:cy]
        image0[:, :, cx:, cy:] = image3[:, :, cx:, cy:]
        return image0
