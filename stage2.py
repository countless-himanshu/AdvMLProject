# # # # # # # # # # # # # # import argparse, os
# # # # # # # # # # # # # # from omegaconf import OmegaConf
# # # # # # # # # # # # # # from einops import rearrange
# # # # # # # # # # # # # # from torchvision import datasets
# # # # # # # # # # # # # # from torch import autocast
# # # # # # # # # # # # # # from contextlib import nullcontext
# # # # # # # # # # # # # # import sys
# # # # # # # # # # # # # # sys.path.append("./")
# # # # # # # # # # # # # # sys.path.append("../")
# # # # # # # # # # # # # # from ldm.util import instantiate_from_config
# # # # # # # # # # # # # # from ldm.models.diffusion.ddim import DDIMSampler
# # # # # # # # # # # # # # from my_transforms import *
# # # # # # # # # # # # # # import warnings
# # # # # # # # # # # # # # warnings.filterwarnings("ignore")
# # # # # # # # # # # # # # dataset='cifar10' # STL10 cifar10

# # # # # # # # # # # # # # import torchvision.transforms as tfs

# # # # # # # # # # # # # # def load_model_from_config(config, ckpt, verbose=False):
# # # # # # # # # # # # # #     print(f"Loading model from {ckpt}")
# # # # # # # # # # # # # #     pl_sd = torch.load(ckpt, map_location="cpu")
# # # # # # # # # # # # # #     if "global_step" in pl_sd:
# # # # # # # # # # # # # #         print(f"Global Step: {pl_sd['global_step']}")
# # # # # # # # # # # # # #     sd = pl_sd["state_dict"]
# # # # # # # # # # # # # #     model = instantiate_from_config(config.model)
# # # # # # # # # # # # # #     m, u = model.load_state_dict(sd, strict=False)
# # # # # # # # # # # # # #     if len(m) > 0 and verbose:
# # # # # # # # # # # # # #         print("missing keys:")
# # # # # # # # # # # # # #         print(m)
# # # # # # # # # # # # # #     if len(u) > 0 and verbose:
# # # # # # # # # # # # # #         print("unexpected keys:")
# # # # # # # # # # # # # #         print(u)

# # # # # # # # # # # # # #     model.cuda()
# # # # # # # # # # # # # #     model.eval()
# # # # # # # # # # # # # #     return model

# # # # # # # # # # # # # # def from_img_generate_sample(sampler,model,c,z_enc,t_enc,scale,uc):
# # # # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # # # #         # decode it
# # # # # # # # # # # # # #         samples_ddim = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
# # # # # # # # # # # # # #                                  unconditional_conditioning=uc, )
# # # # # # # # # # # # # #         x_samples_ddim = model.decode_first_stage(samples_ddim)
# # # # # # # # # # # # # #     return x_samples_ddim

# # # # # # # # # # # # # # def saveImageto(x_samples_ddim, sample_path, class_id, epoch, i):
# # # # # # # # # # # # # #     x_samples_ddim = tfs.Resize(224)(x_samples_ddim)
# # # # # # # # # # # # # #     x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
# # # # # # # # # # # # # #     for x_sample in x_samples_ddim:  # x_checked_image_torch:
# # # # # # # # # # # # # #         x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
# # # # # # # # # # # # # #         img = Image.fromarray(x_sample.astype(np.uint8))
# # # # # # # # # # # # # #         img.save(os.path.join(sample_path, f"{class_id}/{epoch}_{i:04}.png"))

# # # # # # # # # # # # # # def train_transforms(inputs):
# # # # # # # # # # # # # #     img_size = inputs.size(-1)
# # # # # # # # # # # # # #     image_gap = random.randint(2, 8)
# # # # # # # # # # # # # #     random_trans = tfs.RandomOrder([
# # # # # # # # # # # # # #         tfs.RandomApply(
# # # # # # # # # # # # # #             [
# # # # # # # # # # # # # #                 tfs.Resize((img_size + image_gap, img_size + image_gap)),
# # # # # # # # # # # # # #                 tfs.CenterCrop((img_size, img_size)),
# # # # # # # # # # # # # #             ],0.5
# # # # # # # # # # # # # #         ),
# # # # # # # # # # # # # #         tfs.RandomHorizontalFlip(),
# # # # # # # # # # # # # #         tfs.RandomApply(
# # # # # # # # # # # # # #             [tfs.RandomRotation(image_gap)],0.5
# # # # # # # # # # # # # #         ),
# # # # # # # # # # # # # #         tfs.RandomApply(
# # # # # # # # # # # # # #             [HandV_translation(image_gap)],0.5
# # # # # # # # # # # # # #         ),
# # # # # # # # # # # # # #         tfs.RandomApply(
# # # # # # # # # # # # # #             [
# # # # # # # # # # # # # #                 tfs.Pad([int(image_gap / 2), int(image_gap / 2),
# # # # # # # # # # # # # #                                   int(image_gap / 2), int(image_gap / 2)]),
# # # # # # # # # # # # # #                 tfs.Resize((img_size, img_size)),
# # # # # # # # # # # # # #             ],0.5
# # # # # # # # # # # # # #         ),
# # # # # # # # # # # # # #         tfs.RandomApply(
# # # # # # # # # # # # # #             [tfs.GaussianBlur(3, sigma=(0.1, 1.0))],0.5
# # # # # # # # # # # # # #         ),
# # # # # # # # # # # # # #         tfs.RandomApply(
# # # # # # # # # # # # # #             [AddGaussianNoise(0.0, 1.0, 0.01)],0.5
# # # # # # # # # # # # # #         ),
# # # # # # # # # # # # # #         tfs.RandomApply(
# # # # # # # # # # # # # #             [AddSaltPepperNoise(0.01)],0.5
# # # # # # # # # # # # # #         ),
# # # # # # # # # # # # # #         tfs.RandomApply(
# # # # # # # # # # # # # #             [tfs.RandomAffine(image_gap)],0.5
# # # # # # # # # # # # # #         ),
# # # # # # # # # # # # # #         tfs.RandomApply(
# # # # # # # # # # # # # #             [tfs.RandomErasing(scale=(0.02, 0.22))],0.5
# # # # # # # # # # # # # #         ),
# # # # # # # # # # # # # #     ])
# # # # # # # # # # # # # #     return random_trans(inputs)

# # # # # # # # # # # # # # def mixup_data(x_s, alpha=1.0):
# # # # # # # # # # # # # #     '''Returns mixed inputs, pairs of targets, and lambda'''
# # # # # # # # # # # # # #     if alpha > 0:
# # # # # # # # # # # # # #         lam = np.random.beta(alpha, alpha)
# # # # # # # # # # # # # #     else:
# # # # # # # # # # # # # #         lam = 1
# # # # # # # # # # # # # #     batch_size = x_s.size()[0]
# # # # # # # # # # # # # #     if batch_size == 1:
# # # # # # # # # # # # # #         return x_s
# # # # # # # # # # # # # #     random_mix = random.randint(0, 1)

# # # # # # # # # # # # # #     index0 = np.random.randint(0, batch_size - 1)
# # # # # # # # # # # # # #     a = train_transforms(x_s[index0:index0+1])
# # # # # # # # # # # # # #     if random_mix == 0:#single-data augmentation
# # # # # # # # # # # # # #         mixed_x = a
# # # # # # # # # # # # # #     else:#multi-data augmentation
# # # # # # # # # # # # # #         index1 = np.random.randint(0, batch_size-1)
# # # # # # # # # # # # # #         b = train_transforms(x_s[index1:index1+1])
# # # # # # # # # # # # # #         random_mix = random.randint(0, 2)
# # # # # # # # # # # # # #         if random_mix == 0:# MixUp
# # # # # # # # # # # # # #             mixed_x = lam * a + (1 - lam) * b
# # # # # # # # # # # # # #         elif random_mix ==1:# CutMix
# # # # # # # # # # # # # #             mixed_x, lam = CutMix(1.0)(a, b)
# # # # # # # # # # # # # #         elif random_mix ==2:# RICAP
# # # # # # # # # # # # # #             index1 = np.random.randint(0, batch_size-1)
# # # # # # # # # # # # # #             b = train_transforms(x_s[index1:index1+1])
# # # # # # # # # # # # # #             index2 = np.random.randint(0, batch_size-1)
# # # # # # # # # # # # # #             c = train_transforms(x_s[index2:index2+1])
# # # # # # # # # # # # # #             index3 = np.random.randint(0, batch_size-1)
# # # # # # # # # # # # # #             d = train_transforms(x_s[index3:index3+1])
# # # # # # # # # # # # # #             mixed_x = RICAP()(a, b, c, d)
# # # # # # # # # # # # # #     return mixed_x

# # # # # # # # # # # # # # def main(opt):

# # # # # # # # # # # # # #     config = OmegaConf.load(f"{opt.config}")
# # # # # # # # # # # # # #     model = load_model_from_config(config, f"{opt.ckpt}")

# # # # # # # # # # # # # #     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# # # # # # # # # # # # # #     model = model.to(device)

# # # # # # # # # # # # # #     sampler = DDIMSampler(model)

# # # # # # # # # # # # # #     os.makedirs(opt.outdir, exist_ok=True)
# # # # # # # # # # # # # #     outpath = os.path.join(opt.outdir)

# # # # # # # # # # # # # #     batch_size = opt.n_samples

# # # # # # # # # # # # # #     sample_path = os.path.join(outpath, "generate_images")
# # # # # # # # # # # # # #     os.makedirs(sample_path, exist_ok=True)

# # # # # # # # # # # # # #     sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
# # # # # # # # # # # # # #     assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
# # # # # # # # # # # # # #     t_enc = int(opt.strength * opt.ddim_steps)
# # # # # # # # # # # # # #     print(f"target t_enc is {t_enc} steps")

# # # # # # # # # # # # # #     codebooks = torch.load(opt.codebookdir)
# # # # # # # # # # # # # #     print(f"load codebooks from '{opt.codebookdir}'")

# # # # # # # # # # # # # #     classnames = []
# # # # # # # # # # # # # #     with open(dataset+'.txt') as f:
# # # # # # # # # # # # # #         for clas in f.readlines():
# # # # # # # # # # # # # #             classnames.append(clas)
# # # # # # # # # # # # # #     precision_scope = autocast if opt.precision == "autocast" else nullcontext
# # # # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # # # #         with precision_scope("cuda"):
# # # # # # # # # # # # # #             with model.ema_scope():
# # # # # # # # # # # # # #                 c1 = model.get_learned_conditioning(classnames)
# # # # # # # # # # # # # #                 uc = None
# # # # # # # # # # # # # #                 if opt.scale != 1.0:
# # # # # # # # # # # # # #                     uc = model.get_learned_conditioning([""]*batch_size)

# # # # # # # # # # # # # #     epoch = 0
# # # # # # # # # # # # # #     class_num = len(classnames)
# # # # # # # # # # # # # #     for i in range(class_num):
# # # # # # # # # # # # # #         os.makedirs(os.path.join(sample_path, str(i)),exist_ok=True)
# # # # # # # # # # # # # #     generate_num = 5 # 1 epoch generate image num

# # # # # # # # # # # # # #     while True:
# # # # # # # # # # # # # #         for class_id in range(class_num): # #
# # # # # # # # # # # # # #             text_cond=c1[class_id:class_id + 1].repeat(batch_size,1,1)
# # # # # # # # # # # # # #             for i in range(generate_num):
# # # # # # # # # # # # # #                 init_latent = mixup_data(torch.cat([codebooks[class_id][li][1] for li in range(len(codebooks[class_id]))]))
# # # # # # # # # # # # # #                 with torch.no_grad():
# # # # # # # # # # # # # #                     with precision_scope("cuda"):
# # # # # # # # # # # # # #                         with model.ema_scope():
# # # # # # # # # # # # # #                             z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc] * batch_size).to(device))
# # # # # # # # # # # # # #                             x_samples_ddim = from_img_generate_sample(sampler, model, text_cond, z_enc, t_enc, opt.scale, uc)
# # # # # # # # # # # # # #                 saveImageto(x_samples_ddim, sample_path, class_id, epoch, i)
# # # # # # # # # # # # # #             print(str(epoch) +" epoch, class=" + str(class_id) + ", completed")
# # # # # # # # # # # # # #         epoch+=1

# # # # # # # # # # # # # # if __name__ == "__main__":
# # # # # # # # # # # # # #     parser = argparse.ArgumentParser()
# # # # # # # # # # # # # #     parser.add_argument(
# # # # # # # # # # # # # #         "--outdir",
# # # # # # # # # # # # # #         type=str,
# # # # # # # # # # # # # #         nargs="?",
# # # # # # # # # # # # # #         help="dir to write results to",
# # # # # # # # # # # # # #         default="outputs/txt2img-tmp"
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     parser.add_argument(
# # # # # # # # # # # # # #         "--codebookdir",
# # # # # # # # # # # # # #         type=str,
# # # # # # # # # # # # # #         nargs="?",
# # # # # # # # # # # # # #         help="dir of codebook",
# # # # # # # # # # # # # #         default="./codebooks/codebook_10_cifar10.pth"
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     parser.add_argument(
# # # # # # # # # # # # # #         "--ddim_steps",
# # # # # # # # # # # # # #         type=int,
# # # # # # # # # # # # # #         default=50,
# # # # # # # # # # # # # #         help="number of ddim sampling steps",
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     parser.add_argument(
# # # # # # # # # # # # # #         "--ddim_eta",
# # # # # # # # # # # # # #         type=float,
# # # # # # # # # # # # # #         default=0.0,
# # # # # # # # # # # # # #         help="ddim eta (eta=0.0 corresponds to deterministic sampling",
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     parser.add_argument(
# # # # # # # # # # # # # #         "--H",
# # # # # # # # # # # # # #         type=int,
# # # # # # # # # # # # # #         default=512,
# # # # # # # # # # # # # #         help="image height, in pixel space",
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     parser.add_argument(
# # # # # # # # # # # # # #         "--W",
# # # # # # # # # # # # # #         type=int,
# # # # # # # # # # # # # #         default=512,
# # # # # # # # # # # # # #         help="image width, in pixel space",
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     parser.add_argument(
# # # # # # # # # # # # # #         "--C",
# # # # # # # # # # # # # #         type=int,
# # # # # # # # # # # # # #         default=4,
# # # # # # # # # # # # # #         help="latent channels",
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     parser.add_argument(
# # # # # # # # # # # # # #         "--f",
# # # # # # # # # # # # # #         type=int,
# # # # # # # # # # # # # #         default=8,
# # # # # # # # # # # # # #         help="downsampling factor",
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     parser.add_argument(
# # # # # # # # # # # # # #         "--n_samples",
# # # # # # # # # # # # # #         type=int,
# # # # # # # # # # # # # #         default=1,
# # # # # # # # # # # # # #         help="how many samples to produce for each given prompt. A.k.a. batch size",
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     parser.add_argument(
# # # # # # # # # # # # # #         "--scale",
# # # # # # # # # # # # # #         type=float,
# # # # # # # # # # # # # #         default=7.5,
# # # # # # # # # # # # # #         help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     parser.add_argument(
# # # # # # # # # # # # # #         "--config",
# # # # # # # # # # # # # #         type=str,
# # # # # # # # # # # # # #         default="../configs/stable-diffusion/v1-inference.yaml",
# # # # # # # # # # # # # #         help="path to config which constructs model",
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     parser.add_argument(
# # # # # # # # # # # # # #         "--ckpt",
# # # # # # # # # # # # # #         type=str,
# # # # # # # # # # # # # #         default="../models/ldm/stable-diffusion-v1/model.ckpt",
# # # # # # # # # # # # # #         help="path to checkpoint of model",
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     parser.add_argument(
# # # # # # # # # # # # # #         "--precision",
# # # # # # # # # # # # # #         type=str,
# # # # # # # # # # # # # #         help="evaluate at this precision",
# # # # # # # # # # # # # #         choices=["full", "autocast"],
# # # # # # # # # # # # # #         default="autocast"
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     parser.add_argument(
# # # # # # # # # # # # # #         "--strength",
# # # # # # # # # # # # # #         type=float,
# # # # # # # # # # # # # #         default=0.75,
# # # # # # # # # # # # # #         help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
# # # # # # # # # # # # # #     )
# # # # # # # # # # # # # #     opt = parser.parse_args()

# # # # # # # # # # # # # #     main(opt)



# # # # # # # # # # # # # import os
# # # # # # # # # # # # # import torch
# # # # # # # # # # # # # from torchvision.utils import save_image
# # # # # # # # # # # # # from vae import VAE  # your custom VAE
# # # # # # # # # # # # # from resnet import resnet34  # your custom classifier
# # # # # # # # # # # # # from torchvision import transforms
# # # # # # # # # # # # # from PIL import Image
# # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # from tqdm import tqdm

# # # # # # # # # # # # # # Configs
# # # # # # # # # # # # # latent_dim = 256
# # # # # # # # # # # # # vae_checkpoint = "../output/checkpoints/vae_epoch_20.pth"
# # # # # # # # # # # # # latent_code_dir = "../result1"
# # # # # # # # # # # # # output_image_dir = "../result2/images"
# # # # # # # # # # # # # os.makedirs(output_image_dir, exist_ok=True)

# # # # # # # # # # # # # # Load decoder
# # # # # # # # # # # # # vae = VAE(latent_dim).cuda()
# # # # # # # # # # # # # vae.load_state_dict(torch.load(vae_checkpoint))
# # # # # # # # # # # # # vae.eval()

# # # # # # # # # # # # # # Load classifier
# # # # # # # # # # # # # classifier = resnet34(num_classes=10).cuda()
# # # # # # # # # # # # # classifier.load_state_dict(torch.load("../resnet_checkpoint.pth"))  # make sure this file exists
# # # # # # # # # # # # # classifier.eval()

# # # # # # # # # # # # # # Transform for classifier input
# # # # # # # # # # # # # transform = transforms.Compose([
# # # # # # # # # # # # #     transforms.Resize((32, 32)),
# # # # # # # # # # # # #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# # # # # # # # # # # # # ])

# # # # # # # # # # # # # # Load latent codes
# # # # # # # # # # # # # latent_codes = torch.load(os.path.join(latent_code_dir, "latent_codes.pt"))  # shape: [N, latent_dim]
# # # # # # # # # # # # # print(f"Loaded latent codes: {latent_codes.shape}")

# # # # # # # # # # # # # # Generate images and classify
# # # # # # # # # # # # # saved = 0
# # # # # # # # # # # # # with torch.no_grad():
# # # # # # # # # # # # #     for idx in tqdm(range(len(latent_codes))):
# # # # # # # # # # # # #         z = latent_codes[idx].unsqueeze(0).cuda()  # [1, latent_dim]
# # # # # # # # # # # # #         img = vae.decoder(z).cpu()  # [1, 3, 32, 32]

# # # # # # # # # # # # #         # Preprocess image for classifier
# # # # # # # # # # # # #         img_for_classifier = transform(img.squeeze(0)).unsqueeze(0).cuda()

# # # # # # # # # # # # #         # Predict class
# # # # # # # # # # # # #         output = classifier(img_for_classifier)
# # # # # # # # # # # # #         pred = output.argmax(dim=1).item()

# # # # # # # # # # # # #         # Save the image with class label in name
# # # # # # # # # # # # #         save_path = os.path.join(output_image_dir, f"{saved:05d}_class{pred}.png")
# # # # # # # # # # # # #         save_image(img, save_path)
# # # # # # # # # # # # #         saved += 1

# # # # # # # # # # # # # print(f"✅ Stage 2 completed: {saved} images saved to {output_image_dir}")



# # # # # # # # # # # # import torch
# # # # # # # # # # # # from torch.utils.data import DataLoader
# # # # # # # # # # # # from torchvision import transforms
# # # # # # # # # # # # from resnet import resnet34  # Import your ResNet model
# # # # # # # # # # # # from vae import VAE  # Import your VAE model
# # # # # # # # # # # # from torchvision.utils import save_image
# # # # # # # # # # # # import os
# # # # # # # # # # # # import numpy as np

# # # # # # # # # # # # # Hyperparameters
# # # # # # # # # # # # latent_dim = 256  # Latent space dimension
# # # # # # # # # # # # batch_size = 64  # Batch size for training
# # # # # # # # # # # # epochs = 20  # Number of epochs for training
# # # # # # # # # # # # image_size = 32  # Image size (CIFAR-10 images are 32x32)

# # # # # # # # # # # # # Output directory for saving generated images and other results
# # # # # # # # # # # # output_dir = './output'
# # # # # # # # # # # # os.makedirs(os.path.join(output_dir, 'generated_images'), exist_ok=True)
# # # # # # # # # # # # os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

# # # # # # # # # # # # # Load the trained ResNet model
# # # # # # # # # # # # model = resnet34(pretrained=False, num_classes=10).cuda()
# # # # # # # # # # # # model.load_state_dict(torch.load('path_to_your_trained_model.pth'))  # Load your checkpoint file
# # # # # # # # # # # # model.eval()  # Set the model to evaluation mode

# # # # # # # # # # # # # Load the trained VAE model (same model as used in Stage 1)
# # # # # # # # # # # # vae = VAE(latent_dim=latent_dim).cuda()
# # # # # # # # # # # # vae.load_state_dict(torch.load('path_to_your_trained_vae_model.pth'))  # Load the trained VAE checkpoint

# # # # # # # # # # # # # Loss function for Stage 2 (if necessary for optimizing latent codes)
# # # # # # # # # # # # def loss_function(recon_x, x, mu, log_var):
# # # # # # # # # # # #     MSE = torch.nn.functional.mse_loss(recon_x.view(-1, 3*image_size*image_size), x.view(-1, 3*image_size*image_size), reduction='sum')
# # # # # # # # # # # #     # KL Divergence
# # # # # # # # # # # #     return MSE + -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

# # # # # # # # # # # # # Define function to classify augmented images from latent codes
# # # # # # # # # # # # def classify_latent_codes(latent_codes):
# # # # # # # # # # # #     latent_codes = latent_codes.cuda()
# # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # #         generated_images = vae.decoder(latent_codes)  # Decode the latent codes into images
# # # # # # # # # # # #         output = model(generated_images)  # Classify the images with the trained ResNet
# # # # # # # # # # # #         _, predicted = torch.max(output, 1)  # Get the predicted class labels
# # # # # # # # # # # #     return generated_images, predicted

# # # # # # # # # # # # # Function to save images and log results
# # # # # # # # # # # # def save_generated_images(epoch, latent_codes):
# # # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # # #         vae.eval()
# # # # # # # # # # # #         generated_images, predicted_classes = classify_latent_codes(latent_codes)

# # # # # # # # # # # #         # Save the generated images
# # # # # # # # # # # #         save_image(generated_images, os.path.join(output_dir, 'generated_images', f'generated_epoch_{epoch+1}.png'))

# # # # # # # # # # # #         # Optionally, save predicted class labels
# # # # # # # # # # # #         np.savetxt(os.path.join(output_dir, 'generated_images', f'predicted_classes_epoch_{epoch+1}.txt'), predicted_classes.cpu().numpy())

# # # # # # # # # # # # # Example: Generate and classify latent codes
# # # # # # # # # # # # latent_codes = torch.randn(64, latent_dim).cuda()  # Sample random latent codes from the latent space

# # # # # # # # # # # # # Optionally, fine-tune latent codes or apply any other optimization if needed
# # # # # # # # # # # # # This can involve adjusting the latent codes to achieve certain goals (e.g., diversity in generated images)

# # # # # # # # # # # # # Run through the stages for a few epochs or steps
# # # # # # # # # # # # for epoch in range(epochs):
# # # # # # # # # # # #     print(f"Epoch {epoch+1}/{epochs}")

# # # # # # # # # # # #     # Generate latent codes (here we assume you already have the latent codes)
# # # # # # # # # # # #     latent_codes = torch.randn(64, latent_dim).cuda()  # Generate a batch of latent codes
    
# # # # # # # # # # # #     # Generate and classify images from the latent codes
# # # # # # # # # # # #     save_generated_images(epoch, latent_codes)
    
# # # # # # # # # # # #     # Optionally, you could do more, such as optimizing or adjusting latent codes, depending on your specific goals

# # # # # # # # # # # # print("Stage 2 completed! Generated images and classification results saved.")



# # # # # # # # # # # import torch
# # # # # # # # # # # from torch.utils.data import DataLoader
# # # # # # # # # # # from torchvision import transforms
# # # # # # # # # # # from resnet import resnet34  # Import your ResNet model
# # # # # # # # # # # from vae import VAE  # Import your VAE model
# # # # # # # # # # # from torchvision.utils import save_image
# # # # # # # # # # # import os
# # # # # # # # # # # import numpy as np

# # # # # # # # # # # # Hyperparameters
# # # # # # # # # # # latent_dim = 256  # Latent space dimension
# # # # # # # # # # # batch_size = 64  # Batch size for training
# # # # # # # # # # # epochs = 20  # Number of epochs for training
# # # # # # # # # # # image_size = 32  # Image size (CIFAR-10 images are 32x32)

# # # # # # # # # # # # Output directory for saving generated images and other results
# # # # # # # # # # # output_dir = './output'
# # # # # # # # # # # os.makedirs(os.path.join(output_dir, 'generated_images'), exist_ok=True)
# # # # # # # # # # # os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

# # # # # # # # # # # # Load the trained ResNet model (without 'pretrained' argument)
# # # # # # # # # # # model = resnet34(num_classes=10).cuda()  # Initialize the ResNet model without pretrained argument
# # # # # # # # # # # model.load_state_dict(torch.load('path_to_your_trained_model.pth'))  # Load the trained ResNet model checkpoint
# # # # # # # # # # # model.eval()  # Set the model to evaluation mode

# # # # # # # # # # # # Load the trained VAE model (same model as used in Stage 1)
# # # # # # # # # # # vae = VAE(latent_dim=latent_dim).cuda()
# # # # # # # # # # # vae.load_state_dict(torch.load('path_to_your_trained_vae_model.pth'))  # Load the trained VAE checkpoint

# # # # # # # # # # # # Loss function for Stage 2 (if necessary for optimizing latent codes)
# # # # # # # # # # # def loss_function(recon_x, x, mu, log_var):
# # # # # # # # # # #     MSE = torch.nn.functional.mse_loss(recon_x.view(-1, 3*image_size*image_size), x.view(-1, 3*image_size*image_size), reduction='sum')
# # # # # # # # # # #     # KL Divergence
# # # # # # # # # # #     return MSE + -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

# # # # # # # # # # # # Define function to classify augmented images from latent codes
# # # # # # # # # # # def classify_latent_codes(latent_codes):
# # # # # # # # # # #     latent_codes = latent_codes.cuda()
# # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # #         generated_images = vae.decoder(latent_codes)  # Decode the latent codes into images
# # # # # # # # # # #         output = model(generated_images)  # Classify the images with the trained ResNet
# # # # # # # # # # #         _, predicted = torch.max(output, 1)  # Get the predicted class labels
# # # # # # # # # # #     return generated_images, predicted

# # # # # # # # # # # # Function to save images and log results
# # # # # # # # # # # def save_generated_images(epoch, latent_codes):
# # # # # # # # # # #     with torch.no_grad():
# # # # # # # # # # #         vae.eval()
# # # # # # # # # # #         generated_images, predicted_classes = classify_latent_codes(latent_codes)

# # # # # # # # # # #         # Save the generated images
# # # # # # # # # # #         save_image(generated_images, os.path.join(output_dir, 'generated_images', f'generated_epoch_{epoch+1}.png'))

# # # # # # # # # # #         # Optionally, save predicted class labels
# # # # # # # # # # #         np.savetxt(os.path.join(output_dir, 'generated_images', f'predicted_classes_epoch_{epoch+1}.txt'), predicted_classes.cpu().numpy())

# # # # # # # # # # # # Example: Generate and classify latent codes
# # # # # # # # # # # latent_codes = torch.randn(64, latent_dim).cuda()  # Sample random latent codes from the latent space

# # # # # # # # # # # # Optionally, fine-tune latent codes or apply any other optimization if needed
# # # # # # # # # # # # This can involve adjusting the latent codes to achieve certain goals (e.g., diversity in generated images)

# # # # # # # # # # # # Run through the stages for a few epochs or steps
# # # # # # # # # # # for epoch in range(epochs):
# # # # # # # # # # #     print(f"Epoch {epoch+1}/{epochs}")

# # # # # # # # # # #     # Generate latent codes (here we assume you already have the latent codes)
# # # # # # # # # # #     latent_codes = torch.randn(64, latent_dim).cuda()  # Generate a batch of latent codes
    
# # # # # # # # # # #     # Generate and classify images from the latent codes
# # # # # # # # # # #     save_generated_images(epoch, latent_codes)
    
# # # # # # # # # # #     # Optionally, you could do more, such as optimizing or adjusting latent codes, depending on your specific goals

# # # # # # # # # # # print("Stage 2 completed! Generated images and classification results saved.")



# # # # # # # # # # import torch
# # # # # # # # # # from torch.utils.data import DataLoader
# # # # # # # # # # from torchvision import transforms, datasets
# # # # # # # # # # from resnet import resnet34  # Import the resnet34 model
# # # # # # # # # # import os
# # # # # # # # # # from tqdm import tqdm

# # # # # # # # # # # Hyperparameters
# # # # # # # # # # batch_size = 64
# # # # # # # # # # epochs = 20
# # # # # # # # # # learning_rate = 1e-3
# # # # # # # # # # num_classes = 10  # CIFAR-10 has 10 classes

# # # # # # # # # # # Load the trained ResNet model
# # # # # # # # # # model = resnet34(pretrained=False, num_classes=num_classes).cuda()

# # # # # # # # # # # Load the checkpoint
# # # # # # # # # # checkpoint_path = './output/checkpoints/resnet34_epoch_20.pth'  # Path to the last saved checkpoint (update if necessary)
# # # # # # # # # # model.load_state_dict(torch.load(checkpoint_path))
# # # # # # # # # # model.eval()  # Set the model to evaluation mode

# # # # # # # # # # # Loss function
# # # # # # # # # # criterion = torch.nn.CrossEntropyLoss()

# # # # # # # # # # # Dataset and DataLoader (CIFAR-10)
# # # # # # # # # # transform = transforms.Compose([
# # # # # # # # # #     transforms.ToTensor(),
# # # # # # # # # #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
# # # # # # # # # # ])

# # # # # # # # # # test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# # # # # # # # # # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# # # # # # # # # # # Evaluation loop
# # # # # # # # # # correct = 0
# # # # # # # # # # total = 0
# # # # # # # # # # running_loss = 0.0

# # # # # # # # # # with torch.no_grad():  # No need to compute gradients during inference
# # # # # # # # # #     for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
# # # # # # # # # #         data, target = data.cuda(), target.cuda()  # Send data to GPU
        
# # # # # # # # # #         # Forward pass
# # # # # # # # # #         output = model(data)
# # # # # # # # # #         loss = criterion(output, target)
        
# # # # # # # # # #         running_loss += loss.item()
# # # # # # # # # #         _, predicted = torch.max(output, 1)
# # # # # # # # # #         correct += (predicted == target).sum().item()
# # # # # # # # # #         total += target.size(0)

# # # # # # # # # # # Calculate average loss and accuracy
# # # # # # # # # # avg_loss = running_loss / len(test_loader.dataset)
# # # # # # # # # # accuracy = 100 * correct / total
# # # # # # # # # # print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

# # # # # # # # # # # Save the final model if necessary (optional)
# # # # # # # # # # # torch.save(model.state_dict(), './output/checkpoints/final_resnet34.pth')

# # # # # # # # # # print("Stage 2 completed!")


# # # # # # # # # import torch
# # # # # # # # # from torch.utils.data import DataLoader
# # # # # # # # # from torchvision import transforms, datasets
# # # # # # # # # from resnet import resnet34  # Import the resnet34 model
# # # # # # # # # import os
# # # # # # # # # from tqdm import tqdm

# # # # # # # # # # Hyperparameters
# # # # # # # # # batch_size = 64
# # # # # # # # # epochs = 20
# # # # # # # # # learning_rate = 1e-3
# # # # # # # # # num_classes = 10  # CIFAR-10 has 10 classes

# # # # # # # # # # Load the trained ResNet model
# # # # # # # # # model = resnet34(num_classes=num_classes).cuda()

# # # # # # # # # # Load the checkpoint
# # # # # # # # # checkpoint_path = './output/checkpoints/resnet34_epoch_20.pth'  # Path to the last saved checkpoint (update if necessary)
# # # # # # # # # model.load_state_dict(torch.load(checkpoint_path))
# # # # # # # # # model.eval()  # Set the model to evaluation mode

# # # # # # # # # # Loss function
# # # # # # # # # criterion = torch.nn.CrossEntropyLoss()

# # # # # # # # # # Dataset and DataLoader (CIFAR-10)
# # # # # # # # # transform = transforms.Compose([
# # # # # # # # #     transforms.ToTensor(),
# # # # # # # # #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
# # # # # # # # # ])

# # # # # # # # # test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# # # # # # # # # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# # # # # # # # # # Evaluation loop
# # # # # # # # # correct = 0
# # # # # # # # # total = 0
# # # # # # # # # running_loss = 0.0

# # # # # # # # # with torch.no_grad():  # No need to compute gradients during inference
# # # # # # # # #     for batch_idx, (data, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
# # # # # # # # #         data, target = data.cuda(), target.cuda()  # Send data to GPU
        
# # # # # # # # #         # Forward pass
# # # # # # # # #         output = model(data)
# # # # # # # # #         loss = criterion(output, target)
        
# # # # # # # # #         running_loss += loss.item()
# # # # # # # # #         _, predicted = torch.max(output, 1)
# # # # # # # # #         correct += (predicted == target).sum().item()
# # # # # # # # #         total += target.size(0)

# # # # # # # # # # Calculate average loss and accuracy
# # # # # # # # # avg_loss = running_loss / len(test_loader.dataset)
# # # # # # # # # accuracy = 100 * correct / total
# # # # # # # # # print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

# # # # # # # # # # Save the final model if necessary (optional)
# # # # # # # # # # torch.save(model.state_dict(), './output/checkpoints/final_resnet34.pth')

# # # # # # # # # print("Stage 2 completed!")



# # # # # # # # import argparse, os
# # # # # # # # import torch
# # # # # # # # from torch import autocast
# # # # # # # # from omegaconf import OmegaConf
# # # # # # # # from torchvision import datasets
# # # # # # # # from torch.utils.data import DataLoader
# # # # # # # # from contextlib import nullcontext
# # # # # # # # import warnings
# # # # # # # # warnings.filterwarnings("ignore")
# # # # # # # # import sys
# # # # # # # # sys.path.append("./")
# # # # # # # # sys.path.append("../")
# # # # # # # # from my_transforms import *

# # # # # # # # # Custom VAE imports
# # # # # # # # from vae import VAE  # Assuming you have a VAE class defined somewhere

# # # # # # # # # Helper functions
# # # # # # # # def load_vae_model(vae_ckpt, device):
# # # # # # # #     """Loads the custom VAE model from checkpoint."""
# # # # # # # #     model = VAE()  # Assuming VAE class is defined
# # # # # # # #     model.load_state_dict(torch.load(vae_ckpt, map_location=device))
# # # # # # # #     model.to(device)
# # # # # # # #     model.eval()
# # # # # # # #     return model

# # # # # # # # def generate_images_from_latent(vae, latent_codes, batch_size):
# # # # # # # #     """Decodes the latent codes using the VAE and generates images."""
# # # # # # # #     with torch.no_grad():
# # # # # # # #         decoded_images = vae.decode(latent_codes)  # Assuming VAE has a decode function
# # # # # # # #         return decoded_images

# # # # # # # # def save_images(images, save_path, class_id, epoch, i):
# # # # # # # #     """Saves generated images."""
# # # # # # # #     images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)  # Normalizing to [0, 1]
# # # # # # # #     for img in images:
# # # # # # # #         img = img.permute(1, 2, 0).cpu().numpy()  # Convert to HWC
# # # # # # # #         img = (img * 255).astype(np.uint8)
# # # # # # # #         img = Image.fromarray(img)
# # # # # # # #         img.save(os.path.join(save_path, f"{class_id}/{epoch}_{i:04}.png"))

# # # # # # # # # Main function for generating images using custom VAE
# # # # # # # # def main(opt):
# # # # # # # #     # Load configuration and models
# # # # # # # #     config = OmegaConf.load(f"{opt.config}")
# # # # # # # #     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
# # # # # # # #     # Load custom VAE model
# # # # # # # #     vae = load_vae_model(opt.vae_ckpt, device)

# # # # # # # #     # Create output directory
# # # # # # # #     os.makedirs(opt.outdir, exist_ok=True)
# # # # # # # #     sample_path = os.path.join(opt.outdir, "generated_images")
# # # # # # # #     os.makedirs(sample_path, exist_ok=True)

# # # # # # # #     # Load codebooks for latent codes
# # # # # # # #     codebooks = torch.load(opt.codebookdir)
# # # # # # # #     print(f"Loaded codebooks from {opt.codebookdir}")

# # # # # # # #     # Generate images from codebook latent codes
# # # # # # # #     epoch = 0
# # # # # # # #     class_num = len(codebooks)
    
# # # # # # # #     for class_id in range(class_num):
# # # # # # # #         os.makedirs(os.path.join(sample_path, str(class_id)), exist_ok=True)
# # # # # # # #         for i in range(opt.n_samples):  # Number of samples to generate per class
# # # # # # # #             # Get latent codes for the class
# # # # # # # #             latent_codes = codebooks[class_id]  # Assuming codebook contains the latent codes
            
# # # # # # # #             # Generate images using the VAE decoder
# # # # # # # #             images = generate_images_from_latent(vae, latent_codes, opt.n_samples)
            
# # # # # # # #             # Save generated images
# # # # # # # #             save_images(images, sample_path, class_id, epoch, i)
        
# # # # # # # #         print(f"Epoch {epoch} completed for class {class_id}")
# # # # # # # #     epoch += 1

# # # # # # # # if __name__ == "__main__":
# # # # # # # #     parser = argparse.ArgumentParser()
# # # # # # # #     parser.add_argument(
# # # # # # # #         "--outdir",
# # # # # # # #         type=str,
# # # # # # # #         default="outputs/generate_images",
# # # # # # # #         help="Directory to save generated images"
# # # # # # # #     )
# # # # # # # #     parser.add_argument(
# # # # # # # #         "--vae_ckpt",
# # # # # # # #         type=str,
# # # # # # # #         required=True,
# # # # # # # #         help="Path to the checkpoint of the custom VAE model"
# # # # # # # #     )
# # # # # # # #     parser.add_argument(
# # # # # # # #         "--codebookdir",
# # # # # # # #         type=str,
# # # # # # # #         required=True,
# # # # # # # #         help="Path to the codebook directory containing latent codes"
# # # # # # # #     )
# # # # # # # #     parser.add_argument(
# # # # # # # #         "--n_samples",
# # # # # # # #         type=int,
# # # # # # # #         default=5,
# # # # # # # #         help="Number of samples to generate per class"
# # # # # # # #     )
# # # # # # # #     parser.add_argument(
# # # # # # # #         "--config",
# # # # # # # #         type=str,
# # # # # # # #         default="../configs/vae_config.yaml",
# # # # # # # #         help="Path to config for VAE"
# # # # # # # #     )
# # # # # # # #     opt = parser.parse_args()

# # # # # # # #     main(opt)


# # # # # # # import os
# # # # # # # import torch
# # # # # # # import argparse
# # # # # # # import torch.nn.functional as F
# # # # # # # from torchvision.utils import save_image
# # # # # # # from vae import VAE
# # # # # # # from resnet import resnet34

# # # # # # # def main(args):
# # # # # # #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # # # # # #     # Load pretrained VAE
# # # # # # #     vae = VAE(latent_dim=args.latent_dim).to(device)
# # # # # # #    # vae_ckpt = os.path.join('checkpoints', 'vae_epoch_20.pth')
# # # # # # #     vae_ckpt = './output/checkpoints/vae_epoch_20.pth'
# # # # # # #     vae.load_state_dict(torch.load(vae_ckpt))
# # # # # # #     vae.eval()

# # # # # # #     # Load pretrained ResNet
# # # # # # #     classifier = resnet34(num_classes=10).to(device)
# # # # # # #     #resnet_ckpt = os.path.join('checkpoints', 'resnet34_epoch_20.pth')
# # # # # # #     resnet_ckpt = './output/checkpoints/resnet34_epoch_20.pth'
# # # # # # #     classifier.load_state_dict(torch.load(resnet_ckpt))
# # # # # # #     classifier.eval()

# # # # # # #     os.makedirs(args.output_path, exist_ok=True)

# # # # # # #     latent_files = sorted([f for f in os.listdir(args.latent_path) if f.endswith('.pt')])
# # # # # # #     print(f"Found {len(latent_files)} latent files.")

# # # # # # #     total_saved = 0

# # # # # # #     for filename in latent_files:
# # # # # # #         path = os.path.join(args.latent_path, filename)
# # # # # # #         latent = torch.load(path).to(device)

# # # # # # #         with torch.no_grad():
# # # # # # #             img = vae.decode(latent)
# # # # # # #             output = classifier(img)
# # # # # # #             probs = F.softmax(output, dim=1)
# # # # # # #             confidence, _ = torch.max(probs, dim=1)

# # # # # # #             if confidence.item() >= args.threshold:
# # # # # # #                 save_path = os.path.join(args.output_path, filename.replace('.pt', '.png'))
# # # # # # #                 save_image(img, save_path)
# # # # # # #                 total_saved += 1

# # # # # # #     print(f"Saved {total_saved} high-confidence images to {args.output_path}")

# # # # # # # if __name__ == '__main__':
# # # # # # #     parser = argparse.ArgumentParser()
# # # # # # #     parser.add_argument('--latent_path', type=str, default='LCA/result1', help='Path to latent code .pt files')
# # # # # # #     parser.add_argument('--output_path', type=str, default='LCA/result2', help='Directory to save generated images')
# # # # # # #     parser.add_argument('--threshold', type=float, default=0.8, help='Confidence threshold')
# # # # # # #     parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension size')

# # # # # # #     args = parser.parse_args()
# # # # # # #     main(args)


# # # # # # import os
# # # # # # import torch
# # # # # # import argparse
# # # # # # import torch.nn.functional as F
# # # # # # from torchvision.utils import save_image
# # # # # # from vae import VAE
# # # # # # from resnet import resnet34, load_resnet_checkpoint  # Import the checkpoint loading function

# # # # # # def main(args):
# # # # # #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # # # # #     # Load pretrained VAE
# # # # # #     vae = VAE(latent_dim=args.latent_dim).to(device)
# # # # # #     vae_ckpt = './output/checkpoints/vae_epoch_20.pth'  # Specify VAE checkpoint path
# # # # # #     vae.load_state_dict(torch.load(vae_ckpt))
# # # # # #     vae.eval()

# # # # # #     # Load pretrained ResNet
# # # # # #     classifier = resnet34(num_classes=10).to(device)
# # # # # #     resnet_ckpt = './output/checkpoints/resnet34_epoch_20.pth'  # Specify ResNet checkpoint path
# # # # # #     load_resnet_checkpoint(classifier, resnet_ckpt)  # Use the custom function to load the ResNet checkpoint
# # # # # #     classifier.eval()

# # # # # #     os.makedirs(args.output_path, exist_ok=True)

# # # # # #     latent_files = sorted([f for f in os.listdir(args.latent_path) if f.endswith('.pt')])
# # # # # #     print(f"Found {len(latent_files)} latent files.")

# # # # # #     total_saved = 0

# # # # # #     for filename in latent_files:
# # # # # #         path = os.path.join(args.latent_path, filename)
# # # # # #         latent = torch.load(path).to(device)

# # # # # #         with torch.no_grad():
# # # # # #             img = vae.decode(latent)
# # # # # #             output = classifier(img)
# # # # # #             probs = F.softmax(output, dim=1)
# # # # # #             confidence, _ = torch.max(probs, dim=1)

# # # # # #             if confidence.item() >= args.threshold:
# # # # # #                 save_path = os.path.join(args.output_path, filename.replace('.pt', '.png'))
# # # # # #                 save_image(img, save_path)
# # # # # #                 total_saved += 1

# # # # # #     print(f"Saved {total_saved} high-confidence images to {args.output_path}")

# # # # # # if __name__ == '__main__':
# # # # # #     parser = argparse.ArgumentParser()
# # # # # #     parser.add_argument('--latent_path', type=str, default='./LCA/scripts/result1', help='Path to latent code .pt files')
# # # # # #     parser.add_argument('--output_path', type=str, default='./LCA/scripts/result2', help='Directory to save generated images')
# # # # # #     parser.add_argument('--threshold', type=float, default=0.8, help='Confidence threshold')
# # # # # #     parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension size')

# # # # # #     args = parser.parse_args()
# # # # # #     main(args)


# # # # # import os
# # # # # import torch
# # # # # import argparse
# # # # # import torch.nn.functional as F
# # # # # from torchvision.utils import save_image
# # # # # from vae import VAE
# # # # # from resnet import resnet34, load_resnet_checkpoint  # Import the checkpoint loading function

# # # # # def main(args):
# # # # #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # # # #     # Load pretrained VAE
# # # # #     vae = VAE(latent_dim=args.latent_dim).to(device)
# # # # #     vae_ckpt = './output/checkpoints/vae_epoch_20.pth'  # Specify VAE checkpoint path
# # # # #     vae.load_state_dict(torch.load(vae_ckpt))
# # # # #     vae.eval()

# # # # #     # Load pretrained ResNet
# # # # #     classifier = resnet34(num_classes=10).to(device)
# # # # #     resnet_ckpt = './output/checkpoints/resnet34_epoch_20.pth'  # Specify ResNet checkpoint path
# # # # #     load_resnet_checkpoint(classifier, resnet_ckpt)  # Use the custom function to load the ResNet checkpoint
# # # # #     classifier.eval()

# # # # #     os.makedirs(args.output_path, exist_ok=True)

# # # # #     latent_files = sorted([f for f in os.listdir(args.latent_path) if f.endswith('.pt')])
# # # # #     print(f"Found {len(latent_files)} latent files.")

# # # # #     total_saved = 0

# # # # #     for filename in latent_files:
# # # # #         path = os.path.join(args.latent_path, filename)
# # # # #         latent = torch.load(path).to(device)

# # # # #         with torch.no_grad():
# # # # #             img = vae.decoder(latent)
# # # # #             output = classifier(img)
# # # # #             probs = F.softmax(output, dim=1)
# # # # #             confidence, _ = torch.max(probs, dim=1)

# # # # #             if confidence.item() >= args.threshold:
# # # # #                 save_path = os.path.join(args.output_path, filename.replace('.pt', '.png'))
# # # # #                 save_image(img, save_path)
# # # # #                 total_saved += 1

# # # # #     print(f"Saved {total_saved} high-confidence images to {args.output_path}")

# # # # # if __name__ == '__main__':
# # # # #     parser = argparse.ArgumentParser()
# # # # #     parser.add_argument('--latent_path', type=str, default='/home/user/project/Try/LCA/scripts/result1', help='Path to latent code .pt files')
# # # # #     parser.add_argument('--output_path', type=str, default='/home/user/project/Try/LCA/scripts/result2', help='Directory to save generated images')
# # # # #     parser.add_argument('--threshold', type=float, default=0.8, help='Confidence threshold')
# # # # #     parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension size')

# # # # #     args = parser.parse_args()
# # # # #     main(args)


# # # # import os
# # # # import torch
# # # # import argparse
# # # # import torch.nn.functional as F
# # # # from torchvision.utils import save_image
# # # # from vae import VAE
# # # # from resnet import resnet34, load_resnet_checkpoint  # Import the checkpoint loading function

# # # # def main(args):
# # # #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # # #     # Load pretrained VAE
# # # #     vae = VAE(latent_dim=args.latent_dim).to(device)
# # # #     vae_ckpt = './output/checkpoints/vae_epoch_20.pth'  # Specify VAE checkpoint path

# # # #     # Load only the encoder's weights from the checkpoint
# # # #     vae_dict = vae.state_dict()  # Get the current state dict of the VAE model
# # # #     checkpoint = torch.load(vae_ckpt)  # Load the checkpoint

# # # #     # Update only the encoder's weights in the VAE
# # # #     vae_dict.update({k: v for k, v in checkpoint.items() if 'encoder' in k})

# # # #     # Load the updated state dict (this will load only the encoder's weights)
# # # #     vae.load_state_dict(vae_dict)
# # # #     vae.eval()

# # # #     # Load pretrained ResNet
# # # #     classifier = resnet34(num_classes=10).to(device)
# # # #     resnet_ckpt = './output/checkpoints/resnet34_epoch_20.pth'  # Specify ResNet checkpoint path
# # # #     load_resnet_checkpoint(classifier, resnet_ckpt)  # Use the custom function to load the ResNet checkpoint
# # # #     classifier.eval()

# # # #     os.makedirs(args.output_path, exist_ok=True)

# # # #     latent_files = sorted([f for f in os.listdir(args.latent_path) if f.endswith('.pt')])
# # # #     print(f"Found {len(latent_files)} latent files.")

# # # #     total_saved = 0

# # # #     for filename in latent_files:
# # # #         path = os.path.join(args.latent_path, filename)
# # # #         latent = torch.load(path).to(device)

# # # #         with torch.no_grad():
# # # #             img = vae.decoder(latent)  # Use the decoder to generate the image
# # # #             output = classifier(img)
# # # #             probs = F.softmax(output, dim=1)
# # # #             confidence, _ = torch.max(probs, dim=1)

# # # #             if confidence.item() >= args.threshold:
# # # #                 save_path = os.path.join(args.output_path, filename.replace('.pt', '.png'))
# # # #                 save_image(img, save_path)
# # # #                 total_saved += 1

# # # #     print(f"Saved {total_saved} high-confidence images to {args.output_path}")

# # # # if __name__ == '__main__':
# # # #     parser = argparse.ArgumentParser()
# # # #     parser.add_argument('--latent_path', type=str, default='/home/user/project/Try/LCA/scripts/result1', help='Path to latent code .pt files')
# # # #     parser.add_argument('--output_path', type=str, default='/home/user/project/Try/LCA/scripts/result2', help='Directory to save generated images')
# # # #     parser.add_argument('--threshold', type=float, default=0.8, help='Confidence threshold')
# # # #     parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension size')

# # # #     args = parser.parse_args()
# # # #     main(args)


# # # import os
# # # import torch
# # # from torchvision.utils import save_image
# # # from vae import VAE

# # # def load_latent_codes(latent_path):
# # #     latent_codes = []
# # #     for file_name in os.listdir(latent_path):
# # #         if file_name.endswith('.pt'):
# # #             latent_codes.append(torch.load(os.path.join(latent_path, file_name)))
# # #     return latent_codes

# # # def generate_images_from_latents(vae, latent_codes, output_path, threshold=0.8):
# # #     vae.eval()  # Set the model to evaluation mode
# # #     with torch.no_grad():
# # #         for i, latent in enumerate(latent_codes):
# # #             # Ensure the latent tensor is of the correct shape [batch_size, latent_dim]
# # #             latent = latent.unsqueeze(0)  # Add batch dimension
            
# # #             # Pass latent vector through decoder
# # #             img = vae.decoder(latent)
            
# # #             # Optional: apply a threshold to the output images (e.g., for binary images)
# # #             img = torch.clamp(img, 0, 1)
            
# # #             # Save image
# # #             save_image(img, os.path.join(output_path, f'generated_{i}.png'))

# # # def main():
# # #     # Arguments (paths, threshold, latent dimension)
# # #     latent_path = '/home/user/project/Try/LCA/scripts/result1'  # Path to latent code files
# # #     output_path = '/home/user/project/Try/LCA/scripts/result2'  # Output directory for generated images
# # #     threshold = 0.8  # Example threshold (if needed for binary image generation)
# # #     latent_dim = 256  # Latent dimension size (you've updated this to 256)

# # #     # Initialize VAE model
# # #     vae = VAE(latent_dim=latent_dim)  # Ensure VAE model uses the correct latent_dim

# # #     # Load latent codes
# # #     latent_codes = load_latent_codes(latent_path)

# # #     # Create output directory if not exists
# # #     os.makedirs(output_path, exist_ok=True)

# # #     # Generate and save images from latent codes
# # #     generate_images_from_latents(vae, latent_codes, output_path, threshold)

# # # if __name__ == "__main__":
# # #     main()



import os
import torch
from torchvision.utils import save_image
from vae import VAE

def load_latent_codes(latent_path):
    latent_codes = []
    for file_name in os.listdir(latent_path):
        if file_name.endswith('.pt'):
            latent_codes.append(torch.load(os.path.join(latent_path, file_name)))
    return latent_codes

def generate_images_from_latents(vae, latent_codes, output_path, threshold=0.8, device='cpu'):
    vae.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for i, latent in enumerate(latent_codes):
            latent = latent.to(device)           # <-- Move latent to GPU
            latent = latent.unsqueeze(0)         # Add batch dimension
            
            img = vae.decoder(latent)            # Decode
            img = torch.clamp(img, 0, 1)         # Optional post-processing
            
            save_image(img, os.path.join(output_path, f'generated_{i}.png'))  # Save image

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Paths and configs
    latent_path = '/home/user/project/Try/LCA/scripts/result1'
    output_path = '/home/user/project/Try/LCA/scripts/result2'
    threshold = 0.8
    latent_dim = 256

    # Load model and move to GPU
    vae = VAE(latent_dim=latent_dim).to(device)

    # Load latents
    latent_codes = load_latent_codes(latent_path)

    # Create output folder
    os.makedirs(output_path, exist_ok=True)

    # Decode images
    generate_images_from_latents(vae, latent_codes, output_path, threshold, device)

if __name__ == "__main__":
    main()

# # import os
# # import torch
# # from torchvision.utils import save_image
# # from torchvision import transforms
# # from tqdm import tqdm
# # from vae import VAE
# # from resnet import resnet34  # Assuming your ResNet is defined in resnet.py

# # # Settings
# # latent_path = '/home/user/project/Try/LCA/scripts/result1/'
# # output_dir = '/home/user/project/Try/LCA/scripts/result2/'
# # vae_ckpt = '/home/user/project/Try/LCA/scripts/output/checkpoints/vae_epoch_20.pth'
# # resnet_ckpt = '/home/user/project/Try/LCA/scripts/output/checkpoints/resnet34_epoch_20.pth'
# # latent_dim = 128
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # threshold = 0.9  # Confidence threshold

# # # Load VAE
# # vae = VAE(latent_dim=latent_dim).to(device)
# # vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
# # vae.eval()

# # # Load ResNet34
# # model = resnet34().to(device)
# # model.load_state_dict(torch.load(resnet_ckpt, map_location=device))
# # model.eval()

# # # Image normalization transform (match what ResNet expects)
# # transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# # # Create output directory
# # os.makedirs(output_dir, exist_ok=True)

# # # Process each latent file
# # latent_files = sorted(f for f in os.listdir(latent_path) if f.endswith('.pt'))

# # for filename in tqdm(latent_files):
# #     path = os.path.join(latent_path, filename)
# #     latent_code = torch.load(path).to(device)
# #     if len(latent_code.shape) == 1:
# #         latent_code = latent_code.unsqueeze(0)

# #     with torch.no_grad():
# #         recon = vae.decode(latent_code)
# #         input_img = transform(recon.squeeze(0))  # normalize
# #         input_img = input_img.unsqueeze(0)  # shape: [1, 3, 32, 32]
# #         pred = model(input_img)
# #         probs = torch.softmax(pred, dim=1)
# #         confidence, pred_label = torch.max(probs, dim=1)

# #     if confidence.item() > threshold:
# #         save_path = os.path.join(output_dir, filename.replace('.pt', '.png'))
# #         save_image(recon, save_path)

# import os
# import torch
# from torchvision.utils import save_image
# from torchvision import transforms
# from tqdm import tqdm
# from vae import VAE
# from resnet import resnet34  # Assuming your ResNet is defined in resnet.py

# # Settings
# latent_path = '/home/user/project/Try/LCA/scripts/result1/'
# output_dir = '/home/user/project/Try/LCA/scripts/result2/'
# vae_ckpt = '/home/user/project/Try/LCA/scripts/output/checkpoints/vae_epoch_20.pth'
# resnet_ckpt = '/home/user/project/Try/LCA/scripts/output/checkpoints/resnet34_epoch_20.pth'
# latent_dim = 256
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# threshold = 0.9  # Confidence threshold

# # Load VAE
# vae = VAE(latent_dim=latent_dim).to(device)
# vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
# vae.eval()

# # Load ResNet34
# model = resnet34().to(device)
# model.load_state_dict(torch.load(resnet_ckpt, map_location=device),strict=False)
# model.eval()

# # Image normalization transform (match what ResNet expects)
# transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# # Create output directory
# os.makedirs(output_dir, exist_ok=True)

# # Process each latent file
# latent_files = sorted(f for f in os.listdir(latent_path) if f.endswith('.pt'))

# for filename in tqdm(latent_files):
#     path = os.path.join(latent_path, filename)
#     latent_code = torch.load(path).to(device)
#     if len(latent_code.shape) == 1:
#         latent_code = latent_code.unsqueeze(0)

#     with torch.no_grad():
#         recon, _, _ = vae(latent_code)  # Get reconstructed image from VAE
#         recon = torch.clamp(recon, 0, 1)  # Ensure the image is in [0, 1] range
#         input_img = recon.squeeze(0)  # Remove batch dimension
        
#         # Normalize for ResNet input
#         input_img = transform(input_img)  # Normalize image based on ResNet expectations
#         input_img = input_img.unsqueeze(0)  # Add batch dimension

#         # Make prediction with ResNet34
#         pred = model(input_img)
#         probs = torch.softmax(pred, dim=1)
#         confidence, pred_label = torch.max(probs, dim=1)

#     # If the confidence is above the threshold, save the image
#     if confidence.item() > threshold:
#         save_path = os.path.join(output_dir, filename.replace('.pt', '.png'))
#         save_image(recon, save_path)


import os
import torch
from torchvision.utils import save_image
from vae import VAE
import torchvision.transforms.functional as TF

# Paths
latent_dir = "/home/user/project/Try/LCA/scripts/result1/"
output_dir = "/home/user/project/Try/LCA/scripts/result2/"
checkpoint_path = "/home/user/project/Try/LCA/scripts/output/checkpoints/vae_epoch_20.pth"

os.makedirs(output_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VAE
vae = VAE()
checkpoint = torch.load(checkpoint_path, map_location=device)
vae.load_state_dict(checkpoint)
vae.to(device)
vae.eval()

# Process .pt files
latent_files = sorted(f for f in os.listdir(latent_dir) if f.endswith(".pt"))

for file_name in latent_files:
    latent_path = os.path.join(latent_dir, file_name)
    latent_code = torch.load(latent_path).to(device)

    with torch.no_grad():
        decoded_img = vae.decoder(latent_code)

        # Clip values to [0, 1] just to be safe
        decoded_img = torch.clamp(decoded_img, 0.0, 1.0)

        # Optional: enhance contrast (simple stretch)
        min_val = decoded_img.min()
        max_val = decoded_img.max()
        if (max_val - min_val) > 1e-5:
            decoded_img = (decoded_img - min_val) / (max_val - min_val)

    # Save image
    save_path = os.path.join(output_dir, file_name.replace(".pt", ".png"))
    save_image(decoded_img, save_path)

    print(f"Saved image: {save_path}")
