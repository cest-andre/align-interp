from pathlib import Path
import os
import argparse
from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from lucent.optvis import render, objectives, transform
import lucent.optvis.param as param

from sae import ModelWrapper


parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str)
parser.add_argument('--sae_weights_dir', type=str)
parser.add_argument('--sae_name', type=str)
parser.add_argument('--layer_dim', type=int)
parser.add_argument('--unit', type=int)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--jitter', type=int, default=16)
args = parser.parse_args()

torch.manual_seed(0)
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
savedir = os.path.join(args.savedir, args.sae_name)
Path(savedir).mkdir(parents=True, exist_ok=True)

# model = models.resnet18(True)
model = models.resnet50(weights='IMAGENET1K_V2')
act_dirs = torch.load(os.path.join(args.sae_weights_dir, f'{args.sae_name}.pth'))['W_dec'].to(device)

model = ModelWrapper(model, act_dirs.shape[0] // args.layer_dim, device, use_sae=True, input_dims=args.layer_dim)
states = model.state_dict()
states['map.weight'] = act_dirs
model.load_state_dict(states)
model = nn.Sequential(model)
model.to(device).eval()

augs = None
if args.jitter < 4:
    augs = [
        transform.random_scale([1, 0.975, 1.025, 0.95, 1.05]),
        transform.random_rotate([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
    ]
else:
    augs = [
        transform.pad(args.jitter),
        transform.jitter(args.jitter),
        transform.random_scale([1, 0.975, 1.025, 0.95, 1.05]),
        transform.random_rotate([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
        transform.jitter(int(args.jitter/2)),
        transforms.CenterCrop(224),
    ]

img = render.render_vis(
    model,
    objectives.neuron('0', args.unit),
    param_f=lambda: param.images.image(224, decorrelate=True),
    transforms=augs,
    thresholds=(2560,),
    show_image=False
)
img = Image.fromarray((img[0][0]*255).astype(np.uint8))
img.save(os.path.join(savedir, f"unit{args.unit}_distill_center.png"))