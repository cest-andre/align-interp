import argparse
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import models, transforms
# from thingsvision import get_extractor

from sae import ModelWrapper
from utils import sort_acts, save_top_imgs


def get_activations(extractor, x, module_name, neuron_coord=None, channel_id=None, use_center=False):
    if len(x.shape) == 3:
        x = torch.unsqueeze(x, 0)

    x = torch.unsqueeze(x, 0)

    activations = extractor.extract_features(
        batches=x,
        module_name=module_name,
        flatten_acts=False
    )

    if use_center:
        neuron_coord = activations.shape[-1] // 2

    if neuron_coord is not None:
        activations = activations[:, :, neuron_coord, neuron_coord]

    if channel_id is not None:
        activations = activations[:, channel_id]

    return activations


def get_imnet_acts(model, imnet_dir, layer_name, device, return_imgs=False, sae_weights=None, batch_size=2048):
    IMAGE_SIZE = 224
    # specify ImageNet mean and standard deviation
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_transform = transforms.Normalize(mean=MEAN, std=STD)
    load_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
    ])

    imagenet_data = torchvision.datasets.ImageFolder(imnet_dir, transform=load_transform)
    trainloader = DataLoader(imagenet_data, num_workers=8, batch_size=batch_size, shuffle=False, drop_last=False)

    all_acts = []
    all_imgs = []
    for i, data in enumerate(trainloader, 0):
        inputs, _ = data

        if return_imgs:
            for input in inputs:
                all_imgs.append(input)

        inputs = norm_transform(inputs.to(device))
        acts = model(inputs, relu=True, center=True)

        # acts = get_activations(extractor, inputs, layer_name, None, None, use_center=True)

        if sae_weights is not None:
            acts = acts.to(torch.float64) @ sae_weights

        all_acts += acts.detach().cpu().numpy().tolist()
        
    if return_imgs:
        all_imgs = torch.stack(all_imgs)

    return all_acts, all_imgs


def save_top_act_imgs(model, imnet_dir, layer_name, save_act_dir, save_img_dir, device, batch_size, sae_weights=None):
    all_acts, all_imgs = get_imnet_acts(model, imnet_dir, layer_name, device, return_imgs=True, batch_size=batch_size)

    if save_act_dir is not None:
        np.save(os.path.join(save_act_dir, f'{layer_name}.npy'), np.array(all_acts))
    
    all_acts = torch.tensor(all_acts, dtype=torch.float64)
    dataloader = DataLoader(TensorDataset(all_acts), batch_size=batch_size, shuffle=False, drop_last=False)

    save_count = 256
    all_sorted_idx = sort_acts(dataloader, device, save_count, sae_weights=sae_weights)
    for i in range(len(all_sorted_idx)):
        sorted_idx = all_sorted_idx[i]
        save_top_imgs(all_imgs[sorted_idx[:9]], save_img_dir, i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imnet_dir', type=str)
    parser.add_argument('--save_img_dir', type=str, default='')
    parser.add_argument('--save_act_dir', type=str, default='')
    parser.add_argument('--layer_name', type=str)
    parser.add_argument('--sae_name', type=str, default=None)
    parser.add_argument('--sae_weights_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # model_name = 'resnet50'
    # param_name = {'weights': 'IMAGENET1K_V2'}

    # model_name = 'resnet18'
    # param_name = {'weights': 'IMAGENET1K_V1'}

    # extractor = get_extractor(
    #     model_name=model_name,
    #     source='torchvision',
    #     device=device,
    #     pretrained=True,
    #     model_parameters=param_name
    # )

    model = models.resnet50(weights='IMAGENET1K_V2').to(device)
    model = ModelWrapper(model, None, device)
    model.eval()

    sae_weights = None
    save_img_dir = None
    save_act_dir = args.save_act_dir
    if args.sae_name is not None:
        sae_weights = torch.load(os.path.join(args.sae_weights_dir, f'{args.sae_name}.pth'))['W_dec'].to(device).T

        #   TODO: init random vector with same mean and var as example sae_weights.  replace weights with this to get imnet valid
        #         acts to random dirs in activation space.

        save_act_dir = os.path.join(args.save_act_dir, args.layer_name)
        save_img_dir = os.path.join(args.save_img_dir, args.layer_name, args.sae_name)
    else:
        save_img_dir = os.path.join(args.save_img_dir, args.layer_name)

    all_acts, _ = get_imnet_acts(model, args.imnet_dir, args.layer_name, device, sae_weights=sae_weights, batch_size=args.batch_size)
    Path(save_act_dir).mkdir(parents=True, exist_ok=True)
    #   TODO:  do not save sae_name if it doesn't exist -_-
    np.save(os.path.join(save_act_dir, f'{args.sae_name}_valid.npy'), np.array(all_acts))

    # Path(save_img_dir).mkdir(parents=True, exist_ok=True)
    # save_top_act_imgs(
    #     model,
    #     args.imnet_dir,
    #     args.layer_name,
    #     args.save_act_dir,
    #     save_img_dir,
    #     device,
    #     args.batch_size,
    #     sae_weights=sae_weights
    # )