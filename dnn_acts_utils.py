import argparse
import os
import numpy as np
import torch
import torchvision
from torchvision import models, transforms
from thingsvision import get_extractor

from sae import ModelWrapper


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imnet_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--layer_name', type=str)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
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

    imagenet_data = torchvision.datasets.ImageFolder(os.path.join(args.imnet_dir, 'training'), transform=load_transform)
    trainloader = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, drop_last=False)

    extractor = get_extractor(
        model_name='resnet50',
        source='torchvision',
        device=device,
        pretrained=True,
        model_parameters={'weights': 'IMAGENET1K_V2'}
    )

    all_acts = []
    for i, data in enumerate(trainloader, 0):
        inputs, _ = data
        inputs = norm_transform(inputs.to(device))
        
        acts = get_activations(extractor, inputs, args.layer_name, None, None, use_center=True)
        all_acts += acts.tolist()

    np.save(os.path.join(args.save_dir, f'{args.layer_name}.npy'), np.array(all_acts))