import argparse
from pathlib import Path
import os
import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'representation-alignment'))
from src.alignment.linear import Linear

from sae import LN
from utils import sort_acts, save_top_imgs, get_coco_imgs, pairwise_corr, RSA, load_vinken_imgs
from monkey_utils import get_mm_imgs


def get_cnn_acts(extractor, x, module_name, neuron_coord=None, channel_id=None, use_center=False):
    if len(x.shape) == 3:
        x = torch.unsqueeze(x, 0)

    x = torch.unsqueeze(x, 0)

    acts = extractor.extract_features(
        batches=x,
        module_name=module_name,
        flatten_acts=False
    )

    if use_center:
        neuron_coord = acts.shape[-1] // 2

    if neuron_coord is not None:
        acts = acts[:, :, neuron_coord, neuron_coord]

    if channel_id is not None:
        acts = acts[:, channel_id]

    return acts


def get_vit_acts(extractor, x, module_name, use_cls_token=False):
    if len(x.shape) == 3:
        x = torch.unsqueeze(x, 0)

    x = torch.unsqueeze(x, 0)

    acts = extractor.extract_features(
        batches=x,
        module_name=module_name,
        flatten_acts=False
    )

    if use_cls_token:
        acts = acts[:, 0]
    else:  
        # acts = acts[:, (acts.shape[1]-1) // 2, :]  #  use center image token patch
        acts = np.mean(acts, axis=1)

    return acts


def get_img_acts(model, img_dir, layer_name, device, return_imgs=False, sae_weights=None, batch_size=2048):
    IMAGE_SIZE = 224
    # specify ImageNet mean and standard deviation (use for all image datasets such as coco?)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_transform = transforms.Normalize(mean=MEAN, std=STD)
    load_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
    ])

    img_ds = None
    if 'ILSVRC2012' in img_dir:  #  is imagenet
        img_ds = torchvision.datasets.ImageFolder(img_dir, transform=load_transform)
    elif 'NSD_Preprocessed' in img_dir:  #  is coco
        imgs = get_coco_imgs(img_dir, None, transform=load_transform)
        img_ds = TensorDataset(torch.stack(imgs))
    elif 'ManyMonkeys' in img_dir:  #  is ManyMonkeys
        imgs = get_mm_imgs(img_dir, transform=load_transform)
        img_ds = TensorDataset(torch.stack(imgs))
    elif 'vinken' in img_dir:  #  is Vinken Face Cells
        imgs = load_vinken_imgs(img_dir)
        img_ds = TensorDataset(imgs)

    dataloader = DataLoader(img_ds, num_workers=1, batch_size=batch_size, shuffle=False, drop_last=False)
    all_acts = []
    all_imgs = []
    for i, data in enumerate(dataloader, 0):
        # inputs, _ = data
        inputs = data[0]

        if return_imgs:
            for input in inputs:
                all_imgs.append(input.cpu())

        inputs = norm_transform(inputs.to(device))

        if 'vit' in extractor.model_name or 'clip' in extractor.model_name:
            acts = get_vit_acts(extractor, inputs, layer_name, use_cls_token=extractor.model_name != 'clip')
            acts = torch.tensor(acts, device=device)
        else:
            acts = get_cnn_acts(extractor, inputs, layer_name, use_center=True)
            acts = torch.clamp(torch.tensor(acts, device=device), min=0, max=None)

        # #   Obtain activations for all patches rather than center-only.
        # #   TODO:  maybe grab just center 3x3 or something?  would scale down from 49, still a 9x bump in data.
        # #          could also grab center and 4 corners if 9x is also too much.
        # center_coord = acts.shape[-1] // 2
        # acts = acts[:, :, center_coord-1:center_coord+2, center_coord-1:center_coord+2]
        # acts = torch.flatten(torch.permute(acts, (0, 2, 3, 1)), start_dim=0, end_dim=-2)

        if sae_weights is not None:
            acts, _, _ = LN(acts)
            acts = (acts @ sae_weights['W_enc']) + sae_weights['b_enc']
            if 'bn' in sae_weights.keys():
                acts = sae_weights['bn'](acts)

            #   topk activation SAE
            topk_res = torch.topk(acts, k=77, dim=-1)
            values = torch.nn.ReLU()(topk_res.values)
            acts = torch.zeros_like(acts, device=acts.device)
            acts.scatter_(-1, topk_res.indices, values)

            acts = torch.nn.ReLU()(acts)

        all_acts += acts.detach().cpu().numpy().tolist()
        
    if return_imgs:
        all_imgs = torch.stack(all_imgs)

    return all_acts, all_imgs


def save_top_act_imgs(extractor, imnet_dir, layer_name, save_act_dir, save_img_dir, device, batch_size, sae_weights=None, sae_name=None):
    all_acts, all_imgs = get_img_acts(extractor, imnet_dir, layer_name, device, return_imgs=True, sae_weights=sae_weights, batch_size=batch_size)

    if save_act_dir is not None:
        Path(save_act_dir).mkdir(parents=True, exist_ok=True)
        np.save(os.path.join(save_act_dir, f'{layer_name if sae_name is None else sae_name}.npy'), np.array(all_acts))

    save_count = 64
    # save_idx = [634, 1463,  200, 1284,  947,  753,  337,  201,  367,  500, 1313,  762,  869,  638,  855, 874]
    all_sorted_idx = sort_acts(all_acts, save_count, save_idx=None)
    for i in range(len(all_sorted_idx)):
        sorted_idx = all_sorted_idx[i]
        save_top_imgs(all_imgs[sorted_idx[:9]], save_img_dir, i)


def dnn_align(source_acts, target_acts):
    metric = Linear()
    scores = metric.fit_kfold_ridge(
        x=source_acts.cpu().to(torch.float),
        y=target_acts.cpu().to(torch.float),
    )
    print(f"Ridge score: {np.array(scores).mean()}")
    results = pairwise_corr(source_acts, target_acts)
    score = torch.mean(torch.max(results, 0)[0]).cpu().numpy()
    print(f"Pairwise score: {score}")
    score = RSA(source_acts, target_acts)[0]
    print(f'RSA: {score}')
    exit()

    # results = pairwise_corr(source_acts, target_acts)
    # print(f'Pearson pairwise: {torch.mean(torch.max(results, 1)[0])}')
    # print(f'Pearson pairwise: {torch.mean(torch.max(results, 0)[0])}')

    # source_val = source_acts[-10000:].cpu().to(torch.float)
    # target_val = target_acts[-10000:].cpu().to(torch.float)
    # source_acts, target_acts = source_acts[:40000], target_acts[:40000] 
    for p in range(25, 125, 25):
        subset_idx = int(source_acts.shape[0] * (p / 100))
        metric = Linear()
        scores = metric.fit_kfold_ridge(
            x=source_acts[:subset_idx].cpu().to(torch.float),
            y=target_acts[:subset_idx].cpu().to(torch.float),
            #val_set={'x': source_val, 'y': target_val}
        )
        print(f"Ridge scores: {scores}\nMean: {np.array(scores).mean()}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='')
    parser.add_argument('--save_img_dir', type=str, default=None)
    parser.add_argument('--save_act_dir', type=str, default=None)
    parser.add_argument('--layer_name', type=str, default='')
    parser.add_argument('--sae_name', type=str, default=None)
    parser.add_argument('--sae_weights_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() and args.device is not None else "cpu"

    from thingsvision import get_extractor

    # model_name = 'resnet50'
    # param_name = {'weights': 'IMAGENET1K_V2'}
    # source = 'torchvision'

    # model_name = 'resnet18'
    # param_name = {'weights': 'IMAGENET1K_V1'}

    # model_name = 'vit_b_16'
    # param_name = {'weights': 'IMAGENET1K_V1'}
    # source = 'torchvision'

    model_name = 'clip'
    param_name = {'variant': 'ViT-B/32'}
    source = 'custom'

    extractor = get_extractor(
        model_name=model_name,
        source=source,
        device=device,
        pretrained=True,
        model_parameters=param_name
    )
    # extractor.model.load_state_dict(torch.load('/home/chkapoor/pytorch-cifar/checkpoint_imagenet/resnet50/seed_2/epoch_78.pth')['net'])
    # extractor.model.load_state_dict(torch.load('/home/alongon/model_weights/vit_b_16/checkpoint_seed1.pth')['model_state_dict'])

    sae_states = None
    save_img_dir = None
    save_act_dir = args.save_act_dir
    if save_act_dir is not None:
        Path(save_act_dir).mkdir(parents=True, exist_ok=True)

    if args.sae_name is not None:
        sae_states = torch.load(os.path.join(args.sae_weights_dir, f'{args.sae_name}.pth'))
        sae_states['W_enc'] = sae_states['W_enc'].to(device)
        sae_states['b_enc'] = sae_states['b_enc'].to(device)

        # # if 'archetype' in args.sae_name or 'nmf' in args.sae_name:
        # #   TODO: load encoder batch norm weights?  or maybe init a layer and load weights into it?
        # bn = torch.nn.BatchNorm1d(sae_states['W_enc'].shape[-1])
        # bn_states = {}
        # for state in sae_states.keys():
        #     if 'enc_bn' in state:
        #         bn_states[state.replace('enc_bn.', '')] = sae_states[state]

        # bn.load_state_dict(bn_states)
        # bn.to(device)
        # sae_states['bn'] = bn

        # sae_states['W_dec'] = sae_states['W_dec'].to(device)

        # sae_weights = sae_states['W'] @ sae_states['C'] + sae_states['Relax']
        # sae_weights = sae_weights.to(device).T

        # sae_weights = torch.load(os.path.join(args.sae_weights_dir, f'{args.sae_name}.pth'))['W_dec'].to(device).T

        #   TODO: init random vector with same mean and var as example sae_weights.  replace weights with this to get imnet valid
        #         acts to random dirs in activation space.

        # save_act_dir = os.path.join(args.save_act_dir, args.layer_name)
        if args.save_img_dir is not None:
            save_img_dir = os.path.join(args.save_img_dir, args.layer_name, args.sae_name)
    else:
        if args.save_img_dir is not None:
            save_img_dir = os.path.join(args.save_img_dir, args.layer_name)

    # all_acts, _ = get_img_acts(extractor, args.img_dir, args.layer_name, device, sae_weights=sae_states, batch_size=args.batch_size)
    # # np.save(os.path.join(save_act_dir, f'{args.layer_name if args.sae_name is None else args.sae_name}.npy'), np.array(all_acts))
    # np.save(os.path.join(save_act_dir, f'{args.layer_name}_train.npy'), np.array(all_acts))

    Path(save_img_dir).mkdir(parents=True, exist_ok=True)
    save_top_act_imgs(
        extractor,
        args.img_dir,
        args.layer_name,
        save_act_dir,
        save_img_dir,
        device,
        args.batch_size,
        sae_weights=sae_states,
        sae_name=args.sae_name
    )