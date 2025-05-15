import argparse
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from voxel_utils import extract_shared_for_subj, save_top_cocos, voxel_to_sae
from utils import sort_acts


#   TODO:  should this be migrated into voxel_utils??
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits_path', type=str)
    parser.add_argument('--nsd_path', type=str)
    parser.add_argument('--coco_dir', type=str)
    parser.add_argument('--sae_name', type=str)
    parser.add_argument('--weights_dir', type=str)
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--subj_id', type=int)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_voxels', type=int, default=0)
    parser.add_argument('--save_count', type=int, default=32)
    parser.add_argument('--filter_path', type=str, default=None)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    coco_ids, voxel_data = extract_shared_for_subj(args.splits_path, args.nsd_path, args.subj_id)
    voxel_data = torch.tensor(voxel_data)
    if args.filter_path is not None:
        voxel_data = voxel_data[:, np.load(args.filter_path)]
        
    if args.num_voxels > 0 and args.num_voxels < voxel_data.shape[1]:
        voxel_data = voxel_data[:, torch.randperm(voxel_data.shape[1])[:args.num_voxels]]

    voxel_ds = TensorDataset(voxel_data)
    dataloader = DataLoader(voxel_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    sae_weights = torch.load(os.path.join(args.weights_dir, f'subj{args.subj_id}', f'{args.sae_name}.pth'))['W_dec'].to(device).T
    all_sae_acts = voxel_to_sae(dataloader, sae_weights)

    all_sorted_idx = sort_acts(all_sae_acts, device, args.save_count)

    savedir = os.path.join(args.savedir, f'subj{args.subj_id}', 'sae_latents', args.sae_name)
    Path(savedir).mkdir(parents=True, exist_ok=True)

    for i in range(len(all_sorted_idx)):
        sorted_idx = all_sorted_idx[i]
        save_top_cocos(args.coco_dir, coco_ids[sorted_idx[:9]], i, savedir)