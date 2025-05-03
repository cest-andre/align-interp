import argparse
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from data_utils import extract_shared_for_subj, save_top_cocos


#   TODO: Create dataloader from val_voxels and compute latent acts via dot product of voxel responses and latent's decoder weights.
#         Record acts to all latents for all stimuli, sort by descending act, use indices from this sort to index coco_ids (follow imnet_acts logic).
#         Pass to a top-K image saver function which takes in indices and coco_ids list and loads the top-K images, then saves to savedir.
#         Allow option to perform on raw voxels, or define separate function is easier.


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
    parser.add_argument('--num_voxels', type=int, default=5931)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    coco_ids, val_voxels = extract_shared_for_subj(args.splits_path, args.nsd_path, args.subj_id)
    val_voxels = torch.tensor(val_voxels)
    voxel_ds = TensorDataset(val_voxels)
    dataloader = DataLoader(voxel_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    latent_weights = torch.load(os.path.join(args.weights_dir, f'subj{args.subj_id}', f'{args.sae_name}.pth'))['W_dec'].to(device).T

    im_count = 0
    all_latent_acts = []
    for i, voxels in enumerate(dataloader):
        voxels = voxels[0][:, :args.num_voxels].to(device)
        im_count += voxels.shape[0]

        latent_acts = voxels @ latent_weights
        all_latent_acts.append(latent_acts.cpu().numpy())

    all_idx = np.arange(im_count).tolist()
    all_latent_acts = [act for batch in all_latent_acts for act in batch]
    all_latent_acts = np.transpose(np.array(all_latent_acts))

    savedir = os.path.join(args.savedir, f'subj{args.subj_id}', 'sae_latents', args.sae_name)
    Path(savedir).mkdir(parents=True, exist_ok=True)

    for i in range(32):
        latent_acts = all_latent_acts[i]
        sorted_acts, sorted_idx = zip(*sorted(zip(latent_acts.tolist(), all_idx), reverse=True))
        sorted_idx = np.array(sorted_idx)

        save_top_cocos(args.coco_dir, coco_ids[sorted_idx[:9]], i, savedir, save_inh=False)