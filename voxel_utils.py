import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
from torchvision import transforms, utils
from sklearn.cluster import KMeans

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'representation-alignment'))
from src.alignment.linear import Linear

from sae import LN
from utils import save_top_imgs, pairwise_corr, pairwise_jaccard, RSA


def extract_shared_for_subj(split_ids_path, nsd_path, subj_id):
    shared_ids = pickle.load(open(split_ids_path, 'rb'))['train']  #  'test' contains the 1k coco images shared across subjects
    all_voxels = pickle.load(open(nsd_path, 'rb'))

    coco_ids = []
    subj_voxels = []
    for coco_id in shared_ids:
        voxels = all_voxels[coco_id][subj_id]
        if voxels is not None:
            coco_ids.append(coco_id)
            subj_voxels.append(voxels)

    coco_ids = np.array(coco_ids)
    subj_voxels = np.array(subj_voxels)
    
    return coco_ids, subj_voxels


def extract_train_for_subj(split_ids_path, nsd_path, subj_id):
    train_ids = pickle.load(open(split_ids_path, 'rb'))['train']
    all_voxels = pickle.load(open(nsd_path, 'rb'))

    subj_voxels = []
    for coco_id in train_ids:
        voxels = all_voxels[coco_id][subj_id]
        if voxels is not None:
            subj_voxels.append(voxels)

    subj_voxels = np.array(subj_voxels)
    return subj_voxels


def voxel_to_sae(dataloader, sae_weights, device):
    all_sae_acts = []
    for _, vox in enumerate(dataloader):
        vox = vox[0].to(device)
        vox, _, _ = LN(vox)

        acts = (vox.to(torch.float) @ sae_weights['W_enc']) + sae_weights['b_enc']

        #   topk activation SAE
        topk_res = torch.topk(acts, k=256, dim=-1)
        values = torch.nn.ReLU()(topk_res.values)
        acts = torch.zeros_like(acts, device=acts.device)
        acts.scatter_(-1, topk_res.indices, values)

        all_sae_acts += acts.cpu().numpy().tolist()

    return all_sae_acts


def viz_voxels():
    pass
    # coco_ids, voxel_data = extract_shared_for_subj(args.splits_path, args.nsd_path, args.subj_id)
    # voxel_data = torch.tensor(voxel_data)
    # if args.filter_path is not None:
    #     voxel_data = voxel_data[:, np.load(args.filter_path)]
        
    # if args.num_voxels > 0 and args.num_voxels < voxel_data.shape[1]:
    #     voxel_data = voxel_data[:, torch.randperm(voxel_data.shape[1])[:args.num_voxels]]

    # voxel_ds = TensorDataset(voxel_data)
    # dataloader = DataLoader(voxel_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # sae_weights = torch.load(os.path.join(args.weights_dir, f'subj{args.subj_id}', f'{args.sae_name}.pth'))['W_dec'].to(device).T
    # all_sae_acts = voxel_to_sae(dataloader, sae_weights)

    # all_sorted_idx = sort_acts(all_sae_acts, device, args.save_count)

    # savedir = os.path.join(args.savedir, f'subj{args.subj_id}', 'sae_latents', args.sae_name)
    # Path(savedir).mkdir(parents=True, exist_ok=True)

    # for i in range(len(all_sorted_idx)):
    #     sorted_idx = all_sorted_idx[i]
    #     save_top_cocos(args.coco_dir, coco_ids[sorted_idx[:9]], i, savedir)


def voxel_dnn_align(splits_path, nsd_path, coco_dir, subj, dnn_acts, voxel_acts):
    subj_coco_ids, _ = extract_shared_for_subj(splits_path, nsd_path, subj)
    all_ids = [f.split('.jpg')[0] for f in os.listdir(coco_dir)]
    subj_coco_ids = [all_ids.index(str(id)) for id in subj_coco_ids]
    dnn_acts = dnn_acts[subj_coco_ids]

    # metric = Linear()
    # results = metric.fit_kfold_ridge(x=dnn_acts.to(torch.float), y=voxel_acts.to(torch.float))
    # score = np.array(results).mean()

    results = pairwise_corr(dnn_acts, voxel_acts)
    score = torch.mean(torch.max(results, 1)[0]).cpu().numpy()

    # print(f'Pairwise score: {pairwise_score}')
    # rsa_score = RSA(dnn_acts.cpu().numpy(), voxel_acts.cpu().numpy())[0]
    # print(f'RSA score: {rsa_score}')
    # exit()

    # for p in range(25, 125, 25):
    #     subset_idx = int(dnn_acts.shape[0] * (p / 100))
    #     metric = Linear()
    #     scores = metric.fit_kfold_ridge(
    #         x=dnn_acts[:subset_idx].cpu().to(torch.float),
    #         y=voxel_acts[:subset_idx].cpu().to(torch.float),
    #         #val_set={'x': source_val, 'y': target_val}
    #     )
    #     print(f"Ridge scores: {scores}\nMean: {np.array(scores).mean()}\n")

    # results = pairwise_corr(dnn_acts, voxel_acts)
    # print(f'Pearson pairwise: {torch.mean(torch.max(results, 1)[0])}')
    # print(f'Pearson pairwise: {torch.mean(torch.max(results, 0)[0])}')
    # score = pairwise_jaccard(dnn_acts, voxel_acts)
    # print(f'Jaccard pairwise: {score}')

    return score


#   Use kmeans to reduce redundant voxels.  Plot voxels in the space of activations across all stimuli (all subject's NSD responses).
#   Set k to the desired number of voxels to be extracted.  After kmeans is complete, each centroid is a tuning curve, so take corrs
#   of all voxel tuning curves with that to obtain the best voxel match for that centroid. 
def kmeans_voxel_reduce(voxels, savedir, subj_id, k=128):
    voxels = np.transpose(voxels)  #  we want voxels to be our samples in "NSD activation space" to find redundancies.
    results = KMeans(n_clusters=k, random_state=0).fit(voxels)

    all_min_idx = []
    for c in range(k):
        cluster_idx = np.nonzero(results.labels_ == c)[0]
        centroid = results.cluster_centers_[c]
        centroid = np.broadcast_to(centroid[None, :], (cluster_idx.shape[0], centroid.shape[0]))
        min_idx = cluster_idx[np.argmin(np.sum((voxels[cluster_idx] - centroid)**2, axis=-1))]
        all_min_idx.append(min_idx)

    np.save(os.path.join(savedir, f'subj{subj_id}_{k}means_voxel_idx.npy'), np.array(all_min_idx))
    return np.array(all_min_idx)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits_path', type=str)
    parser.add_argument('--nsd_path', type=str)
    parser.add_argument('--coco_dir', type=str)
    parser.add_argument('--dnn_acts_path', type=str)
    parser.add_argument('--voxel_acts_path', type=str)
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--subj_id', type=int)
    parser.add_argument('--sae_name', type=str)
    parser.add_argument('--weights_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--save_count', type=int, default=32)
    parser.add_argument('--filter_path', type=str, default=None)
    parser.add_argument('--device', type=int)
    parser.add_argument('--reduce_k', type=int, default=0)
    args = parser.parse_args()

    # subj_voxels = extract_train_for_subj(args.splits_path, args.nsd_path, args.savedir, args.subj_id)

    # file_name = f'subj_{args.subj_id}_'
    # if args.reduce_k > 0:
    #     filtered_idx = kmeans_voxel_reduce(subj_voxels, args.savedir, args.subj_id, k=args.reduce_k)
    #     subj_voxels = subj_voxels[:, filtered_idx]
    #     file_name += f'{args.reduce_k}means_filtered_'

    # file_name += 'train_voxels.npy'
    # print(subj_voxels.shape)
    # np.save(os.path.join(args.savedir, file_name), subj_voxels)


    #   Extract voxel sae latent acts
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    subj_voxels = np.load(args.voxel_acts_path)
    voxel_ds = TensorDataset(torch.tensor(subj_voxels))
    dataloader = DataLoader(voxel_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    sae_weights = torch.load(os.path.join(args.weights_dir, f'subj{args.subj_id}', f'{args.sae_name}.pth'))
    sae_weights['W_enc'] = sae_weights['W_enc'].to(device)
    sae_weights['b_enc'] = sae_weights['b_enc'].to(device)

    sae_acts = voxel_to_sae(dataloader, sae_weights, device)
    np.save(os.path.join(args.savedir, f'subj{args.subj_id}', 'voxel_sae_acts', f'{args.sae_name}.npy'), np.array(sae_acts))


    # #   Load DNN coco acts, extract only images that are for subj.  Then load voxel acts (raw or sae) and run pairwise corrs.
    # device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    # dnn_acts = np.load(args.dnn_acts_path)
    # dnn_acts = np.clip(dnn_acts, a_min=0, a_max=None)
    # voxel_acts = np.load(args.voxel_acts_path)

    # voxel_dnn_align(args.splits_path, args.nsd_path, args.coco_dir, args.subj_id, dnn_acts, voxel_acts, device)