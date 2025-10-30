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
from utils import save_top_cocos, sort_acts, RSA, SemiMatching, SoftMatching, RidgeRegression


def extract_coco_ids(splits_path, nsd_root, subj_id, split=None):
    split_ids = pickle.load(open(splits_path, 'rb'))
    if split is None:
        id_accum = []
        for k in split_ids.keys():
            id_accum += split_ids[k]
        split_ids = id_accum
    else:
        split_ids = split_ids[split]

    all_ids = np.load(os.path.join(nsd_root, 'ventral_visual_data', f'subj{subj_id}_coco_IDs.npy'))
    coco_ids = []
    idx = []
    for i in range(all_ids.shape[0]):
        if all_ids[i] in split_ids:
            coco_ids.append(all_ids[i])
            idx.append(i)

    return np.array(coco_ids), idx


# def extract_shared_for_subj(split_ids_path, nsd_path, subj_id):
#     shared_ids = pickle.load(open(split_ids_path, 'rb'))['train']  #  'test' contains the 1k coco images shared across subjects
#     all_voxels = pickle.load(open(nsd_path, 'rb'))

#     coco_ids = []
#     subj_voxels = []
#     for coco_id in shared_ids:
#         voxels = all_voxels[coco_id][subj_id]
#         if voxels is not None:
#             coco_ids.append(coco_id)
#             subj_voxels.append(voxels)

#     coco_ids = np.array(coco_ids)
#     subj_voxels = np.array(subj_voxels)
    
#     return coco_ids, subj_voxels


# def extract_train_for_subj(split_ids_path, nsd_path, subj_id):
#     train_ids = pickle.load(open(split_ids_path, 'rb'))['train']
#     all_voxels = pickle.load(open(nsd_path, 'rb'))

#     subj_voxels = []
#     for coco_id in train_ids:
#         voxels = all_voxels[coco_id][subj_id]
#         if voxels is not None:
#             subj_voxels.append(voxels)

#     subj_voxels = np.array(subj_voxels)
#     return subj_voxels


def voxel_to_sae(dataloader, sae_weights, device, topk=False):
    all_sae_acts = []
    for _, vox in enumerate(dataloader):
        vox = vox[0].to(device)
        # vox, _, _ = LN(vox)

        acts = (vox.to(torch.float) @ sae_weights['W_enc']) + sae_weights['b_enc']

        #   topk activation SAE
        if topk:
            topk_res = torch.topk(acts, k=8, dim=-1)
            values = torch.nn.ReLU()(topk_res.values)
            acts = torch.zeros_like(acts, device=acts.device)
            acts.scatter_(-1, topk_res.indices, values)
        else:
            acts = torch.nn.ReLU()(acts)

        all_sae_acts += acts.cpu().numpy().tolist()

    return all_sae_acts


def viz_voxels(acts, coco_dir, coco_ids, save_img_dir, device, save_count=64):
    # save_idx = [200, 1148, 1201, 1337, 1320,  231,  276, 1389,  916,  620,  702,  590,  223, 1268, 226,  715, 1247,  273,  195,  442,  421,  272, 1227,  756,  132, 1079,  900,  263, 472,  610,  470, 1317]
    all_sorted_idx = sort_acts(acts, save_count, save_idx=None)
    for i in range(len(all_sorted_idx)):
        sorted_idx = all_sorted_idx[i]
        save_top_cocos(coco_dir, coco_ids[sorted_idx[:9]], i, save_img_dir)


def voxel_dnn_align(coco_dir, subj_id, nsd_root, dnn_acts, voxel_acts, n_splits=5):
    subj_coco_ids = np.load(os.path.join(nsd_root, 'ventral_visual_data', f'subj{subj_id}_coco_IDs.npy'))
    all_ids = [f.split('.jpg')[0] for f in os.listdir(coco_dir)]
    subj_coco_ids = [all_ids.index(str(id)) for id in subj_coco_ids]
    dnn_acts = dnn_acts[subj_coco_ids]

    coeffs = None
    x_rem_indices = None
    # scores = np.array(SemiMatching(dnn_acts.numpy().astype(np.float64), voxel_acts.numpy().astype(np.float64)))
    # scores = np.array(SoftMatching(dnn_acts.numpy().astype(np.float64), voxel_acts.numpy().astype(np.float64)))
    scores, coeffs, x_rem_indices = RidgeRegression(dnn_acts.numpy().astype(np.float64), voxel_acts.numpy().astype(np.float64), n_splits=n_splits)
    # scores = np.array(RSA(dnn_acts, voxel_acts))

    if coeffs is not None:
        coeffs = coeffs[0]

    return scores, coeffs, x_rem_indices


def get_nsd_surrogate_sae(data_dir, save_dir, coco_dir, subj_id, nsd_root, dnn_acts, voxels):
    model = 'resnet50'
    layer = 'layer4.2'
    dnn_opt_topk = {'k': 41, 'epoch': 300, 'exp': 2}

    source_sae_acts = torch.tensor(
        np.load(os.path.join(basedir, 'coco_acts', model, 'sae_latents', layer, f'top{dnn_opt_topk["k"]}_aux_{dnn_opt_topk["exp"]}exp_sae_weights_{dnn_opt_topk["epoch"]}ep.npy'))
    )
    voxel_acts = torch.tensor(
        np.load(os.path.join(nsd_root, 'ventral_visual_data', f'subj{subj_id}.npy'))
    )
    align_scores, coeffs, x_rem_indices = voxel_dnn_align(coco_dir, subj_id, nsd_root, dnn_acts, voxels, n_splits=2)
    np.save(os.path.join(save_dir, f'{model}_{layer}_top{dnn_opt_topk["k"]}_{dnn_opt_topk["exp"]}exp_sae_weights_{dnn_opt_topk["epoch"]}ep_linreg_coeff.npy'), coeffs)


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
    parser.add_argument('--nsd_root', type=str)
    parser.add_argument('--brain_region', type=str)
    parser.add_argument('--kmeans_subset', type=str, default=None)
    parser.add_argument('--save_acts_dir', type=str)
    parser.add_argument('--save_img_dir', type=str)
    parser.add_argument('--subj_id', type=int)
    parser.add_argument('--sae_name', type=str)
    parser.add_argument('--sae_weights_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=int)
    parser.add_argument('--reduce_k', type=int, default=0)
    args = parser.parse_args()

    # subj_voxels = np.load(os.path.join(args.nsd_root, f'{args.brain_region}_data', f'subj{args.subj_id}.npy'))
    # file_name = f'subj{args.subj_id}_'
    # if args.reduce_k > 0:
    #     filtered_idx = kmeans_voxel_reduce(subj_voxels, args.save_acts_dir, args.subj_id, k=args.reduce_k)
    #     subj_voxels = subj_voxels[:, filtered_idx]
    #     file_name += f'{args.reduce_k}means_filtered_'

    # file_name += 'train_voxels.npy'
    # print(subj_voxels.shape)
    # np.save(os.path.join(args.save_acts_dir, file_name), subj_voxels)
    # exit()


    #   Extract voxel sae latent acts
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    subj_voxels = np.load(os.path.join(args.nsd_root, f'{args.brain_region}_data', f'subj{args.subj_id}.npy'))
    if args.kmeans_subset is not None:
        kmeans_idx = np.load(args.kmeans_subset)
        print(kmeans_idx[-32:])
        subj_voxels = subj_voxels[:, kmeans_idx]

    subj_voxels, _, _ = LN(torch.tensor(subj_voxels))
    subj_voxels = subj_voxels.numpy()

    voxel_ds = TensorDataset(torch.tensor(subj_voxels))
    dataloader = DataLoader(voxel_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # sae_weights = torch.load(os.path.join(args.sae_weights_dir, f'{args.brain_region}', f'subj{args.subj_id}', f'{args.sae_name}.pth'))
    # sae_weights['W_enc'] = sae_weights['W_enc'].to(device)
    # sae_weights['b_enc'] = sae_weights['b_enc'].to(device)

    sae_weights = {}
    sae_weights['W_enc'] = torch.tensor(
        np.load(os.path.join(args.sae_weights_dir, f'{args.brain_region}', f'subj{args.subj_id}', f'{args.sae_name}.npy'))
    ).to(device).to(torch.float)
    sae_weights['b_enc'] = torch.zeros(sae_weights['W_enc'].shape[1]).to(device).to(torch.float)

    save_acts_dir = os.path.join(args.save_acts_dir, f'{args.brain_region}', f'subj{args.subj_id}', 'voxel_sae_acts')
    Path(save_acts_dir).mkdir(parents=True, exist_ok=True)
    sae_acts = voxel_to_sae(dataloader, sae_weights, device)
    np.save(os.path.join(save_acts_dir, f'{args.sae_name}.npy'), np.array(sae_acts))

    #   Save top images
    # coco_ids, _ = extract_shared_for_subj(args.splits_path, args.nsd_root, args.subj_id)
    coco_ids, _ = extract_coco_ids(args.splits_path, args.nsd_root, args.subj_id)
    save_img_dir = os.path.join(args.save_img_dir, f'{args.brain_region}', f'subj{args.subj_id}', 'sae_latents', f'{args.sae_name}')
    Path(save_img_dir).mkdir(parents=True, exist_ok=True)
    viz_voxels(sae_acts, os.path.join(args.nsd_root, 'images'), coco_ids, save_img_dir, device)