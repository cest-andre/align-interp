import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import transforms, utils
# import nimfa
from scipy.stats import kendalltau
import ot
from sklearn.model_selection import KFold

from sae import LN


# def sparse_nmf(acts, rank):
#     snmf = nimfa.Snmf(acts, seed=None, rank=rank, max_iter=30, version='l', beta=1e-4, i_conv=10, w_min_change=0)
#     results = snmf()

#     return results.basis()


def sort_acts(all_acts, save_count):
    idx = np.arange(len(all_acts)).tolist()
    all_acts = np.transpose(np.array(all_acts))

    all_sorted_idx = []
    for i in range(save_count):
        act = all_acts[i]
        _, sorted_idx = zip(*sorted(zip(act.tolist(), idx), reverse=True))
        sorted_idx = np.array(sorted_idx)
        all_sorted_idx.append(sorted_idx)

    return all_sorted_idx


def save_top_imgs(top_imgs, savedir, unit_id, bot_imgs=None):
    grids_path = os.path.join(savedir, "all_grids")
    Path(grids_path).mkdir(exist_ok=True)

    grid = utils.make_grid(top_imgs, nrow=3)
    grid = transforms.ToPILImage()(grid)
    grid.save(os.path.join(grids_path, f"{unit_id}.png"))

    if bot_imgs is not None:
        bot_grids_path = os.path.join(path, "bot_grids")
        Path(bot_grids_path).mkdir(exist_ok=True)

        # inh_imgs = []
        # for i in range(n):
        #     img = imgs[int(lista[-(i+1)])]
        #     inh_imgs.append(img)
        #     img = topil(img)
        #     img.save(os.path.join(neuron_path, f"{unit_id}_bottom.png"))

        grid = utils.make_grid(bot_imgs, nrow=3)
        grid = transforms.ToPILImage()(grid)
        grid.save(os.path.join(bot_grids_path, f"{unit_id}.png"))


def get_coco_imgs(coco_dir, coco_ids, transform=None):
    if coco_ids is None:
        coco_ids = [f.split('.jpg')[0] for f in os.listdir(coco_dir)]
    imgs = []
    for id in coco_ids:
        img = Image.open(os.path.join(coco_dir, f'{id}.jpg'))
        if transform is not None:
            img = transform(img)
        imgs.append(img)

    return imgs


def save_top_cocos(coco_dir, coco_ids, unit_id, savedir, bot_ids=None):
    to_tensor = transforms.ToTensor()
    top_imgs = get_coco_imgs(coco_dir, coco_ids)
    top_imgs = [to_tensor(img) for img in top_imgs]

    bot_imgs = None
    if bot_ids is not None:
        bot_imgs = get_coco_imgs(coco_dir, bot_ids)
        bot_imgs = [to_tensor(img) for img in bot_imgs]

    save_top_imgs(top_imgs, savedir, unit_id, bot_imgs)


#   x and y are torch tensors with shape: (num_data, num_units).  Pairwise corrs obtained between all units.
def pairwise_corr(x, y):
    # x = torch.clamp(x, min=0)
    # y = torch.clamp(y, min=0)

    #   Measure sparseness (percent 0) to see if that's what's driving higher corr with SAE latents.
    # print(torch.nonzero(torch.flatten(x) > 0).shape[0] / (x.shape[0] * x.shape[1]))
    # print(torch.nonzero(torch.flatten(y) > 0).shape[0] / (y.shape[0] * y.shape[1]))

    x, _, _ = LN(x)
    y, _, _ = LN(y)

    x_cent = x - torch.mean(x, 0)
    x_ss = torch.sum(torch.pow(x_cent, 2), 0)

    y_cent = y - torch.mean(y, 0)
    y_ss = torch.sum(torch.pow(y_cent, 2), 0)

    corrs = (x_cent.T @ y_cent) / (torch.sqrt(torch.outer(x_ss, y_ss)) + 1e-10)
    corrs = torch.clamp(corrs, min=-1, max=1)

    return corrs


def pairwise_jaccard(x, y):
    x = torch.clamp(x, min=0, max=None).T
    y = torch.clamp(y, min=0, max=None).T

    scores = []
    for i in range(y.shape[0]):
        top_score = 0
        for j in range(x.shape[0]):
            score = (torch.logical_and(y[i], x[j]).sum(dim=-1) / torch.logical_or(y[i], x[j]).sum(dim=-1)).cpu().numpy()
            if score > top_score:
                top_score = score

        scores.append(top_score)

    return np.mean(np.array(scores))


def neural_alignment(source, target):
    metric = Linear()
    print(f"Ridge scores: {metric.fit_kfold_ridge(x=source, y=target)}\n")

    # results = pairwise_corr(dnn_acts, voxel_acts)
    # print(f'Pearson pairwise: {torch.mean(torch.max(results, 1)[0])}')
    # print(f'Pearson pairwise: {torch.mean(torch.max(results, 0)[0])}')
    # score = pairwise_jaccard(dnn_acts, voxel_acts)
    # print(f'Jaccard pairwise: {score}')


#   Sourced from:  https://github.com/anshksoni/NeuroAIMetrics/blob/main/utils/metrics.py#L193
def RSA(X, Y):
    y_coef = 1 - np.corrcoef(Y, dtype=np.float32)
    x_coef = 1 - np.corrcoef(X, dtype=np.float32)

    return kendalltau(np.triu(y_coef, k=1), np.triu(x_coef, k=1))