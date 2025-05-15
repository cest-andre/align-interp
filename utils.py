import os
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms, utils


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


#   x and y are torch tensors with shape: (num_data, num_units).  Pairwise corrs obtained between all units.
def pairwise_corr(x, y):
    #   TODO: filter dead neurons???  no nonzeros after relu
    x = torch.clamp(x, min=0)
    y = torch.clamp(y, min=0)
    x_cent = x - torch.mean(x, 0)
    x_ss = torch.sum(torch.pow(x_cent, 2), 0)

    y_cent = y - torch.mean(y, 0)
    y_ss = torch.sum(torch.pow(y_cent, 2), 0)

    corrs = (x_cent.T @ y_cent) / (torch.sqrt(torch.outer(x_ss, y_ss)) + 1e-10)
    corrs = torch.clamp(corrs, min=-1, max=1)

    return corrs