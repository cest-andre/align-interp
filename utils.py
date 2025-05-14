import os
from pathlib import Path
import numpy as np
from torchvision import transforms, utils


def sort_acts(dataloader, device, save_count, sae_weights=None):
    act_count = 0
    all_acts = []
    for _, act in enumerate(dataloader):
        act = act[0].to(device)
        act_count += act.shape[0]

        if sae_weights is not None:
            act = act @ sae_weights

        all_acts.append(act.cpu().numpy())

    idx = np.arange(act_count).tolist()
    all_acts = [act for batch in all_acts for act in batch]
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