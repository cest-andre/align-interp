import os
import argparse
from pathlib import Path
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms, utils


def get_coco_imgs(coco_dir, coco_ids):
    imgs = []
    for id in coco_ids:
        imgs.append(Image.open(os.path.join(coco_dir, f'{id}.jpg')))

    return imgs


def save_top_cocos(coco_dir, coco_ids, unit_id, savedir, save_inh=False):
    to_tensor = transforms.ToTensor()
    imgs = get_coco_imgs(coco_dir, coco_ids)
    imgs = [to_tensor(img) for img in imgs]

    grids_path = os.path.join(savedir, "all_grids")
    Path(grids_path).mkdir(exist_ok=True)

    grid = utils.make_grid(imgs, nrow=3)
    grid = transforms.ToPILImage()(grid)
    grid.save(os.path.join(grids_path, f"{unit_id}.png"))

    if save_inh:
        inh_grids_path = os.path.join(path, "inh_grids")
        Path(inh_grids_path).mkdir(exist_ok=True)

        inh_imgs = []
        for i in range(n):
            img = imgs[int(lista[-(i+1)])]
            inh_imgs.append(img)
            img = topil(img)
            img.save(os.path.join(neuron_path, f"{i}_bottom.png"))

        grid = utils.make_grid(inh_imgs, nrow=3)
        grid = torchvision.transforms.ToPILImage()(grid)
        grid.save(os.path.join(inh_grids_path, f"{neuron_id}.png"))


def extract_shared_for_subj(split_ids_path, nsd_path, subj_id):
    shared_ids = pickle.load(open(split_ids_path, 'rb'))['train']  #  test contains the 1k coco images shared across subjects
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


def extract_train_for_subj(split_ids_path, nsd_path, savedir, subj_id):
    train_ids = pickle.load(open(split_ids_path, 'rb'))['train']
    all_voxels = pickle.load(open(nsd_path, 'rb'))

    subj_voxels = []
    for coco_id in train_ids:
        voxels = all_voxels[coco_id][subj_id]
        if voxels is not None:
            subj_voxels.append(voxels)

    subj_voxels = np.array(subj_voxels)
    print(subj_voxels.shape)
    np.save(os.path.join(savedir, f'subj_{subj_id}_train_voxels.npy'), subj_voxels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits_path', type=str)
    parser.add_argument('--nsd_path', type=str)
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--subj_id', type=int)
    args = parser.parse_args()

    extract_train_for_subj(args.splits_path, args.nsd_path, args.savedir, args.subj_id)