import os
import argparse
import numpy as np
import pickle


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