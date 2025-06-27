import sys
import os
import h5py
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'representation-alignment'))
from src.alignment.linear import Linear


#   mm = ManyMonkeys electrophys dataset.
def get_mm_imgs(mm_path, transform):
    with h5py.File(mm_path, "r") as f:
        imgs = np.array(list(f['stimuli']))
        imgs = (imgs*255).astype(np.uint8)
        imgs = [transform(Image.fromarray(img)) for img in imgs]

    return imgs


def get_mm_responses(mm_path, name):
    monkey_names = ['bento', 'chabo', 'magneto', 'nano', 'solo', 'tito']
    assert name in monkey_names

    with h5py.File(mm_path, "r") as f:
        mm_data = []
        # for monkey in monkey_names:
        responses = np.nanmean(np.asarray(list(f[name]['left']['rates'])), -1)
        if 'right' in list(f[name].keys()):
            right = np.nanmean(np.asarray(list(f[name]['right']['rates'])), -1)
            responses = np.hstack((responses, right))

        responses = responses[:, responses.sum(0) != 0]  # eliminate dead neurons
        responses = np.nan_to_num(responses, posinf=0, neginf=0)

    return responses


def monkey_dnn_align(responses, dnn_acts, device):
    dnn_acts = torch.tensor(dnn_acts, device=device)
    responses = torch.tensor(responses, device=device)

    metric = Linear()
    print(f"Linear scores: {metric.fit_kfold_score(x=dnn_acts, y=responses)}")
    print(f"Ridge scores: {metric.fit_kfold_ridge(x=dnn_acts.cpu().to(torch.float), y=responses.cpu().to(torch.float))}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mm_path', type=str)
    parser.add_argument('--monkey_name', type=str)
    parser.add_argument('--dnn_acts_path', type=str)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    # get_mm_responses('/mnt/cogsci/ManyMonkeys/many_monkeys2.h5', 'bento')
    # get_mm_imgs('/mnt/cogsci/ManyMonkeys/many_monkeys2.h5')

    dnn_acts = np.load(args.dnn_acts_path)
    responses = get_mm_responses(args.mm_path, args.monkey_name)
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    monkey_dnn_align(responses, dnn_acts, device)