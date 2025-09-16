import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import transforms, utils
# import nimfa
from scipy.stats import kendalltau, pearsonr
import ot
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

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
    print(f'Num dead source: {torch.all(x == 0, dim=0).nonzero().shape[0]}')
    print(f'Num dead target: {torch.all(y == 0, dim=0).nonzero().shape[0]}')
    x = x[:, torch.any(x != 0, dim=0).nonzero()[:, 0]]
    y = y[:, torch.any(y != 0, dim=0).nonzero()[:, 0]]

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
    #   Remove dead latents
    X = X[:, torch.any(X != 0, dim=0).nonzero()[:, 0]]
    Y = Y[:, torch.any(Y != 0, dim=0).nonzero()[:, 0]]

    #   Remove dead stimuli (L0=0)
    X = X[torch.any(X != 0, dim=1).nonzero()[:, 0]].cpu()
    Y = Y[torch.any(Y != 0, dim=1).nonzero()[:, 0]].cpu()

    X = (X.T - X.mean(dim=1)).T
    X = (X.T / torch.sqrt(torch.sum(X**2, dim=1))).T

    Y = (Y.T - Y.mean(dim=1)).T
    Y = (Y.T / torch.sqrt(torch.sum(Y**2, dim=1))).T
    
    y_coef = 1 - np.corrcoef(Y.numpy(), dtype=np.float32)
    x_coef = 1 - np.corrcoef(X.numpy(), dtype=np.float32)
    triu_idx = np.triu_indices(len(y_coef), k=1)

    # return kendalltau(y_coef[triu_idx], x_coef[triu_idx]).statistic
    return pearsonr(y_coef[triu_idx], x_coef[triu_idx]).statistic


def cdist(X,Y):
    cross_term = np.dot(X.T, Y)
    dist_matrix = 1 - cross_term

    return dist_matrix


def remove_dead_units(X, Y, kf):
    x_rem_indices = np.array([], dtype=np.int32)
    y_rem_indices = np.array([], dtype=np.int32)
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        x_train, x_test = X[train_idx], X[test_idx]        
        x_rem_indices = np.unique(np.concatenate((x_rem_indices, np.all(x_train == 0, axis=0).nonzero()[0])))
        x_rem_indices = np.unique(np.concatenate((x_rem_indices, np.all(x_test == 0, axis=0).nonzero()[0])))

        y_train, y_test = Y[train_idx], Y[test_idx]
        y_rem_indices = np.unique(np.concatenate((y_rem_indices, np.all(y_train == 0, axis=0).nonzero()[0])))
        y_rem_indices = np.unique(np.concatenate((y_rem_indices, np.all(y_test == 0, axis=0).nonzero()[0])))

    X, Y = np.delete(X, x_rem_indices, axis=1), np.delete(Y, y_rem_indices, axis=1)
    
    return X, Y


# def remove_dead_units(train_acts, test_acts):
#     rem_indices = np.array([], dtype=np.int32)
#     rem_indices = np.unique(np.concatenate((rem_indices, np.all(train_acts == 0, axis=0).nonzero()[0])))
#     rem_indices = np.unique(np.concatenate((rem_indices, np.all(test_acts == 0, axis=0).nonzero()[0])))

#     train_acts, test_acts = np.delete(train_acts, rem_indices, axis=1), np.delete(test_acts, rem_indices, axis=1)

#     return train_acts, test_acts


def SemiMatching(X, Y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print('BEFORE DEAD REMOVAL')
    print(X.shape)
    X, Y = remove_dead_units(X, Y, kf)
    print('AFTER DEAD REMOVAL')
    print(X.shape)
    print('\n')

    scores = []
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        x_train = x_train - x_train.mean(axis=0)
        x_train = x_train / np.sqrt(np.sum(x_train**2, axis=0))
        y_train = y_train - y_train.mean(axis=0)
        y_train = y_train / np.sqrt(np.sum(y_train**2, axis=0))

        sim_matrix = 1 - cdist(x_train, y_train)
        max_indices = np.argmax(sim_matrix, axis=0)

        x_test = x_test - x_test.mean(axis=0)
        x_test = x_test / np.sqrt(np.sum(x_test**2, axis=0))
        y_test = y_test - y_test.mean(axis=0)
        y_test = y_test / np.sqrt(np.sum(y_test**2, axis=0))

        sim_matrix = 1 - cdist(x_test[:, max_indices], y_test)
        scores.append(np.mean(np.diag(sim_matrix)))

    return scores


def SoftMatching(X, Y, itermax=1000):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print('BEFORE DEAD REMOVAL')
    print(X.shape)
    X, Y = remove_dead_units(X, Y, kf)
    print('AFTER DEAD REMOVAL')
    print(X.shape)
    print('\n')

    scores = []
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        x_train, x_test = X[train_idx], X[test_idx]
        # x_train, x_test = remove_dead_units(x_train, x_test)
        y_train, y_test = Y[train_idx], Y[test_idx]
        # y_train, y_test = remove_dead_units(y_train, y_test)

        x_train = x_train - x_train.mean(axis=0)
        x_train = x_train / np.sqrt(np.sum(x_train**2, axis=0))
        y_train = y_train - y_train.mean(axis=0)
        y_train = y_train / np.sqrt(np.sum(y_train**2, axis=0))

        nx = x_train.shape[1]
        ny = y_train.shape[1]
        dist_matrix = cdist(x_train, y_train) 
        soft_assignments,log = ot.emd(
            np.ones(nx, dtype=np.float64) / nx,
            np.ones(ny, dtype=np.float64) / ny,
            dist_matrix,
            numItermax=100000*itermax,
            log=True
        )
        if log['warning'] != None:
            print('Did not converge, increase itermax')
            return np.nan
        
        x_test = x_test - x_test.mean(axis=0)
        x_test = x_test / np.sqrt(np.sum(x_test**2, axis=0))
        y_test = y_test - y_test.mean(axis=0)
        y_test = y_test / np.sqrt(np.sum(y_test**2, axis=0))

        sim_matrix = 1 - cdist(x_test, y_test)
        # scores.append(np.sum(soft_assignments * (1 - dist_matrix)))
        scores.append(np.sum(soft_assignments * sim_matrix))

    return scores


def RidgeRegression(X, Y, alpha_min=-8, alpha_max=8, num_alpha=17):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print('BEFORE DEAD REMOVAL')
    print(X.shape)
    X, Y = remove_dead_units(X, Y, kf)
    print('AFTER DEAD REMOVAL')
    print(X.shape)
    print('\n')

    scores = []
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        x_train = x_train - x_train.mean(axis=0)
        # x_train = x_train / np.sqrt(np.sum(x_train**2, axis=0))
        y_train = y_train - y_train.mean(axis=0)
        # y_train = y_train / np.sqrt(np.sum(y_train**2, axis=0))

        predictor = RidgeCV(alphas=np.logspace(alpha_min, alpha_max, num_alpha), fit_intercept=False)
        predictor.fit(x_train, y_train)

        # y_pred = predictor.predict(x_train)
        # y_pred = y_pred / np.sqrt(np.sum(y_pred**2, axis=0))
        # y_train = y_train / np.sqrt(np.sum(y_train**2, axis=0))
        # sim_matrix = 1 - cdist(y_train, y_pred)
        
        x_test = x_test - x_test.mean(axis=0)
        # x_test = x_test / np.sqrt(np.sum(x_test**2, axis=0))
        y_test = y_test - y_test.mean(axis=0)
        # y_test = y_test / np.sqrt(np.sum(y_test**2, axis=0))

        y_pred = predictor.predict(x_test)
        y_pred = y_pred / np.sqrt(np.sum(y_pred**2, axis=0))
        y_test = y_test / np.sqrt(np.sum(y_test**2, axis=0))
        sim_matrix = 1 - cdist(y_test, y_pred)
        scores.append(np.mean(np.diag(sim_matrix)))

    return scores