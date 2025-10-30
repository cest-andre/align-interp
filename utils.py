import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import transforms, utils
# import nimfa
from scipy.stats import kendalltau, pearsonr
from scipy.io import loadmat
import ot
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

from sae import LN


# def sparse_nmf(acts, rank):
#     snmf = nimfa.Snmf(acts, seed=None, rank=rank, max_iter=30, version='l', beta=1e-4, i_conv=10, w_min_change=0)
#     results = snmf()

#     return results.basis()


def sort_acts(all_acts, save_count, save_idx=None):
    idx = np.arange(len(all_acts)).tolist()
    all_acts = np.transpose(np.array(all_acts))

    acts_iter = range(save_count)
    if save_idx is not None:
        acts_iter = save_idx

    all_sorted_idx = []
    for i in acts_iter:
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
    grid.save(os.path.join(grids_path, f"{unit_id}_vinken.png"))

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


def load_vinken_imgs(img_dir):
    imgs = loadmat(img_dir)['imarray']
    imgs = np.transpose(imgs / 255, (3, 2, 0, 1))
    imgs = torch.tensor(imgs, dtype=torch.float)

    return imgs


#   x and y are torch tensors with shape: (num_data, num_units).  Pairwise corrs obtained between all units.
def pairwise_corr(x, y):
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


def var_explained(Y, Y_pred):
    E = Y_pred - Y
    SS_res = torch.sum(E**2)
    SS_tot = torch.sum(Y**2)
    R2 = 1 - SS_res / SS_tot

    return R2.cpu().numpy()


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
    
    return X, Y, x_rem_indices


def SemiMatching(X, Y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print('BEFORE DEAD REMOVAL')
    print(X.shape)
    X, Y, _ = remove_dead_units(X, Y, kf)
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

    scores = []
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

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


def RidgeRegression(X, Y, alpha_min=-8, alpha_max=8, num_alpha=17, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    print('BEFORE DEAD REMOVAL')
    print(X.shape)
    print(Y.shape)
    X, Y, x_rem_indices = remove_dead_units(X, Y, kf)
    print('AFTER DEAD REMOVAL')
    print(X.shape)
    print(Y.shape)
    print('\n')
    X, _, _ = LN(torch.tensor(X))
    X = X.numpy()
    Y, _, _ = LN(torch.tensor(Y))
    Y = Y.numpy()

    scores = []
    var_exps = []
    mses = []
    all_coeffs = []
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        x_train, x_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        # x_train = x_train - x_train.mean(axis=0)
        # # x_train = x_train / np.sqrt(np.sum(x_train**2, axis=0))
        # y_train = y_train - y_train.mean(axis=0)
        # # y_train = y_train / np.sqrt(np.sum(y_train**2, axis=0))

        # x_train, _, _ = LN(torch.tensor(x_train))
        # x_train = x_train.numpy()
        # y_train, _, _ = LN(torch.tensor(y_train))
        # y_train = y_train.numpy()

        predictor = RidgeCV(alphas=np.logspace(alpha_min, alpha_max, num_alpha), fit_intercept=False)
        predictor.fit(x_train, y_train)

        all_coeffs.append(predictor.coef_)

        # y_pred = predictor.predict(x_train)
        # y_pred = y_pred / np.sqrt(np.sum(y_pred**2, axis=0))
        # y_train = y_train / np.sqrt(np.sum(y_train**2, axis=0))
        # sim_matrix = 1 - cdist(y_train, y_pred)
        
        # x_test = x_test - x_test.mean(axis=0)
        # x_test = x_test / np.sqrt(np.sum(x_test**2, axis=0))
        # y_test = y_test - y_test.mean(axis=0)
        # y_test = y_test / np.sqrt(np.sum(y_test**2, axis=0))

        y_pred = predictor.predict(x_test)
        mse = np.mean(np.sum((y_pred - y_test)**2, axis=1))
        mses.append(mse)

        y_pred = y_pred / np.sqrt(np.sum(y_pred**2, axis=0))
        y_test = y_test / np.sqrt(np.sum(y_test**2, axis=0))
        sim_matrix = 1 - cdist(y_test, y_pred)
        scores.append(np.mean(np.diag(sim_matrix)))

        var_exps.append(var_explained(torch.tensor(y_test), torch.tensor(y_pred)))

    return scores, all_coeffs, x_rem_indices#, var_exps, mses