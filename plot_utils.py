import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'representation-alignment'))
from src.alignment.linear import Linear
from src.alignment.soft_match import SoftMatch

from utils import pairwise_corr, pairwise_jaccard, RSA


def plot_toy_alignments(basedir):
    num_feats = 64
    neurons = [8, 16, 32]
    opt_topk = [7, 7, 7]

    # train_split = 0.8
    train_split = 0.99  #  for NNLS
    # train_split = 0.999  #  for RSA
    source_seed = 0
    target_seed = 1
    device = f"cuda:4" if torch.cuda.is_available() else "cpu"

    width = 0.25
    fig, ax = plt.subplots(layout='constrained')
    x_labels = ()
    legend_labels = ['Neuron->Neuron', 'SAE->Neuron', 'SAE->SAE']
    for i in range(len(neurons)):
        datadir = os.path.join(basedir, f'{num_feats}feats_{neurons[i]}neurons_powerlaw')
        x_labels += (f'{num_feats} features, {neurons[i]} neurons\nTopK={opt_topk[i]}',)

        source_neurons = np.load(os.path.join(datadir, f'model{source_seed}_raw_neuron_acts.npy'))
        subset = -int(source_neurons.shape[0]*(1-train_split))
        source_neurons = torch.tensor(source_neurons[subset:], device=device)
        target_neurons = np.load(os.path.join(datadir, f'model{target_seed}_raw_neuron_acts.npy'))
        target_neurons = torch.tensor(target_neurons[subset:], device=device)

        source_latents = np.load(os.path.join(datadir, f'model{source_seed}_top{opt_topk[i]}_sae_acts.npy'))
        source_latents = torch.tensor(source_latents[subset:], device=device)
        target_latents = np.load(os.path.join(datadir, f'model{target_seed}_top{opt_topk[i]}_sae_acts.npy'))
        target_latents = torch.tensor(target_latents[subset:], device=device)

        # results = pairwise_corr(source_neurons, target_neurons)
        # base_score = torch.mean(torch.max(results, 1)[0]).cpu().numpy()
        metric = Linear()
        # scores = metric.fit_kfold_nnls(x=source_neurons.cpu().to(torch.float), y=target_neurons.cpu().to(torch.float))
        scores = metric.fit_kfold_ridge(x=source_neurons.cpu().to(torch.float), y=target_neurons.cpu().to(torch.float))
        base_score = np.array(scores).mean()

        # base_score = SoftMatching(source_neurons.cpu().numpy(), target_neurons.cpu().numpy())
        # metric = SoftMatch()
        # base_score = torch.tensor(metric.fit_kfold_score(source_neurons, target_neurons)).mean()

        # #   NOTE: need to remove activation vectors which contain all-zeros to avoid NaN in RSA.
        # #         I suppose this happens in response to an all-zero input feature vector.
        # is_invalid_n = torch.logical_or(
        #     torch.all(source_neurons == 0, dim=1),
        #     torch.all(target_neurons == 0, dim=1),
        # )
        # is_invalid_l = torch.logical_or(
        #     torch.all(source_latents == 0, dim=1),
        #     torch.all(target_latents == 0, dim=1),
        # )
        # is_invalid = torch.logical_or(is_invalid_n, is_invalid_l)
        # valid_idx = torch.nonzero(torch.logical_not(is_invalid))[:, 0]
        # source_neurons, target_neurons = source_neurons[valid_idx].cpu(), target_neurons[valid_idx].cpu()
        # source_latents, target_latents = source_latents[valid_idx].cpu(), target_latents[valid_idx].cpu()
        # base_score = RSA(source_neurons, target_neurons)[0]

        # results = pairwise_corr(source_latents, target_neurons)
        # sae_base_score = torch.mean(torch.max(results, 1)[0]).cpu().numpy()
        scores = metric.fit_kfold_ridge(x=source_latents.cpu().to(torch.float), y=target_neurons.cpu().to(torch.float))
        # scores = metric.fit_kfold_nnls(x=source_latents.cpu().to(torch.float), y=target_neurons.cpu().to(torch.float))
        sae_base_score = np.array(scores).mean()
        # sae_base_score = RSA(source_latents, target_neurons)[0]
        # sae_base_score = torch.tensor(metric.fit_kfold_score(source_latents, target_neurons)).mean()

        # results = pairwise_corr(source_latents, target_latents)
        # sae_sae_score = torch.mean(torch.max(results, 1)[0]).cpu().numpy()
        scores = metric.fit_kfold_ridge(x=source_latents.cpu().to(torch.float), y=target_latents.cpu().to(torch.float))
        # scores = metric.fit_kfold_nnls(x=source_latents.cpu().to(torch.float), y=target_latents.cpu().to(torch.float))
        sae_sae_score = np.array(scores).mean()
        # sae_sae_score = RSA(source_latents, target_latents)[0]
        # sae_sae_score = torch.tensor(metric.fit_kfold_score(source_latents, target_latents)).mean()

        ax.bar(i, base_score, width, color='c', label=legend_labels[0])
        ax.bar(i + width, sae_base_score, width, color='m', label=legend_labels[1])
        ax.bar(i + (2*width), sae_sae_score, width, color='y', label=legend_labels[2])

        if i == 0:
            legend_labels = [None, None, None]

    ax.set_ylabel('Pearson r (5-fold Mean)')
    # ax.set_ylabel('Pairwise Mean Similarity')
    # ax.set_ylabel('RSA (Kendall\'s tau)')
    # ax.set_ylabel('Soft Match Corr')
    # ax.set_ylabel('NNLS Pearson r')
    ax.set_title('Toy Model (powerlaw) Alignment of Base Neurons vs SAE Latents')
    ax.set_xticks(np.arange(len(neurons)) + width, x_labels)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)

    plt.savefig('/home/alongon/figures/superposition_alignment/toy_model_powerlaw_ridge_small.png', bbox_inches='tight', dpi=300)


def plot_toy_weight_histos(basedir):
    source_seed = 0
    target_seed = 1
    num_feats = 64
    neurons = [8, 16, 32]

    fig, axs = plt.subplots(3, len(neurons), sharex='row', sharey='row', layout='constrained')
    for i in range(len(neurons)):
        neuron = neurons[i]
        w_1 = torch.load(os.path.join(basedir, f'{num_feats}feats_{neuron}neurons_bias', f'model{source_seed}_weights.pth'))['W']
        w_2 = torch.load(os.path.join(basedir, f'{num_feats}feats_{neuron}neurons_bias', f'model{target_seed}_weights.pth'))['W']

        feat_overlaps = w_1.norm(dim=1) * w_2.norm(dim=1)
        shared_feat_idx = torch.nonzero(feat_overlaps >= 1)[:, 0]
        axs[0, i].hist(feat_overlaps.cpu().numpy(), bins=20)

        # axs[1, i].hist(torch.diagonal(F.normalize(w_1) @ F.normalize(w_2).T)[torch.nonzero(feat_overlaps >= 1)[:, 0]].cpu().numpy(), bins=20)
        results = pairwise_corr(w_1[shared_feat_idx], w_2[shared_feat_idx])
        weight_sim = torch.max(results, 1)[0].cpu().numpy()
        axs[1, i].hist(weight_sim, bins=20)

        #   TODO:  axs[2, i] will contain a histogram of correlations between recon errors for the shared features.
        #          Do just a scatter plot for now to see if there is a trend.
        mse_1 = np.load(os.path.join(basedir, f'{num_feats}feats_{neuron}neurons_bias', f'model{source_seed}_recon_errors.npy'))[-10000:, shared_feat_idx.cpu()]
        mse_2 = np.load(os.path.join(basedir, f'{num_feats}feats_{neuron}neurons_bias', f'model{target_seed}_recon_errors.npy'))[-10000:, shared_feat_idx.cpu()]
        axs[2, i].scatter(np.mean(mse_1, axis=1), np.mean(mse_2, axis=1), s=0.1)

        axs[2, i].set_xlabel(f'Model 1 MSE\n{num_feats} features, {neuron} neurons')
        axs[2, i].set_ylabel('Model 2 MSE')

        if i == (len(neurons) // 2):
            axs[0, i].set_title('Feature Overlap\nw_1 Norm * w_2 Norm')
            axs[1, i].set_title('Different Feature Superposition Arrangements\nPairwise Sim of Neuron Weights for Shared Feature')
            axs[2, i].set_title('MSE of Shared Features Across Models')

        axs[1, i].set_xlim(0, 1)

    plt.savefig('/home/alongon/figures/superposition_alignment/toy_model_overlap_histos.png', bbox_inches='tight', dpi=300)


def plot_ds_size(source_neurons, source_latents, target_neurons):
    width = 0.25
    fig, ax = plt.subplots(layout='constrained')
    x_labels = ()
    legend_labels = ['Neuron->Neuron', 'SAE->Neuron']
    subset_percs = np.arange(25, 125, 25)
    for i in range(subset_percs.shape[0]):
        p = subset_percs[i]
        x_labels += (f'{p}%',)
        subset_idx = int(source_neurons.shape[0] * (p / 100))

        metric = Linear()
        scores = metric.fit_kfold_ridge(
            x=source_neurons[:subset_idx].cpu().to(torch.float),
            y=target_neurons[:subset_idx].cpu().to(torch.float),
        )
        neuron_score = np.array(scores).mean()

        metric = Linear()
        scores = metric.fit_kfold_ridge(
            x=source_latents[:subset_idx].cpu().to(torch.float),
            y=target_neurons[:subset_idx].cpu().to(torch.float),
        )
        latent_score = np.array(scores).mean()

        ax.bar(i, neuron_score, width, color='c', label=legend_labels[0])
        bar = ax.bar(i + width, latent_score, width, color='m', label=legend_labels[1])
        ax.bar_label(bar, labels=['+{:0.2f}%'.format(((latent_score / neuron_score) - 1) * 100)], padding=3)

        if i == 0:
            legend_labels = [None, None, None]

    ax.set_ylabel('Pearson r (5-fold Mean)')
    ax.set_xlabel(r'% of ImageNet Val Dataset')
    ax.set_title('ResNet18->ResNet50 Alignments Across Dataset Sizes')
    ax.set_xticks(np.arange(len(subset_percs)) + (width/2), x_labels)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)

    plt.savefig('/home/alongon/figures/superposition_alignment/dataset_size_align.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    plot_toy_alignments('/home/alongon/data/toy_data')

    # plot_toy_weight_histos('/home/alongon/data/toy_data')

    # source_neurons = torch.clamp(torch.tensor(np.load('/home/alongon/data/imagenet_acts/resnet18/raw_neurons/layer4.1_center_patch_valid.npy')), min=0, max=None)
    # source_latents = torch.clamp(torch.tensor(np.load('/home/alongon/data/imagenet_acts/resnet18/sae_latents/layer4.1/top16_5exp_sae_weights_100ep.npy')), min=0, max=None)
    # target_neurons = torch.clamp(torch.tensor(np.load('/home/alongon/data/imagenet_acts/resnet50/raw_neurons/layer4.2_center_patch_valid.npy')), min=0, max=None)

    # plot_ds_size(source_neurons, source_latents, target_neurons)