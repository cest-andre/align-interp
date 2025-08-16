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
from voxel_utils import voxel_dnn_align


def plot_dnn_brain_alignment(basedir, savedir, splits_path, nsd_path, coco_dir, subj_id=1):
    source_model = 'resnet50'
    layer = 'layer4.2'
    dnn_opt_topk = {'k': 82, 'epoch': 300}
    voxel_opt_topk = {'k': 256, 'epoch': 1000}

    width = 0.25
    fig, ax = plt.subplots(layout='constrained')
    x_labels = ()
    legend_labels = ['Neuron->Voxel', 'DNN SAE->Voxel', 'DNN SAE->Voxel SAE']
    # for i in range(len(layers)):
    x_labels += (f'{layer}, TopK={dnn_opt_topk["k"]}',)

    source_neurons = torch.tensor(
        np.load(os.path.join(basedir, 'coco_acts', source_model, 'raw_neurons', f'{layer}.npy'))
    )
    voxel_acts = torch.tensor(
        np.load(os.path.join(basedir, 'nsd/subj1', f'train_voxels.npy'))
    )
    source_sae_acts = torch.tensor(
        np.load(os.path.join(basedir, 'coco_acts', source_model, 'sae_latents', layer, f'top{dnn_opt_topk["k"]}_aux_4exp_sae_weights_{dnn_opt_topk["epoch"]}ep.npy'))
    )
    voxel_sae_acts = torch.tensor(
        np.load(os.path.join(basedir, 'nsd/subj1/voxel_sae_acts', f'top{voxel_opt_topk["k"]}_aux_2exp_sae_weights_{voxel_opt_topk["epoch"]}ep.npy'))
    )

    base_score = voxel_dnn_align(splits_path, nsd_path, coco_dir, subj_id, source_neurons, voxel_acts)
    sae_base_score = voxel_dnn_align(splits_path, nsd_path, coco_dir, subj_id, source_sae_acts, voxel_acts)
    sae_sae_score = voxel_dnn_align(splits_path, nsd_path, coco_dir, subj_id, source_sae_acts, voxel_sae_acts)
    
    ax.bar(0, base_score, width, color='c', label=legend_labels[0])
    ax.bar(width, sae_base_score, width, color='m', label=legend_labels[1])
    bar = ax.bar(2*width, sae_sae_score, width, color='y', label=legend_labels[2])

    ax.bar_label(bar, labels=['{:+0.2f}%'.format(((sae_sae_score / base_score) - 1) * 100)], padding=3)

    # if i == 0:
    #     legend_labels = [None, None]#, None]

    # ax.set_ylabel('Pearson r (5-fold Mean)')
    ax.set_ylabel('Pairwise Mean Similarity')
    # ax.set_ylabel('RSA (Kendall\'s tau)')
    # ax.set_ylabel('Soft Match Corr')
    # ax.set_ylabel('NNLS Pearson r')
    ax.set_title(f'{source_model}->NSD Subj1 VSS Alignment of Base Neurons vs SAE Latents')
    ax.set_xticks(np.arange(1) + width, x_labels)
    ax.legend(loc='upper left', ncols=1)
    ax.set_ylim(0, 1)

    plt.savefig(os.path.join(savedir, f'{source_model}-nsd_subj1_vss_pairwise_top82_aux_4x_300ep.png'), bbox_inches='tight', dpi=300)


def plot_dnn_seed_alignment(basedir, savedir):
    source_model = 'resnet50_seed1'
    target_model = 'resnet50_seed2'
    # layers = ['layer2', 'layer4']
    # opt_topk = [{'k': 64, 'epoch': 300}, {'k': 256, 'epoch': 300}]

    layers = ['layer2.3.bn2']
    opt_topk = [{'k': 64, 'epoch': 300}]

    width = 0.25
    fig, ax = plt.subplots(layout='constrained')
    x_labels = ()
    legend_labels = ['Neuron->Neuron', 'SAE->Neuron', 'SAE->SAE']
    for i in range(len(layers)):
        x_labels += (f'{layers[i]}, TopK={opt_topk[i]["k"]}',)

        source_neurons = torch.tensor(
            np.load(os.path.join(basedir, source_model, 'raw_neurons', f'{layers[i]}_center_patch_valid.npy'))
        )
        target_neurons = torch.tensor(
            np.load(os.path.join(basedir, target_model, 'raw_neurons', f'{layers[i]}_center_patch_valid.npy'))
        )
        source_latents = torch.tensor(
            np.load(os.path.join(basedir, source_model, 'sae_latents', layers[i], f'top{opt_topk[i]["k"]}_aux_4exp_sae_weights_{opt_topk[i]["epoch"]}ep.npy'))
        )
        target_latents = torch.tensor(
            np.load(os.path.join(basedir, target_model, 'sae_latents', layers[i], f'top{opt_topk[i]["k"]}_aux_4exp_sae_weights_{opt_topk[i]["epoch"]}ep.npy'))
        )

        metric = Linear()
        # metric = SoftMatch()
        
        print('base score')
        # results = pairwise_corr(source_neurons, target_neurons)
        # base_score = torch.mean(torch.max(results, 0)[0]).cpu().numpy()
        # scores = metric.fit_kfold_nnls(x=source_neurons.to(torch.float), y=target_neurons.to(torch.float))
        scores = metric.fit_kfold_ridge(x=source_neurons.to(torch.float), y=target_neurons.to(torch.float))
        base_score = np.array(scores).mean()
        # base_score = torch.tensor(metric.fit_kfold_score(source_neurons, target_neurons)).mean()
        # base_score = RSA(source_neurons, target_neurons)[0]

        print('sae base score')
        # results = pairwise_corr(source_latents, target_neurons)
        # sae_base_score = torch.mean(torch.max(results, 0)[0]).cpu().numpy()
        # scores = metric.fit_kfold_nnls(x=source_latents.to(torch.float), y=target_neurons.to(torch.float))
        scores = metric.fit_kfold_ridge(x=source_latents.to(torch.float), y=target_neurons.to(torch.float))
        sae_base_score = np.array(scores).mean()
        # sae_base_score = torch.tensor(metric.fit_kfold_score(source_latents, target_neurons)).mean()
        # sae_base_score = RSA(source_latents, target_neurons)[0]
        
        print('sae sae score')
        # results = pairwise_corr(source_latents, target_latents)
        # sae_sae_score = torch.mean(torch.max(results, 0)[0]).cpu().numpy()
        # scores = metric.fit_kfold_nnls(x=source_latents.to(torch.float), y=target_latents.to(torch.float))
        scores = metric.fit_kfold_ridge(x=source_latents.to(torch.float), y=target_latents.to(torch.float))
        sae_sae_score = np.array(scores).mean()
        # sae_sae_score = torch.tensor(metric.fit_kfold_score(source_latents, target_latents)).mean()
        # sae_sae_score = RSA(source_latents, target_latents)[0]
        
        ax.bar(i, base_score, width, color='c', label=legend_labels[0])
        bar = ax.bar(i + width, sae_base_score, width, color='m', label=legend_labels[1])
        ax.bar(i + (2*width), sae_sae_score, width, color='y', label=legend_labels[2])

        ax.bar_label(bar, labels=['{:+0.2f}%'.format(((sae_base_score / base_score) - 1) * 100)], padding=3)

        if i == 0:
            legend_labels = [None, None, None]

    ax.set_ylabel('Pearson r (5-fold Mean)')
    # ax.set_ylabel('Pairwise Mean Similarity')
    # ax.set_ylabel('RSA (Kendall\'s tau)')
    # ax.set_ylabel('Soft Match Corr')
    # ax.set_ylabel('NNLS Pearson r')
    ax.set_title(f'{source_model}->{target_model} Alignment of Base Neurons vs SAE Latents')
    ax.set_xticks(np.arange(len(layers)) + width, x_labels)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)

    plt.savefig(os.path.join(savedir, f'{source_model}-{target_model}_ridge_2.3.bn2_top64_4x_300ep.png'), bbox_inches='tight', dpi=300)


def plot_toy_alignments(basedir):
    num_feats = 64
    neurons = [8, 16, 32]
    opt_topk = [7, 7, 7]  #  for powerlaw
    # opt_topk = [4, 5, 7]

    train_split = 0.8
    # train_split = 0.99  #  for NNLS
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
        x_labels += (f'{neurons[i]} neurons\nTopK={opt_topk[i]}',)

        source_neurons = np.load(os.path.join(datadir, f'model{source_seed}_raw_neuron_acts.npy'))
        subset = -int(source_neurons.shape[0]*(1-train_split))
        source_neurons = torch.tensor(source_neurons[subset:], device=device)
        target_neurons = np.load(os.path.join(datadir, f'model{target_seed}_raw_neuron_acts.npy'))
        target_neurons = torch.tensor(target_neurons[subset:], device=device)

        source_latents = np.load(os.path.join(datadir, f'model{source_seed}_top{opt_topk[i]}_sae_acts.npy'))
        source_latents = torch.tensor(source_latents[subset:], device=device)
        target_latents = np.load(os.path.join(datadir, f'model{target_seed}_top{opt_topk[i]}_sae_acts.npy'))
        target_latents = torch.tensor(target_latents[subset:], device=device)

        metric = Linear()
        # metric = SoftMatch()

        # results = pairwise_corr(source_neurons, target_neurons)
        # base_score = torch.mean(torch.max(results, 0)[0]).cpu().numpy()
        # scores = metric.fit_kfold_nnls(x=source_neurons.cpu().to(torch.float), y=target_neurons.cpu().to(torch.float))
        scores = metric.fit_kfold_ridge(x=source_neurons.cpu().to(torch.float), y=target_neurons.cpu().to(torch.float))
        base_score = np.array(scores).mean()
        # base_score = torch.tensor(metric.fit_kfold_score(source_neurons, target_neurons)).mean()
        # base_score = RSA(source_neurons, target_neurons)[0]

        # results = pairwise_corr(source_latents, target_neurons)
        # sae_base_score = torch.mean(torch.max(results, 0)[0]).cpu().numpy()
        scores = metric.fit_kfold_ridge(x=source_latents.cpu().to(torch.float), y=target_neurons.cpu().to(torch.float))
        # scores = metric.fit_kfold_nnls(x=source_latents.cpu().to(torch.float), y=target_neurons.cpu().to(torch.float))
        sae_base_score = np.array(scores).mean()
        # sae_base_score = RSA(source_latents, target_neurons)[0]
        # sae_base_score = torch.tensor(metric.fit_kfold_score(source_latents, target_neurons)).mean()

        # results = pairwise_corr(source_latents, target_latents)
        # sae_sae_score = torch.mean(torch.max(results, 0)[0]).cpu().numpy()
        scores = metric.fit_kfold_ridge(x=source_latents.cpu().to(torch.float), y=target_latents.cpu().to(torch.float))
        # scores = metric.fit_kfold_nnls(x=source_latents.cpu().to(torch.float), y=target_latents.cpu().to(torch.float))
        sae_sae_score = np.array(scores).mean()
        # sae_sae_score = RSA(source_latents, target_latents)[0]
        # sae_sae_score = torch.tensor(metric.fit_kfold_score(source_latents, target_latents)).mean()

        ax.bar(i, base_score, width, color='c', label=legend_labels[0])
        bar = ax.bar(i + width, sae_base_score, width, color='m', label=legend_labels[1])
        ax.bar(i + (2*width), sae_sae_score, width, color='y', label=legend_labels[2])

        ax.bar_label(bar, labels=['{:+0.2f}%'.format(((sae_base_score / base_score) - 1) * 100)], padding=3)

        if i == 0:
            legend_labels = [None, None, None]

    ax.set_ylabel('Pearson r (5-fold Mean)')
    # ax.set_ylabel('Pairwise Mean Similarity')
    # ax.set_ylabel('RSA (Kendall\'s tau)')
    # ax.set_ylabel('Soft Match Corr')
    # ax.set_ylabel('NNLS Pearson r')
    ax.set_title('Toy Model (Powerlaw) Alignment of Base Neurons vs SAE Latents')
    ax.set_xticks(np.arange(len(neurons)) + width, x_labels)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)

    plt.savefig('/home/alongon/figures/superposition_alignment/toy_model_powerlaw_ridge.png', bbox_inches='tight', dpi=300)


def plot_toy_weight_histos(basedir):
    source_seed = 0
    target_seed = 1
    num_feats = 64
    neurons = [8, 16, 32]

    fig, axs = plt.subplots(2, len(neurons), sharex='row', sharey='row', layout='constrained')
    for i in range(len(neurons)):
        neuron = neurons[i]

        w_1 = torch.load(os.path.join(basedir, f'{num_feats}feats_{neuron}neurons_powerlaw', f'model{source_seed}_weights.pth'))['W']
        w_2 = torch.load(os.path.join(basedir, f'{num_feats}feats_{neuron}neurons_powerlaw', f'model{target_seed}_weights.pth'))['W']

        feat_overlaps = w_1.norm(dim=1) * w_2.norm(dim=1)
        shared_feat_idx = torch.nonzero(feat_overlaps >= 1)[:, 0]
        axs[0, i].hist(feat_overlaps.cpu().numpy(), bins=20)

        # axs[1, i].hist(torch.diagonal(F.normalize(w_1) @ F.normalize(w_2).T)[torch.nonzero(feat_overlaps >= 1)[:, 0]].cpu().numpy(), bins=20)
        results = pairwise_corr(w_1[shared_feat_idx], w_2[shared_feat_idx])
        weight_sim = torch.max(results, 1)[0].cpu().numpy()
        axs[1, i].hist(weight_sim, bins=20)

        # #   NOTE:  axs[2, i] will contain a scatter of errors between the two models for the shared features.
        # mse_1 = np.load(os.path.join(basedir, f'{num_feats}feats_{neuron}neurons_bias', f'model{source_seed}_recon_errors.npy'))[-10000:, shared_feat_idx.cpu()]
        # mse_2 = np.load(os.path.join(basedir, f'{num_feats}feats_{neuron}neurons_bias', f'model{target_seed}_recon_errors.npy'))[-10000:, shared_feat_idx.cpu()]
        # axs[2, i].scatter(np.mean(mse_1, axis=1), np.mean(mse_2, axis=1), s=0.1)

        # axs[2, i].set_xlabel(f'Model 1 MSE\n{num_feats} features, {neuron} neurons')
        # axs[2, i].set_ylabel('Model 2 MSE')

        if i == (len(neurons) // 2):
            axs[0, i].set_title('Represented Feature Overlap\nSeed 1 Norm * Seed 2 Norm of Feature Weights')
            axs[1, i].set_title('Superposition Arrangement Similarity\nPairwise Sim of Neuron Weights amongst Shared Feature')
            # axs[2, i].set_title('MSE of Shared Features Across Models')

        axs[1, i].set_xlim(0, 1)
        axs[1, i].set_xlabel(f'{neuron} neurons')

    plt.savefig('/home/alongon/figures/superposition_alignment/toy_model_powerlaw_overlap_histos.png', bbox_inches='tight', dpi=300)


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
    plot_dnn_brain_alignment(
        '/home/alongon/data',
        '/home/alongon/figures/superposition_alignment',
        '/mnt/cogsci/NSD_preprocessed_datasets_shreya/ventral_visual_data/ventral_visual_data_splits_1257.pickle',
        '/mnt/cogsci/NSD_preprocessed_datasets_shreya/ventral_visual_data/ventral_visual_data_1257_filtered.pickle',
        '/mnt/cogsci/KhoslaLab/NSD_Preprocessed/images/'
    )

    # plot_dnn_seed_alignment('/home/alongon/data/imagenet_acts', '/home/alongon/figures/superposition_alignment')

    # plot_toy_alignments('/home/alongon/data/toy_data')

    # plot_toy_weight_histos('/home/alongon/data/toy_data')

    # source_neurons = torch.clamp(torch.tensor(np.load('/home/alongon/data/imagenet_acts/resnet18/raw_neurons/layer4.1_center_patch_valid.npy')), min=0, max=None)
    # source_latents = torch.clamp(torch.tensor(np.load('/home/alongon/data/imagenet_acts/resnet18/sae_latents/layer4.1/top16_5exp_sae_weights_100ep.npy')), min=0, max=None)
    # target_neurons = torch.clamp(torch.tensor(np.load('/home/alongon/data/imagenet_acts/resnet50/raw_neurons/layer4.2_center_patch_valid.npy')), min=0, max=None)

    # plot_ds_size(source_neurons, source_latents, target_neurons)