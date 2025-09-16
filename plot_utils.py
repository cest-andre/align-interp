import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import sem

from utils import RSA, SemiMatching, SoftMatching, RidgeRegression, pairwise_corr
from voxel_utils import voxel_dnn_align, extract_coco_ids


def plot_brain_align_hierarchy(nsd_root, sae_acts_root, splits_path, savedir, subj1=1, subj2=2):
    # subj1_nc = np.load(os.path.join(basedir, 'nsd/noise_ceilings/noise_ceilings_NSD.npy'), allow_pickle=True)[()][subj1].mean()
    # subj2_nc = np.load(os.path.join(basedir, 'nsd/noise_ceilings/noise_ceilings_NSD.npy'), allow_pickle=True)[()][subj2].mean()
    regions = ['V1v', 'V2v', 'V3v', 'v4']
    # regions = ['ventral_visual']
    sae_name = 'vanilla_2lambda_4.0exp_sae_weights_2000ep'

    _, subj1_idx = extract_coco_ids(splits_path, nsd_root, subj1, split='test')
    _, subj2_idx = extract_coco_ids(splits_path, nsd_root, subj2, split='test')
    
    width = 0.25
    fig, ax = plt.subplots(layout='constrained')
    x_labels = ()
    legend_labels = ['Voxel->Voxel', 'SAE->SAE']
    for i in range(len(regions)):
        region = regions[i]
        x_labels += (f'{region}',)

        subj1_voxels = torch.tensor(
            np.load(os.path.join(nsd_root, f'{region}_data', f'subj{subj1}.npy'))[subj1_idx]
        )
        subj1_sae_acts = torch.tensor(
            np.load(os.path.join(sae_acts_root, region, f'subj{subj1}', 'voxel_sae_acts', f'{sae_name}.npy'))[subj1_idx]
        )
        subj2_voxels = torch.tensor(
            np.load(os.path.join(nsd_root, f'{region}_data', f'subj{subj2}.npy'))[subj2_idx]
        )
        subj2_sae_acts = torch.tensor(
            np.load(os.path.join(sae_acts_root, region, f'subj{subj2}', 'voxel_sae_acts', f'{sae_name}.npy'))[subj2_idx]
        )

        base_score = RSA(subj1_voxels, subj2_voxels)
        sae_score = RSA(subj1_sae_acts, subj2_sae_acts)
        
        ax.bar(i, base_score, width, color='c', label=legend_labels[0])
        bar = ax.bar(i + width, sae_score, width, color='m', label=legend_labels[1])

        ax.bar_label(bar, labels=['{:+0.2f}%'.format(((sae_score / base_score) - 1) * 100)], padding=3)

        if i == 0:
            legend_labels = [None, None]

    # ax.set_ylabel('Noise-normalized Ridge Pearson r')
    # ax.set_ylabel('Pairwise Mean Similarity')
    ax.set_ylabel('RSA (Pearson r)')
    # ax.set_ylabel('Soft Match Corr')
    # ax.set_ylabel('NNLS Pearson r')
    ax.set_title(f'NSD Subj{subj1}->Subj{subj2} Voxel Alignment')
    ax.set_xticks(np.arange(len(regions)) + (0.5*width), x_labels)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)

    plt.savefig(os.path.join(savedir, f'nsd_subj{subj1}-subj{subj2}_rsa.png'), bbox_inches='tight', dpi=300)


def plot_dnn_brain_alignment(basedir, savedir, splits_path, nsd_path, coco_dir, nsd_root, subj_id=1):
    # # source_model = 'resnet50'
    # # layer = 'layer4.2.bn2'
    # source_model = 'vit_b_16'
    # layer = 'encoder.layers.encoder_layer_11'
    # dnn_opt_topk = {'k': 64, 'epoch': 300}
    # voxel_opt_topk = {'k': 256, 'epoch': 1000}

    # noise_ceiling = np.load(os.path.join(basedir, 'nsd/noise_ceilings/noise_ceilings_NSD.npy'), allow_pickle=True)[()][subj_id].mean()
    noise_ceiling = 1

    models = ['resnet50', 'vit_b_16']
    layers = ['layer4.2', 'encoder.layers.encoder_layer_11']
    dnn_opt_topk_list = [{'k': 82, 'epoch': 300, 'exp': 4}, {'k': 64, 'epoch': 300, 'exp': 4}]
    width = 0.25
    fig, ax = plt.subplots(layout='constrained')
    x_labels = ()
    legend_labels = ['Neuron->Voxel', 'SAE->Voxel', 'SAE->Voxel SAE', 'Rand SAE->Voxel SAE', 'Rand SAE->Voxel']
    for i in range(len(models)):
        model = models[i]
        layer = layers[i]
        dnn_opt_topk = dnn_opt_topk_list[i]

        x_labels += (f'{model}, {layer},\nTopK={dnn_opt_topk["k"]}, expansion={dnn_opt_topk["exp"]}x',)

        source_neurons = torch.tensor(
            np.load(os.path.join(basedir, 'coco_acts', model, 'raw_neurons', f'{layer}.npy'))
        )
        # voxel_acts = torch.tensor(
        #     np.load(os.path.join(basedir, f'nsd/subj{subj_id}', f'train_voxels.npy'))
        # )
        voxel_acts = torch.tensor(
            np.load(os.path.join(nsd_root, f'subj{subj_id}.npy'))
        )
        source_sae_acts = torch.tensor(
            np.load(os.path.join(basedir, 'coco_acts', model, 'sae_latents', layer, f'top{dnn_opt_topk["k"]}_aux_{dnn_opt_topk["exp"]}exp_sae_weights_{dnn_opt_topk["epoch"]}ep.npy'))
        )
        # voxel_sae_acts = torch.tensor(
        #     np.load(os.path.join(basedir, 'nsd/subj1/voxel_sae_acts', f'top{voxel_opt_topk["k"]}_aux_2exp_sae_weights_{voxel_opt_topk["epoch"]}ep.npy'))
        # )
        source_rand_sae_acts = torch.tensor(
            np.load(os.path.join(basedir, 'coco_acts', model, 'sae_latents', layer, f'top{dnn_opt_topk["k"]}_aux_{dnn_opt_topk["exp"]}exp_sae_weights_randinit.npy'))
        )

        base_scores = voxel_dnn_align(splits_path, nsd_path, coco_dir, subj_id, nsd_root, source_neurons, voxel_acts) / noise_ceiling
        base_score = base_scores.mean()
        base_error = sem(base_scores)

        sae_base_scores = voxel_dnn_align(splits_path, nsd_path, coco_dir, subj_id, nsd_root, source_sae_acts, voxel_acts) / noise_ceiling
        sae_base_score = sae_base_scores.mean()
        sae_base_error = sem(sae_base_scores)

        # sae_sae_scores = voxel_dnn_align(splits_path, nsd_path, coco_dir, subj_id, source_sae_acts, voxel_sae_acts) / noise_ceiling
        # sae_sae_score = sae_sae_scores.mean()
        # sae_sae_error = sem(sae_sae_scores)

        rand_sae_scores = voxel_dnn_align(splits_path, nsd_path, coco_dir, subj_id, nsd_root, source_rand_sae_acts, voxel_acts) / noise_ceiling
        rand_sae_score = rand_sae_scores.mean()
        rand_sae_error = sem(rand_sae_scores)
        
        ax.bar(i, base_score, width, yerr=base_error, color='c', label=legend_labels[0])
        bar = ax.bar(i + width, sae_base_score, width, yerr=sae_base_error, color='m', label=legend_labels[1])
        # ax.bar(2*width, sae_sae_score, width, yerr=sae_sae_error, color='y', label=legend_labels[2])
        ax.bar(i + (2*width), rand_sae_score, width, yerr=rand_sae_error, color='0.4', label=legend_labels[4])

        ax.bar_label(bar, labels=['{:+0.2f}%'.format(((sae_base_score / base_score) - 1) * 100)], padding=3)

        if i == 0:
            legend_labels = [None, None, None, None, None]

    # ax.set_ylabel('Noise-normalized Ridge Pearson r')
    # ax.set_ylabel('Pairwise Mean Similarity')
    ax.set_ylabel('RSA (Pearson r)')
    # ax.set_ylabel('Soft Match Corr')
    # ax.set_ylabel('NNLS Pearson r')
    ax.set_title(f'DNN->NSD Subject {subj_id} hV4 Voxel Alignment')
    ax.set_xticks(np.arange(len(models)) + (1*width), x_labels)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)

    plt.savefig(os.path.join(savedir, f'dnn-nsd_subj{subj_id}_hV4_rsa.png'), bbox_inches='tight', dpi=300)


def plot_dnn_seed_alignment(basedir, savedir):
    source_model = 'vit_b_16_seed1'
    target_model = 'vit_b_16_seed2'
    layers = ['encoder.layers.encoder_layer_5.mlp', 'encoder.layers.encoder_layer_11.mlp']
    opt_topk = [{'k': 153, 'epoch': '300ep', 'exp': 2}, {'k': 153, 'epoch': '300ep', 'exp': 2}]

    # layers = ['layer2.3.bn2', 'layer4.2.bn2']
    # opt_topk = [{'k': 8, 'epoch': '100ep', 'exp': 2}, {'k': 32, 'epoch': '300ep', 'exp': 4}]

    width = 0.15  #  use 0.15 for ridge
    fig, ax = plt.subplots(layout='constrained')
    x_labels = ()
    legend_labels = ['Neuron->Neuron', 'SAE->Neuron', 'SAE->SAE', 'Rand SAE->SAE', 'Rand SAE->Neuron']
    for i in range(len(layers)):
        x_labels += (f'{layers[i]},\nTopK={opt_topk[i]["k"]}, expansion={opt_topk[i]["exp"]}x',)

        source_neurons = torch.tensor(
            np.load(os.path.join(basedir, source_model, 'raw_neurons', f'{layers[i]}_center_patch_valid.npy'))
        )
        target_neurons = torch.tensor(
            np.load(os.path.join(basedir, target_model, 'raw_neurons', f'{layers[i]}_center_patch_valid.npy'))
        )
        source_latents = torch.tensor(
            np.load(os.path.join(basedir, source_model, 'sae_latents', layers[i], f'top{opt_topk[i]["k"]}_aux_{opt_topk[i]["exp"]}exp_sae_weights_{opt_topk[i]["epoch"]}.npy'))
        )
        target_latents = torch.tensor(
            np.load(os.path.join(basedir, target_model, 'sae_latents', layers[i], f'top{opt_topk[i]["k"]}_aux_{opt_topk[i]["exp"]}exp_sae_weights_{opt_topk[i]["epoch"]}.npy'))
        )
        source_rand_latents = torch.tensor(
            np.load(os.path.join(basedir, source_model, 'sae_latents', layers[i], f'top{opt_topk[i]["k"]}_aux_{opt_topk[i]["exp"]}exp_sae_weights_randinit.npy'))
        )
        # target_rand_latents = torch.tensor(
        #     np.load(os.path.join(basedir, target_model, 'sae_latents', layers[i], f'top{opt_topk[i]["k"]}_aux_{opt_topk[i]["exp"]}exp_sae_weights_randinit.npy'))
        # )
        
        print('base score')
        # base_scores = np.array(SemiMatching(source_neurons.numpy().astype(np.float64), target_neurons.numpy().astype(np.float64)))
        # base_scores = np.array(SoftMatching(source_neurons.numpy().astype(np.float64), target_neurons.numpy().astype(np.float64)))
        base_scores = np.array(RidgeRegression(source_neurons.numpy().astype(np.float64), target_neurons.numpy().astype(np.float64)))
        base_score = base_scores.mean()
        base_error = sem(base_scores)

        print('sae base score')
        sae_base_scores = np.array(RidgeRegression(source_latents.numpy().astype(np.float64), target_neurons.numpy().astype(np.float64)))
        sae_base_score = sae_base_scores.mean()
        sae_base_error = sem(sae_base_scores)
        
        print('sae sae score')
        # sae_sae_scores = np.array(SemiMatching(source_latents.numpy().astype(np.float64), target_latents.numpy().astype(np.float64)))
        # sae_sae_scores = np.array(SoftMatching(source_latents.numpy().astype(np.float64), target_latents.numpy().astype(np.float64)))
        sae_sae_scores = np.array(RidgeRegression(source_latents.numpy().astype(np.float64), target_latents.numpy().astype(np.float64)))
        sae_sae_score = sae_sae_scores.mean()
        sae_sae_error = sem(sae_sae_scores)

        print('rand sae score')
        # rand_sae_scores = np.array(SemiMatching(source_rand_latents.numpy().astype(np.float64), target_latents.numpy().astype(np.float64)))
        # rand_sae_scores = np.array(SoftMatching(source_rand_latents.numpy().astype(np.float64), target_latents.numpy().astype(np.float64)))
        rand_sae_scores = np.array(RidgeRegression(source_rand_latents.numpy().astype(np.float64), target_neurons.numpy().astype(np.float64)))
        rand_sae_score = rand_sae_scores.mean()
        rand_sae_error = sem(rand_sae_scores)
        
        ax.bar(i, base_score, width, yerr=base_error, color='c', label=legend_labels[0])
        bar = ax.bar(i + (1*width), sae_base_score, width, yerr=sae_base_error, color='m', label=legend_labels[1])
        ax.bar(i + (2*width), sae_sae_score, width, yerr=sae_sae_error, color='y', label=legend_labels[2])
        ax.bar(i + (3*width), rand_sae_score, width, yerr=rand_sae_error, color='0.4', label=legend_labels[4])

        ax.bar_label(bar, labels=['{:+0.2f}%'.format(((sae_base_score / base_score) - 1) * 100)], padding=3)

        if i == 0:
            legend_labels = [None, None, None, None, None]

    ax.set_ylabel('Ridge Pearson r')
    # ax.set_ylabel('Semi Match Corr')
    # ax.set_ylabel('RSA (Kendall\'s tau)')
    # ax.set_ylabel('Soft Match Corr')
    # ax.set_ylabel('NNLS Pearson r')
    ax.set_title(f'{source_model}->{target_model} Alignment of Base Neurons vs SAE Latents')
    ax.set_xticks(np.arange(len(layers)) + (1.5*width), x_labels)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 1)

    plt.savefig(os.path.join(savedir, f'{source_model}-{target_model}_mlp.5-mlp.11_ridge.png'), bbox_inches='tight', dpi=300)


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
    device = 'cpu'

    width = 0.25
    fig, ax = plt.subplots(layout='constrained')
    x_labels = ()
    legend_labels = ['Neuron->Neuron', 'SAE->Neuron', 'SAE->SAE', 'Rand SAE->SAE', 'Rand SAE->Neuron']
    for i in range(len(neurons)):
        datadir = os.path.join(basedir, f'{num_feats}feats_{neurons[i]}neurons_powerlaw_final')
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

        source_rand_latents = np.load(os.path.join(datadir, f'model{source_seed}_top{opt_topk[i]}_randinit_sae_acts.npy'))
        source_rand_latents = torch.tensor(source_rand_latents[subset:], device=device)

        # base_scores = np.array(SemiMatching(source_neurons.numpy().astype(np.float64), target_neurons.numpy().astype(np.float64)))
        base_scores = np.array(SoftMatching(source_neurons.numpy().astype(np.float64), target_neurons.numpy().astype(np.float64)))
        # base_scores = np.array(RidgeRegression(source_neurons.numpy().astype(np.float64), target_neurons.numpy().astype(np.float64)))
        base_score = base_scores.mean()
        base_error = sem(base_scores)

        # # sae_base_scores = np.array(SemiMatching(source_latents.numpy().astype(np.float64), target_neurons.numpy().astype(np.float64)))
        # # sae_base_scores = np.array(SoftMatching(source_latents.numpy().astype(np.float64), target_neurons.numpy().astype(np.float64)))
        # sae_base_scores = np.array(RidgeRegression(source_latents.numpy().astype(np.float64), target_neurons.numpy().astype(np.float64)))
        # sae_base_score = sae_base_scores.mean()
        # sae_base_error = sem(base_scores)

        # sae_sae_scores = np.array(SemiMatching(source_latents.numpy().astype(np.float64), target_latents.numpy().astype(np.float64)))
        sae_sae_scores = np.array(SoftMatching(source_latents.numpy().astype(np.float64), target_latents.numpy().astype(np.float64)))
        # sae_sae_scores = np.array(RidgeRegression(source_latents.numpy().astype(np.float64), target_latents.numpy().astype(np.float64)))
        sae_sae_score = sae_sae_scores.mean()
        sae_sae_error = sem(sae_sae_scores)

        # rand_sae_scores = np.array(SemiMatching(source_rand_latents.numpy().astype(np.float64), target_latents.numpy().astype(np.float64)))
        rand_sae_scores = np.array(SoftMatching(source_rand_latents.numpy().astype(np.float64), target_latents.numpy().astype(np.float64)))
        # rand_sae_scores = np.array(RidgeRegression(source_rand_latents.numpy().astype(np.float64), target_neurons.numpy().astype(np.float64)))
        rand_sae_score = rand_sae_scores.mean()
        rand_sae_error = sem(rand_sae_scores)

        ax.bar(i, base_score, width, yerr=base_error, color='c', label=legend_labels[0])
        # ax.bar(i + (1*width), sae_base_score, width, yerr=sae_base_error, color='m', label=legend_labels[1])
        bar = ax.bar(i + (1*width), sae_sae_score, width, yerr=sae_sae_error, color='y', label=legend_labels[2])
        ax.bar(i + (2*width), rand_sae_score, width, yerr=rand_sae_error, color='0.4', label=legend_labels[3])

        ax.bar_label(bar, labels=['{:+0.2f}%'.format(((sae_sae_score / base_score) - 1) * 100)], padding=3)

        if i == 0:
            legend_labels = [None, None, None, None, None]

    # ax.set_ylabel('Ridge Pearson r')
    # ax.set_ylabel('Semi Match Corr')
    # ax.set_ylabel('RSA (Kendall\'s tau)')
    ax.set_ylabel('Soft Match Corr')
    # ax.set_ylabel('NNLS Pearson r')
    ax.set_title('Toy Model Alignment of Base Neurons vs SAE Latents')
    ax.set_xticks(np.arange(len(neurons)) + (1*width), x_labels)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)

    plt.savefig('/home/alongon/figures/superposition_alignment/toy_model_powerlaw_final_softmatch.png', bbox_inches='tight', dpi=300)


def plot_toy_weight_histos(basedir):
    plt.rcParams['text.usetex'] = True
    source_seed = 0
    target_seed = 1
    num_feats = 64
    neurons = [8, 16, 32]

    fig, axs = plt.subplots(2, len(neurons), sharex='row', sharey='row', layout='constrained')
    for i in range(len(neurons)):
        neuron = neurons[i]

        w_1 = torch.load(os.path.join(basedir, f'{num_feats}feats_{neuron}neurons_powerlaw_final', f'model{source_seed}_weights.pth'))['W']
        w_2 = torch.load(os.path.join(basedir, f'{num_feats}feats_{neuron}neurons_powerlaw_final', f'model{target_seed}_weights.pth'))['W']

        feat_overlaps = w_1.norm(dim=1) * w_2.norm(dim=1)
        shared_feat_idx = torch.nonzero(feat_overlaps >= 1)[:, 0]
        axs[0, i].hist(feat_overlaps.cpu().numpy(), bins=20)

        axs[0, i].set_xlabel(f'Multiplied Norms', fontsize=12)

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

        if i == 0:
            axs[0, i].set_ylabel(f'\# of Features', fontsize=12)
            axs[1, i].set_ylabel(f'\# of Neurons', fontsize=12)
        elif i == (len(neurons) // 2):
            # 
            axs[0, i].set_title('Represented Feature Overlap: ' + r'$\|\mathbf{W}^{(1)}_i\|_2 \times \|\mathbf{W}^{(2)}_i\|_2$', fontsize=16)
            axs[1, i].set_title('Semi-Match of Neuron Weights amongst Shared Feature', fontsize=16)
            # axs[2, i].set_title('MSE of Shared Features Across Models')

        axs[1, i].set_xlim(0, 1)
        axs[1, i].set_xlabel(f'Semi-match score\n{neuron} neurons', fontsize=12)
        
    plt.savefig('/home/alongon/figures/superposition_alignment/toy_model_powerlaw_final_overlap_histos.png', bbox_inches='tight', dpi=300)


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
    plot_brain_align_hierarchy(
        '/mnt/cogsci/KhoslaLab/NSD_Preprocessed/',
        '/home/alongon/data/nsd',
        '/mnt/cogsci/NSD_preprocessed_datasets_shreya/ventral_visual_data/ventral_visual_data_splits_1257.pickle',
        '/home/alongon/figures/superposition_alignment',
        subj1=1, subj2=2
    )

    # nsd_subj_ids = [1,2,5,7]
    # for sub_id in nsd_subj_ids:
    # plot_dnn_brain_alignment(
    #     '/home/alongon/data',
    #     '/home/alongon/figures/superposition_alignment',
    #     '/mnt/cogsci/NSD_preprocessed_datasets_shreya/ventral_visual_data/ventral_visual_data_splits_1257.pickle',
    #     '/mnt/cogsci/NSD_preprocessed_datasets_shreya/ventral_visual_data/ventral_visual_data_1257_filtered.pickle',
    #     '/mnt/cogsci/KhoslaLab/NSD_Preprocessed/images/',
    #     '/mnt/cogsci/KhoslaLab/NSD_Preprocessed/ventral_visual_data',
    #     subj_id=1
    # )

    # plot_dnn_seed_alignment('/home/alongon/data/imagenet_acts', '/home/alongon/figures/superposition_alignment')

    # plot_toy_alignments('/home/alongon/data/toy_data')

    # plot_toy_weight_histos('/home/alongon/data/toy_data')

    # source_neurons = torch.clamp(torch.tensor(np.load('/home/alongon/data/imagenet_acts/resnet18/raw_neurons/layer4.1_center_patch_valid.npy')), min=0, max=None)
    # source_latents = torch.clamp(torch.tensor(np.load('/home/alongon/data/imagenet_acts/resnet18/sae_latents/layer4.1/top16_5exp_sae_weights_100ep.npy')), min=0, max=None)
    # target_neurons = torch.clamp(torch.tensor(np.load('/home/alongon/data/imagenet_acts/resnet50/raw_neurons/layer4.2_center_patch_valid.npy')), min=0, max=None)

    # plot_ds_size(source_neurons, source_latents, target_neurons)