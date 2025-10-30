import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import sem
from scipy.io import loadmat

from utils import RSA, SemiMatching, SoftMatching, RidgeRegression, pairwise_corr
from sae import LN
from voxel_utils import voxel_dnn_align, extract_coco_ids


def face_dprime_histo(acts_dir, vinken_path, savedir):
    model = 'clip_vit-b-32'
    layer = 'visual.transformer.resblocks.11'
    dnn_opt_topk = {'k': 77, 'epoch': 300, 'exp': 2.0}
    fig, ax = plt.subplots(layout='constrained')

    #   Face labels are:  1 - human faces, 2 - monkey faces, 3 - nonfaces
    face_labels = np.squeeze(loadmat(vinken_path)['imsets'])
    # face_labels = face_labels[np.nonzero(face_labels != 2)[0]]
    face_idx = np.nonzero(face_labels == 2)[0]
    nonface_idx = np.nonzero(np.logical_or(face_labels == 1, face_labels == 3))[0]

    base_neurons = np.load(os.path.join(acts_dir, model, 'raw_neurons', f'{layer}.npy'))
    sae_latents = np.load(os.path.join(acts_dir, model, 'sae_latents', layer, f'top{dnn_opt_topk["k"]}_aux_{dnn_opt_topk["exp"]}exp_sae_weights_{dnn_opt_topk["epoch"]}ep.npy'))

    all_acts = [base_neurons, sae_latents]
    all_dprimes = []
    for acts in all_acts:
        # acts = acts[np.concatenate((face_idx, nonface_idx))]
        dead_indices = np.all(acts == 0, axis=0).nonzero()[0]
        act_idx = np.delete(np.arange(acts.shape[1]), dead_indices)
        acts = np.delete(acts, dead_indices, axis=1)
        acts = acts - acts.mean(axis=0)
        acts = acts / np.sqrt(np.sum(acts**2, axis=0))

        face_acts = acts[face_idx]
        face_act_perc = np.sum(face_acts > 0, axis=0) / face_acts.shape[0]

        nonface_acts = acts[nonface_idx]
        nonface_act_perc = np.sum(nonface_acts > 0, axis=0) / nonface_acts.shape[0]

        dprimes = (np.mean(face_acts, axis=0) - np.mean(nonface_acts, axis=0)) / np.sqrt((np.var(face_acts, axis=0) + np.var(nonface_acts, axis=0)) / 2)
        print(face_act_perc[np.argsort(dprimes)[-16:]])
        print(act_idx[np.argsort(dprimes)[-16:]])
        print('---')
        all_dprimes.append(dprimes)

    # exit()
    ax.hist(np.array(all_dprimes[0]), bins=20, color='c', alpha=0.5, label='Base Neurons')
    ax.hist(np.array(all_dprimes[1]), bins=20, color='m', alpha=0.5, label='SAE Latents')

    face_base_neurons = all_dprimes[0][np.nonzero(all_dprimes[0] > 0)[0]]
    face_sae_latents = all_dprimes[1][np.nonzero(all_dprimes[1] > 0)[0]]
    ax.axvline(face_base_neurons.mean(), color='c', linestyle='--')
    ax.axvline(face_sae_latents.mean(), color='m', linestyle='--')
    
    ax.set_xlabel('D Prime')
    ax.set_ylabel('# Units')
    ax.set_title('CLIP ViT resblocks.11 Monkey Face Selectivity of Base Neurons vs. SAE Latents')
    ax.legend(loc='upper right')
    plt.savefig(os.path.join(savedir, 'clip_vit_resblocks.11_monkey_face_dprime.png'), bbox_inches='tight', dpi=300)


#   Perform SAE->voxel linreg alignment.  Obtain learned linreg mapping and investigate:
#   For each SAE latent, measure the magnitude of its voxel weights.  Rank order latents by their mags,
#   print out the latent IDs of top K mags, plot histogram of all mags.  My guess is it would be fairly bimodal.
#   Make sure to factor in IDs and dead latent removal.
#   NOTE:  if I want to visualize the weights as a voxel population code, I'll have to be mindful of how the voxels
#          are normalized prior to lin reg.
#   TODO:  HOW TO HANDLE KFOLDS?
def dnn_brain_linreg_interp(basedir, savedir, coco_dir, nsd_root, subj_id=1):
    model = 'resnet50'
    layer = 'layer4.2'
    dnn_opt_topk = {'k': 41, 'epoch': 300, 'exp': 2}
    fig, ax = plt.subplots(layout='constrained')

    source_neurons = torch.tensor(
        np.load(os.path.join(basedir, 'coco_acts', model, 'raw_neurons', f'{layer}.npy'))
    )
    source_sae_acts = torch.tensor(
        np.load(os.path.join(basedir, 'coco_acts', model, 'sae_latents', layer, f'top{dnn_opt_topk["k"]}_aux_{dnn_opt_topk["exp"]}exp_sae_weights_{dnn_opt_topk["epoch"]}ep.npy'))
    )
    voxel_acts = torch.tensor(
        np.load(os.path.join(nsd_root, 'ventral_visual_data', f'subj{subj_id}.npy'))
    )
    # voxel_acts = torch.tensor(
    #     np.load(os.path.join(basedir, 'nsd', 'ventral_visual', f'subj{subj_id}', '256means_filtered_train_voxels.npy'))
    # )
    print(voxel_acts.shape)

    source_acts = [source_sae_acts]
    all_mags = []
    for acts in source_acts:
        _, coeffs, x_rem_indices = voxel_dnn_align(coco_dir, subj_id, nsd_root, acts, voxel_acts, n_splits=2)
        print(coeffs.shape)
        
        mags = []
        live_mags = []
        live_idx = 0
        for i in range(source_neurons.shape[1]):
            if i in x_rem_indices:
                mags.append(-1)
            else:
                sae_voxel_mag = np.linalg.norm(coeffs[:, live_idx])
                mags.append(sae_voxel_mag)
                live_mags.append(sae_voxel_mag)
                live_idx += 1

        mags = np.array(mags)
        live_mags = np.array(live_mags)
        print(np.argsort(mags)[-32:])
        print(np.argsort(live_mags)[-32:])
        all_mags.append(live_mags)
        np.save('/home/alongon/model_weights/voxel_saes/ventral_visual/subj1/resnet50_layer4.2_top41_2exp_300ep_sae_linreg_coeff.npy', coeffs)
        exit()

    ax.hist(np.array(all_mags[0]), bins=20, color='c', alpha=0.5, label='Base Neurons')
    ax.hist(np.array(all_mags[1]), bins=20, color='m', alpha=0.5, label='SAE Latents')
    ax.set_xlabel('DNN-Voxel LinReg Map Magnitude')
    ax.set_ylabel('# Units')
    ax.legend(loc='upper right')
    plt.savefig(os.path.join(savedir, 'resnet50_layer4.2_nsd_ventral_linreg_mags.png'), bbox_inches='tight', dpi=300)


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


def plot_dnn_brain_alignment(basedir, savedir, coco_dir, nsd_root, subj_id=1):
    # # source_model = 'resnet50'
    # # layer = 'layer4.2.bn2'
    # source_model = 'vit_b_16'
    # layer = 'encoder.layers.encoder_layer_11'
    # dnn_opt_topk = {'k': 64, 'epoch': 300}
    # voxel_opt_topk = {'k': 256, 'epoch': 1000}

    # noise_ceiling = np.load(os.path.join(basedir, 'nsd/noise_ceilings/noise_ceilings_NSD.npy'), allow_pickle=True)[()][subj_id].mean()
    noise_ceiling = 1

    models = ['resnet50', 'vit_b_16']
    layers = ['layer4.2.bn2', 'encoder.layers.encoder_layer_11']
    dnn_opt_topk_list = [{'k': 102, 'epoch': 300, 'exp': 2}, {'k': 64, 'epoch': 300, 'exp': 4}]
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
            np.load(os.path.join(nsd_root, 'ffa_data', f'subj{subj_id}.npy'))
        )
        source_sae_acts = torch.tensor(
            np.load(os.path.join(basedir, 'coco_acts', model, 'sae_latents', layer, f'top{dnn_opt_topk["k"]}_aux_{dnn_opt_topk["exp"]}exp_sae_weights_{dnn_opt_topk["epoch"]}ep.npy'))
        )
        # voxel_sae_acts = torch.tensor(
        #     np.load(os.path.join(basedir, 'nsd/subj1/voxel_sae_acts', f'top{voxel_opt_topk["k"]}_aux_2exp_sae_weights_{voxel_opt_topk["epoch"]}ep.npy'))
        # )
        # source_rand_sae_acts = torch.tensor(
        #     np.load(os.path.join(basedir, 'coco_acts', model, 'sae_latents', layer, f'top{dnn_opt_topk["k"]}_aux_{dnn_opt_topk["exp"]}exp_sae_weights_randinit.npy'))
        # )

        base_scores, _, _ = voxel_dnn_align(coco_dir, subj_id, nsd_root, source_neurons, voxel_acts)
        base_score = base_scores.mean() / noise_ceiling
        base_error = sem(base_scores)

        sae_base_scores, _, _ = voxel_dnn_align(coco_dir, subj_id, nsd_root, source_sae_acts, voxel_acts)
        sae_base_score = sae_base_scores.mean() / noise_ceiling
        sae_base_error = sem(sae_base_scores)

        # sae_sae_scores = voxel_dnn_align(coco_dir, subj_id, source_sae_acts, voxel_sae_acts) / noise_ceiling
        # sae_sae_score = sae_sae_scores.mean()
        # sae_sae_error = sem(sae_sae_scores)

        # rand_sae_scores = voxel_dnn_align(coco_dir, subj_id, nsd_root, source_rand_sae_acts, voxel_acts) / noise_ceiling
        # rand_sae_score = rand_sae_scores.mean()
        # rand_sae_error = sem(rand_sae_scores)
        
        ax.bar(i, base_score, width, yerr=base_error, color='c', label=legend_labels[0])
        bar = ax.bar(i + width, sae_base_score, width, yerr=sae_base_error, color='m', label=legend_labels[1])
        # ax.bar(2*width, sae_sae_score, width, yerr=sae_sae_error, color='y', label=legend_labels[2])
        # ax.bar(i + (2*width), rand_sae_score, width, yerr=rand_sae_error, color='0.4', label=legend_labels[4])

        ax.bar_label(bar, labels=['{:+0.2f}%'.format(((sae_base_score / base_score) - 1) * 100)], padding=3)

        if i == 0:
            legend_labels = [None, None, None, None, None]

    # ax.set_ylabel('Noise-normalized Ridge Pearson r')
    ax.set_ylabel('Pairwise Mean Similarity')
    # ax.set_ylabel('RSA (Pearson r)')
    # ax.set_ylabel('Soft Match Corr')
    # ax.set_ylabel('NNLS Pearson r')
    ax.set_title(f'DNN->NSD Subject {subj_id} FFA Voxel Alignment')
    ax.set_xticks(np.arange(len(models)) + (1*width), x_labels)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)

    plt.savefig(os.path.join(savedir, f'dnn-nsd_subj{subj_id}_ffa_pairwise.png'), bbox_inches='tight', dpi=300)


def plot_dnn_alignment(basedir, savedir):
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


def plot_dnn_alignment_grid(act_root, nsd_root, coco_dir, savedir):
    models = ['resnet50', 'vit_b_16', 'clip_vit-b-32']
    layers = ['layer4.2', 'encoder.layers.encoder_layer_11', 'visual.transformer.resblocks.11']
    opt_topk = [
        {'k': 41, 'epoch': '300ep', 'exp': 2},
        {'k': 153, 'epoch': '300ep', 'exp': 2},
        {'k': 153, 'epoch': '300ep', 'exp': 2.0}
    ]
    opt_vanilla = [
        {'lambda': '1e-2', 'epoch': '300ep', 'exp': 4.0},
        {'lambda': '1e-2', 'epoch': '300ep', 'exp': 4.0},
        {'lambda': '1e-2', 'epoch': '300ep', 'exp': 4.0}
    ]
    uses_topk = [False, False, False]

    subj1_coco_ids = np.load(os.path.join(nsd_root, f'subj1_coco_IDs.npy'))
    all_ids = [f.split('.jpg')[0] for f in os.listdir(coco_dir)]
    subj1_coco_ids = [all_ids.index(str(id)) for id in subj1_coco_ids]

    base_results = []
    sae_results = []
    results = []
    for i in range(len(models)):
        source_model = models[i]
        source_layer = layers[i]

        source_neurons = torch.tensor(
            np.load(os.path.join(act_root, source_model, 'raw_neurons', f'{source_layer}.npy'))[subj1_coco_ids]
        )
        source_latents = torch.tensor(
            np.load(
                os.path.join(act_root, source_model, 'sae_latents', source_layer,
                f'top{opt_topk[i]["k"]}_aux_{opt_topk[i]["exp"]}exp_sae_weights_{opt_topk[i]["epoch"]}.npy' if uses_topk[i]
                else f'vanilla_{opt_vanilla[i]["lambda"]}lambda_{opt_vanilla[i]["exp"]}exp_sae_weights_{opt_vanilla[i]["epoch"]}.npy')
            )[subj1_coco_ids]
        )

        for j in range(len(models)):
            if i == j:
                results.append(0)
                base_results.append(1)
                sae_results.append(1)
                continue

            target_model = models[j]
            target_layer = layers[j]

            target_neurons = torch.tensor(
                np.load(os.path.join(act_root, target_model, 'raw_neurons', f'{target_layer}.npy'))[subj1_coco_ids]
            )
            target_latents = torch.tensor(
                np.load(
                    os.path.join(act_root, target_model, 'sae_latents', target_layer,
                    f'top{opt_topk[j]["k"]}_aux_{opt_topk[j]["exp"]}exp_sae_weights_{opt_topk[j]["epoch"]}.npy' if uses_topk[j]
                    else f'vanilla_{opt_vanilla[j]["lambda"]}lambda_{opt_vanilla[j]["exp"]}exp_sae_weights_{opt_vanilla[j]["epoch"]}.npy')
                )[subj1_coco_ids]
            )
            
            # base_score = np.array(RSA(source_neurons, target_neurons))
            base_score, base_ve, base_mse = RidgeRegression(source_neurons.numpy().astype(np.float64), target_neurons.numpy().astype(np.float64))
            base_score, base_ve, base_mse = np.array(base_score).mean(), np.array(base_ve).mean(), np.array(base_mse).mean()
            base_results.append(base_mse)

            # sae_score = np.array(RSA(source_latents, target_latents))
            sae_score, sae_ve, sae_mse = RidgeRegression(source_latents.numpy().astype(np.float64), target_neurons.numpy().astype(np.float64))
            sae_score, sae_ve, sae_mse = np.array(sae_score).mean(), np.array(sae_ve).mean(), np.array(sae_mse).mean()
            sae_results.append(sae_mse)

            if base_score < 0 and sae_score > 0:
                result = 200
            else:
                result = ((sae_score / base_score) - 1)*100

            results.append(result)
            print(f'{source_model}->{target_model}:\nSAE={sae_score},\nBase Neuron={base_score}\nResult={((sae_score / base_score) - 1)*100}\n')

    base_results = np.reshape(np.array(base_results), (len(models), len(models)))
    sae_results = np.reshape(np.array(sae_results), (len(models), len(models)))
    results = np.reshape(np.array(results), (len(models), len(models)))
    viridis = mpl.colormaps['viridis'].resampled(256)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), layout='constrained', squeeze=False)
    psm = ax[0, 0].pcolor(results, cmap=viridis, rasterized=True, vmin=-200, vmax=200)
    fig.colorbar(psm, ax=ax)

    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            # Position text at the center of each cell
            ax[0, 0].text(
                j + 0.5, i + 0.5, f'{results[i, j]:.2f}%', 
                ha='center', va='center', color='black', fontsize=6
            )

    ax[0, 0].set_xticks(np.arange(results.shape[0]) + (1/2), models, size=7)
    # ax[0, 0].set_xlabel('Feature 2 Corr', size=9)
    ax[0, 0].set_yticks(np.arange(results.shape[0]) + (1/2), models, size=7)
    # ax[0, 0].set_ylabel('Feature 1 Corr', size=9)
    ax[0, 0].set_title('Cross-Model Ridge SAE->Neuron % Relative to Neuron->Neuron', size=10)

    plt.savefig(os.path.join(savedir, f'dnn_rsa_grid.png'), bbox_inches='tight', dpi=300)


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


def plot_toy_sae_val_hist(basedir):
    seeds = [0, 1]
    num_feats = 64
    neurons = [8, 16, 32]

    train_split = 0.8
    feature_data = torch.tensor(
        np.load(os.path.join(basedir, f'{num_feats}feats_dataset.npy'))
    )
    feature_data = feature_data[-int(feature_data.shape[0]*(1-train_split)):]

    #   One histogram per toy model size?  That would be two models' neuron acts and SAE latents, so 4 separate acts total.
    #   Maybe best to split up per-model, but still plot neurons and SAE latents on same fig.
    fig, axs = plt.subplots(len(neurons), 2, sharey='row', layout='constrained')
    for i in range(len(neurons)):
        for j in range(len(seeds)):
            neuron_acts = torch.tensor(
                np.load(os.path.join(basedir, f'{num_feats}feats_{neurons[i]}neurons_powerlaw_final', f'model{seeds[j]}_raw_neuron_acts.npy'))
            )[-feature_data.shape[0]:]
            sae_acts = torch.tensor(
                np.load(os.path.join(basedir, f'{num_feats}feats_{neurons[i]}neurons_powerlaw_final', f'model{seeds[j]}_top7_sae_acts.npy'))
            )

            neuron_results = pairwise_corr(neuron_acts, feature_data).numpy()
            neuron_results = np.max(neuron_results, axis=0)
            sae_results = pairwise_corr(sae_acts, feature_data).numpy()
            sae_results = np.max(sae_results, axis=0)

            _, _, neuron_patches = axs[i, j].hist(neuron_results, bins=20, range=(0,1), color='c', alpha=0.5)
            _, _, sae_patches = axs[i, j].hist(sae_results, bins=20, range=(0,1), color='m', alpha=0.5)

            axs[i, j].axvline(neuron_results.mean(), color='c', linestyle='--')
            axs[i, j].axvline(sae_results.mean(), color='m', linestyle='--')
            axs[i, j].set_xlim(0, 1)

            if i == 0 and j == 0:
                neuron_patches.set_label('Base Neurons')
                sae_patches.set_label('SAE Latents')

                axs[i, j].legend(loc='upper right')
            elif j == 1:
                axs[i, j].set_ylabel(f'N={neurons[i]}                ', fontsize=12, rotation='horizontal')

        if i == 0:
            axs[i, 0].set_title('Seed 1', fontsize=10)
            axs[i, 1].set_title('Seed 2', fontsize=10)
        elif i == len(neurons)-1:
            axs[i, 0].set_xlabel('Max Pearson r', fontsize=10)
            axs[i, 0].set_ylabel('# of Features', fontsize=10)

    plt.savefig('/home/alongon/figures/superposition_alignment/toy_model_sae_validate.png', bbox_inches='tight', dpi=300)


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


def plot_convergence_grid(grid_path):
    grid_results = np.load(grid_path)

    viridis = mpl.colormaps['viridis'].resampled(256)
    n = 1

    fig, ax = plt.subplots(1, n, figsize=(n * 2 + 2, 3), layout='constrained', squeeze=False)
    psm = ax[0, 0].pcolor(grid_results, cmap=viridis, rasterized=True, vmin=0, vmax=1)
    fig.colorbar(psm, ax=ax)

    ax[0, 0].set_xticks(np.arange(grid_results.shape[0]) + (1/2), np.arange(-1, 1.25, 0.25), size=7)
    ax[0, 0].set_xlabel('Feature 2 Corr', size=9)
    ax[0, 0].set_yticks(np.arange(grid_results.shape[0]) + (1/2), np.arange(-1, 1.25, 0.25), size=7)
    ax[0, 0].set_ylabel('Feature 1 Corr', size=9)
    ax[0, 0].set_title('Superposition Arrangement Consistency\nCross-Model Mean Pairwise Corr', size=12)

    plt.savefig('/home/alongon/data/toy_arrange_converge/3feats_2neurons/converge_results.png', bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    face_dprime_histo(
        '/home/alongon/data/vinken_acts',
        '/home/alongon/data/vinken_face_cells/data/images.mat',
        '/home/alongon/figures/misc'
    )

    # dnn_brain_linreg_interp(
    #     '/home/alongon/data',
    #     '/home/alongon/figures/superposition_alignment',
    #     '/mnt/cogsci/KhoslaLab/NSD_Preprocessed/images/',
    #     '/mnt/cogsci/KhoslaLab/NSD_Preprocessed/',
    #     subj_id=1
    # )

    # plot_toy_sae_val_hist('/home/alongon/data/toy_align')

    # plot_dnn_alignment_grid(
    #     '/home/alongon/data/coco_acts',
    #     '/mnt/cogsci/KhoslaLab/NSD_Preprocessed/ventral_visual_data',
    #     '/mnt/cogsci/KhoslaLab/NSD_Preprocessed/images/',
    #     '/home/alongon/figures/superposition_alignment'
    # )

    # plot_convergence_grid(
    #     '/home/alongon/data/toy_arrange_converge/3feats_2neurons/convergence_results_grid.npy'
    # )

    # plot_brain_align_hierarchy(
    #     '/mnt/cogsci/KhoslaLab/NSD_Preprocessed/',
    #     '/home/alongon/data/nsd',
    #     '/mnt/cogsci/NSD_preprocessed_datasets_shreya/ventral_visual_data/ventral_visual_data_splits_1257.pickle',
    #     '/home/alongon/figures/superposition_alignment',
    #     subj1=1, subj2=2
    # )

    # nsd_subj_ids = [1,2,5,7]
    # for sub_id in nsd_subj_ids:
    # plot_dnn_brain_alignment(
    #     '/home/alongon/data',
    #     '/home/alongon/figures/superposition_alignment',
    #     '/mnt/cogsci/KhoslaLab/NSD_Preprocessed/images/',
    #     '/mnt/cogsci/KhoslaLab/NSD_Preprocessed',
    #     subj_id=1
    # )

    # plot_dnn_alignment('/home/alongon/data/imagenet_acts', '/home/alongon/figures/superposition_alignment')

    # plot_toy_alignments('/home/alongon/data/toy_data')

    # plot_toy_weight_histos('/home/alongon/data/toy_data')

    # source_neurons = torch.clamp(torch.tensor(np.load('/home/alongon/data/imagenet_acts/resnet18/raw_neurons/layer4.1_center_patch_valid.npy')), min=0, max=None)
    # source_latents = torch.clamp(torch.tensor(np.load('/home/alongon/data/imagenet_acts/resnet18/sae_latents/layer4.1/top16_5exp_sae_weights_100ep.npy')), min=0, max=None)
    # target_neurons = torch.clamp(torch.tensor(np.load('/home/alongon/data/imagenet_acts/resnet50/raw_neurons/layer4.2_center_patch_valid.npy')), min=0, max=None)

    # plot_ds_size(source_neurons, source_latents, target_neurons)