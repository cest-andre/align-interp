import os
import argparse
import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans

from sae import SAE, LN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--acts_path', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--relu', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--expansion', type=int, default=32)
    parser.add_argument('--topk', type=int, default=32)
    parser.add_argument('--num_input_dims', type=int, default=0)
    parser.add_argument('--archetype_k', type=int, default=0)  # value used in paper = 32000
    parser.add_argument('--device', type=int)
    args = parser.parse_args()
    assert args.start_epoch < args.epochs

    torch.manual_seed(0)
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    acts_data = np.load(args.acts_path)
    acts_data = torch.tensor(acts_data, dtype=torch.float)
    if args.relu:
        acts_data = torch.clamp(acts_data, min=0)

    acts_data, _, _ = LN(acts_data)

    num_dims = args.num_input_dims
    if num_dims > 0 and num_dims < acts_data.shape[1]:
        acts_data = acts_data[:, torch.randperm(acts_data.shape[1])[:num_dims]]
    else:
        num_dims = acts_data.shape[1]

    archetypes = None
    if args.archetype_k > 0:
        # archetypes = KMeans(n_clusters=args.archetype_k, random_state=0).fit(acts_data.numpy()).cluster_centers_
        # np.save(f'/home/alongon/model_weights/dnn_saes/resnet50/layer4.1.bn2/archetype_{args.archetype_k}means_results.npy', archetypes)
        # archetypes = torch.tensor(archetypes)

        archetypes = torch.tensor(np.load(f'/home/alongon/model_weights/dnn_saes/resnet50/layer4.1.bn2/archetype_{args.archetype_k}means_results.npy'))

    acts_ds = TensorDataset(acts_data)
    dataloader = DataLoader(acts_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    lr = 0.0004
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    aux_alpha = 1
    l1_lambda = 1e-8
    mse_scale = (
        1 / ((acts_data.float().mean(dim=0) - acts_data.float()) ** 2).mean()
    )

    sae = SAE(
        num_dims,
        args.expansion,
        topk=args.topk if args.topk > 0 else None,
        auxk=256,
        dead_steps_threshold=64,
        device=device,
        enc_data=None,
        archetypes=archetypes
    ).to(device)

    optimizer = optim.Adam(
        sae.parameters(),
        lr=lr,
        betas=(
            adam_beta1,
            adam_beta2,
        ),
        eps=6.25e-10
    )

    label = f'top{args.topk}_{args.expansion}exp_archetype' if args.topk > 0 else f'vanilla_1e-8lambda_{args.expansion}exp'
    if args.start_epoch > 0:
        sae.load_state_dict(torch.load(f"{args.ckpt_dir}/{label}_sae_weights_{args.start_epoch}ep.pth"))
        optimizer.load_state_dict(torch.load(f"{args.ckpt_dir}/{label}_opt_states_{args.start_epoch}ep.pth"))

    all_mses = []
    mse = torch.nn.functional.mse_loss
    for ep in range(args.start_epoch, args.epochs):
        print(f"Epoch {ep}")
        total_mse = 0
        total_l1 = 0
        total_dead = 0
        for i, acts in enumerate(dataloader, 0):
            optimizer.zero_grad()

            acts = acts[0].to(device)
            latents, acts_hat, preact_feats, num_dead = sae(acts)

            mse_loss = mse_scale * mse(acts_hat, acts, reduction="none").sum(-1)
            # weighted_latents = latents * sae.W_dec.norm(dim=1)
            # sparsity = weighted_latents.norm(p=1, dim=-1)

            total_mse += mse_loss.mean().cpu().detach().numpy()
            # total_l1 += sparsity.mean().cpu().detach().numpy()
            total_dead += num_dead.cpu().numpy()

            loss = mse_loss.mean()

            #   REANIMATE LOSS!!!  Credit to Thomas Fel
            is_dead = ((latents > 0).sum(dim=0) == 0).float().detach()
            # we push the pre_codes (before relu) towards the positive orthant
            reanim_loss = (preact_feats * is_dead[None, :]).mean()

            loss -= reanim_loss * 1e-3

            # #  Auxiliary loss (prevents dead latents)
            # if dead_acts_recon is not None:
            #     error = acts - acts_hat
            #     aux_mse = mse(dead_acts_recon, error, reduction="none").sum(-1) / mse(error.mean(dim=0)[None, :].broadcast_to(error.shape), error, reduction="none").sum(-1)
            #     aux_loss = aux_alpha * aux_mse.nan_to_num(0)

            #     loss += aux_loss.mean()

            # loss += l1_lambda * sparsity.mean()

            loss.backward()
            optimizer.step()

        print(f"Average mse:  {total_mse / i}")
        all_mses.append(total_mse / i)
        print(f"Average L1:  {total_l1 / i}")
        print(f"Average Dead: {total_dead / i}")

        if (ep+1) % 50 == 0:
            torch.save(sae.state_dict(), f"{args.ckpt_dir}/{label}_sae_weights_{ep+1}ep.pth")
            torch.save(optimizer.state_dict(), f"{args.ckpt_dir}/{label}_opt_states_{ep+1}ep.pth")
        
    np.save(f"{args.ckpt_dir}/{label}_sae_maes_{ep+1}.npy", np.array(all_mses))