import os
import argparse
import numpy as np
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from sae import SAE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxel_path', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--expansion', type=int, default=32)
    parser.add_argument('--topk', type=int, default=32)
    parser.add_argument('--num_voxels', type=int, default=5931)
    parser.add_argument('--device', type=int)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    voxel_data = np.load(args.voxel_path)
    voxel_data = torch.tensor(voxel_data)
    voxel_ds = TensorDataset(voxel_data)
    dataloader = DataLoader(voxel_ds, batch_size=args.batch_size, shuffle=True)
    input_dims = voxel_data[0].shape[-1]

    sae = SAE(args.num_voxels, args.expansion, topk=args.topk if args.topk > 0 else None, auxk=128, dead_steps_threshold=16, device='cpu', enc_data=None).to(device)
    # sae.load_state_dict(torch.load(f"{args.ckpt_dir}/{label}_sae_weights_{10}ep.pth"))

    lr=0.0004
    adam_beta1=0.9
    adam_beta2=0.999
    aux_alpha = 256
    label = f'top{args.topk}_{args.num_voxels}subset_{args.batch_size}batch_{aux_alpha}aux_{args.expansion}exp' if args.topk > 0 else f'vanilla_{args.expansion}exp'

    mse_scale = (
        1 / ((voxel_data.float().mean(dim=0) - voxel_data.float()) ** 2).mean()
    )

    optimizer = optim.Adam(
        sae.parameters(),
        lr=lr,
        betas=(
            adam_beta1,
            adam_beta2,
        ),
        eps=6.25e-10
    )

    all_mses = []
    mse = torch.nn.functional.mse_loss
    for ep in range(0, args.epochs):
        print(f"Epoch {ep}")
        total_mse = 0
        total_l1 = 0
        total_dead = 0
        for i, voxels in enumerate(dataloader, 0):
            optimizer.zero_grad()

            voxels = voxels[0][:, :args.num_voxels].to(device)
            latents, voxels_hat, dead_voxels_recon, num_dead = sae(voxels)

            mse_loss = mse_scale * mse(voxels_hat, voxels, reduction="none").sum(-1)
            weighted_latents = latents * sae.W_dec.norm(dim=1)
            sparsity = weighted_latents.norm(p=1, dim=-1)

            total_mse += mse_loss.mean().cpu().detach().numpy()
            total_l1 += sparsity.mean().cpu().detach().numpy()
            total_dead += num_dead.cpu().numpy()

            loss = mse_loss

            #  Auxiliary loss (prevents dead latents)
            if dead_voxels_recon is not None:
                error = voxels - voxels_hat
                aux_mse = mse(dead_voxels_recon, error, reduction="none").sum(-1) / mse(error.mean(dim=0)[None, :].broadcast_to(error.shape), error, reduction="none").sum(-1)
                aux_loss = aux_alpha * aux_mse.nan_to_num(0)

                loss += aux_loss

            loss = loss.mean()
            loss.backward()

            optimizer.step()

        print(f"Average mse:  {total_mse / i}")
        all_mses.append(total_mse / i)

        print(f"Average L1:  {total_l1 / i}")

        print(f"Average Dead: {total_dead / i}")

        if (ep+1) % 100 == 0:
            torch.save(sae.state_dict(), f"{args.ckpt_dir}/{label}_sae_weights_{ep+1}ep.pth")
        
    np.save(f"{args.ckpt_dir}/{label}_sae_maes_{ep+1}.npy", np.array(all_mses))