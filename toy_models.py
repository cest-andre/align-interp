import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'representation-alignment'))
from src.alignment.linear import Linear

from sae import SAE
from utils import pairwise_corr, pairwise_jaccard


#   NOTE:  toy model code derived from https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb
class Model(nn.Module):
    def __init__(
        self, 
        n_features,
        n_hidden,
        device,
        feature_probability = None,
        importance = None
    ):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.W = nn.Parameter(torch.empty((n_features, n_hidden), device=device))
        nn.init.xavier_normal_(self.W)
        self.b_final = nn.Parameter(torch.zeros((n_features), device=device))

        if feature_probability is None:
            feature_probability = torch.ones(()) * 0.1
        self.feature_probability = feature_probability.to(device)
        if importance is None:
            #   NOTE: change importance to be a simple powerlaw of x^-2, sampled 64 times uniformly from 1 to 3 (importance=1 to 0.1)
            # importance = torch.ones(())
            importance = torch.pow(torch.arange(1, 3, (2 / self.n_features)), -2)
        self.importance = importance.to(device)

    def forward(self, features):
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        hidden = features @ self.W #torch.einsum("...if,ifh->...ih", features, self.W)
        hidden = F.relu(hidden)
        out = hidden @ self.W.T #torch.einsum("...ih,ifh->...if", hidden, self.W)
        out = out + self.b_final
        out = F.relu(out)
        return out, hidden

    def generate_batch(self, n_batch):
        feat = torch.rand((n_batch, self.n_features), device=self.W.device)
        batch = torch.where(
            torch.rand((n_batch, self.n_features), device=self.W.device) <= self.feature_probability,
            feat,
            torch.zeros((), device=self.W.device),
        )
        return batch

        
class ToySAE(nn.Module):
    def __init__(
        self, 
        n_inputs,
        n_hidden,
        device,
        topk=None
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.topk = topk
        self.device = device

        self.W_enc = nn.Parameter(torch.empty((n_inputs, n_hidden), device=device))
        self.W_dec = nn.Parameter(torch.empty((n_hidden, n_inputs), device=device))
        nn.init.xavier_normal_(self.W_enc)
        nn.init.xavier_normal_(self.W_dec)
        self.b_enc = nn.Parameter(torch.zeros(n_hidden, device=device))
        self.b_dec = nn.Parameter(torch.zeros(n_inputs, device=device))

    def forward(self, x):
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        hidden = (x @ self.W_enc) + self.b_enc #torch.einsum("...if,ifh->...ih", features, self.W)
        preact_feats = hidden
        hidden = F.relu(hidden)

        if self.topk is not None:
            topk_res = torch.topk(hidden, k=self.topk, dim=-1)
            values = nn.ReLU()(topk_res.values)
            hidden = torch.zeros_like(hidden, device=self.device)
            hidden.scatter_(-1, topk_res.indices, values)

        out = (hidden @ self.W_dec) + self.b_dec #torch.einsum("...ih,ifh->...if", hidden, self.W)
        out = F.relu(out)

        return out, hidden, preact_feats


if __name__ == '__main__':
    #   3 features with 0.1 freq and equal importance (1).  Data produced with seed=1.
    #   seed 1 produces A=[0, 1],      B=[.75, .25],  C=[1, 0].
    #   seed 2 produces A=[.75, .25],  B=[1, 0],      C=[0, 1].
    #
    #   ~ indicates that it is actually closer to [0.75, 0.25].
    #   0s are in reality negatives to reduce interference, but here we're focused on activations.
    seed = 1
    device = f"cuda:4" if torch.cuda.is_available() else "cpu"
    n_feats = 64
    n_neurons = 32
    savedir = f'/home/alongon/data/toy_data/{n_feats}feats_{n_neurons}neurons_bias'
    Path(savedir).mkdir(parents=True, exist_ok=True)

    #   GENERATE DATA, TRAIN MODEL, AND OBTAIN RAW NEURON ACTS
    torch.manual_seed(seed)
    n_batch=1024
    steps=10_000
    print_freq=1000

    model = Model(n_feats, n_neurons, device)
    opt = torch.optim.AdamW(list(model.parameters()), lr=1e-3)

    # if seed == 0:
    #     all_batches = []
    #     for i in range(steps):
    #         batch = model.generate_batch(n_batch)
    #         all_batches += batch.detach().cpu().numpy().tolist()
    #     np.save(os.path.join(savedir, 'dataset.npy'), np.array(all_batches))

    dataset = np.load(os.path.join(savedir, 'dataset.npy'))
    dataset = TensorDataset(torch.tensor(dataset, dtype=torch.float))
    dataloader = DataLoader(dataset, batch_size=n_batch, shuffle=False, drop_last=False)

    # for i, batch in enumerate(dataloader, 0):
    #     batch = batch[0].to(device)

    #     out, _ = model(batch)
    #     error = (model.importance*(batch.abs() - out)**2)
    #     loss = error.mean()

    #     loss.backward()
    #     opt.step()
    #     opt.zero_grad(set_to_none=True)

    #     if i % print_freq == 0 or (i + 1 == steps):
    #         print(loss.item())

    # torch.save(model.state_dict(), os.path.join(savedir, f'model{seed}_weights.pth'))

    model.load_state_dict(torch.load(os.path.join(savedir, f'model{seed}_weights.pth')))
    model.eval()
    all_acts = []
    all_errors = []
    for i, batch in enumerate(dataloader, 0):
        batch = batch[0].to(device)
       
        out, acts = model(batch)
        all_acts += acts.detach().cpu().numpy().tolist()
        all_errors += ((batch.abs() - out)**2).detach().cpu().numpy().tolist()

    np.save(os.path.join(savedir, f'model{seed}_raw_neuron_acts.npy'), np.array(all_acts))
    np.save(os.path.join(savedir, f'model{seed}_recon_errors.npy'), np.array(all_errors))
    exit()

    # #   Train SAE.
    # train_split = 0.8
    # n_batch=1024
    # l1_lambda = 1e-1
    # topk = 7
    # print_freq=1000

    # torch.manual_seed(0)
    # data = np.load(os.path.join(savedir, f'model{seed}_raw_neuron_acts.npy'))
    # dataset = TensorDataset(torch.tensor(data[:int(data.shape[0]*0.8)], dtype=torch.float))
    # dataloader = DataLoader(dataset, batch_size=n_batch, shuffle=False, drop_last=False)

    # sae = ToySAE(n_neurons, n_feats, device, topk=topk)
    # opt = torch.optim.AdamW(list(sae.parameters()), lr=1e-3)
    # for i, batch in enumerate(dataloader, 0):
    #     batch = batch[0].to(device)
    #     opt.zero_grad(set_to_none=True)

    #     out, latents, preact_feats = sae(batch)

    #     error = (batch.abs() - out)**2
    #     loss = error.mean()

    #     #   REANIMATE LOSS!!!  Credit to Thomas Fel
    #     is_dead = ((latents > 0).sum(dim=0) == 0).float().detach()
    #     # we push the pre_codes (before relu) towards the positive orthant
    #     reanim_loss = (preact_feats * is_dead[None, :]).mean()

    #     loss -= reanim_loss * 1e-3

    #     weighted_latents = latents * sae.W_dec.norm(dim=1)
    #     sparsity = weighted_latents.norm(p=1, dim=-1)
    #     # loss += l1_lambda * sparsity.mean()

    #     loss.backward()
    #     opt.step()

    #     if i % print_freq == 0:
    #         print(f'MSE:  {error.mean().item()}')
    #         print(f'L1:  {sparsity.mean().item()}\n')
    #         # print(f'Num dead: {num_dead}')

    # sae.eval()

    # dataset = TensorDataset(torch.tensor(data[-int(data.shape[0]*(1-train_split)):], dtype=torch.float))
    # dataloader = DataLoader(dataset, batch_size=n_batch, shuffle=False, drop_last=False)

    # all_acts = []
    # for i, batch in enumerate(dataloader, 0):
    #     batch = batch[0].to(device)
    #     _, latents, _ = sae(batch)

    #     all_acts += latents.detach().cpu().numpy().tolist()

    # np.save(os.path.join(savedir, f'model{seed}_top{topk}_sae_acts.npy'), np.array(all_acts))
    # exit()

    #   Pairwise pearson correlation and jaccard scores.
    train_split = 0.8
    source_seed = 0
    target_seed = 1

    source_acts = np.load(os.path.join(savedir, f'model{source_seed}_raw_neuron_acts.npy'))
    source_acts = torch.tensor(source_acts[-int(source_acts.shape[0]*(1-train_split)):], device=device)
    target_acts = np.load(os.path.join(savedir, f'model{target_seed}_raw_neuron_acts.npy'))
    target_acts = torch.tensor(target_acts[-int(target_acts.shape[0]*(1-train_split)):], device=device)

    # print('---Base Neurons---')
    # # results = pairwise_corr(acts1, acts2)
    # # print(f'Pearson pairwise: {torch.mean(torch.max(results, 1)[0])}')
    # # print(f'Pearson pairwise: {torch.mean(torch.max(results, 0)[0])}')
    # # score = pairwise_jaccard(acts1, acts2)
    # # print(f'Jaccard pairwise: {score}')
    metric = Linear()
    print(f"Ridge scores: {metric.fit_kfold_ridge(x=source_acts.cpu().to(torch.float), y=target_acts.cpu().to(torch.float))}")

    source_acts = np.load(os.path.join(savedir, f'model{source_seed}_top{topk}_sae_acts.npy'))
    source_acts = torch.tensor(source_acts, device=device)
    # target_acts = np.load(os.path.join(savedir, f'model{target_seed}_top{topk}_sae_acts.npy'))
    # target_acts = torch.tensor(target_acts, device=device)

    print('---SAE Latents---')
    # results = pairwise_corr(acts1, acts2)
    # print(f'Pearson pairwise: {torch.mean(torch.max(results, 1)[0])}')
    # print(f'Pearson pairwise: {torch.mean(torch.max(results, 0)[0])}')
    # score = pairwise_jaccard(acts1, acts2)
    # print(f'Jaccard pairwise: {score}')
    metric = Linear()
    print(f"Ridge scores: {metric.fit_kfold_ridge(x=source_acts.cpu().to(torch.float), y=target_acts.cpu().to(torch.float))}")
    # results = pairwise_corr(source_acts, target_acts)
    # sae_sae_score = torch.mean(torch.max(results, 1)[0]).cpu().numpy()
    # print(sae_sae_score)