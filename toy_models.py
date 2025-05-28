import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from sae import SAE
from utils import pairwise_corr, pairwise_jaccard


#   NOTE:  code derived from https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb


class ToySAE(nn.Module):
    def __init__(
        self, 
        n_inputs,
        n_hidden,
        device
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.W_enc = nn.Parameter(torch.empty((n_inputs, n_hidden), device=device))
        self.W_dec = nn.Parameter(torch.empty((n_hidden, n_inputs), device=device))
        nn.init.xavier_normal_(self.W_enc)
        nn.init.xavier_normal_(self.W_dec)
        # self.b_final = nn.Parameter(torch.zeros((n_features), device=device))

        self.b_enc = nn.Parameter(torch.zeros(n_hidden, device=device))
        self.b_dec = nn.Parameter(torch.zeros(n_inputs, device=device))

    def forward(self, x):
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        hidden = (x @ self.W_enc) + self.b_enc #torch.einsum("...if,ifh->...ih", features, self.W)
        hidden = F.relu(hidden)
        out = (hidden @ self.W_dec) + self.b_dec #torch.einsum("...ih,ifh->...if", hidden, self.W)
        out = F.relu(out)
        return out, hidden


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
        # self.b_final = nn.Parameter(torch.zeros((n_features), device=device))

        if feature_probability is None:
            feature_probability = torch.ones(()) * 0.1
        self.feature_probability = feature_probability.to(device)
        if importance is None:
            importance = torch.ones(())
        self.importance = importance.to(device)

    def forward(self, features):
        # features: [..., instance, n_features]
        # W: [instance, n_features, n_hidden]
        hidden = features @ self.W #torch.einsum("...if,ifh->...ih", features, self.W)
        hidden = F.relu(hidden)
        out = hidden @ self.W.T #torch.einsum("...ih,ifh->...if", hidden, self.W)
        out = out# + self.b_final
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


if __name__ == '__main__':
    #   3 features with 0.1 freq and equal importance (1).  Data produced with seed=1.
    #   seed 1 produces A=[0, 1],      B=[.75, .25],  C=[1, 0].
    #   seed 2 produces A=[.75, .25],  B=[1, 0],      C=[0, 1].
    #
    #   ~ indicates that it is actually closer to [0.75, 0.25].
    #   0s are in reality negatives to reduce interference, but here we're focused on activations.
    torch.manual_seed(1)
    device = f"cuda:7" if torch.cuda.is_available() else "cpu"
    n_feats = 1024
    n_neurons = 256
    savedir = f'/home/alongon/data/toy_data/{n_feats}feats_{n_neurons}neurons'

    # #   TRAIN MODELS AND OBTAIN RAW NEURON ACTS
    # n_batch=1024
    # steps=10_000
    # print_freq=1000

    # dataset = np.load(os.path.join(savedir, 'dataset.npy'))
    # dataset = TensorDataset(torch.tensor(dataset, dtype=torch.float))
    # dataloader = DataLoader(dataset, batch_size=n_batch, shuffle=False, drop_last=False)

    # model = Model(n_feats, n_neurons, device)
    # opt = torch.optim.AdamW(list(model.parameters()), lr=1e-3)

    # all_batches = []
    # for i, batch in enumerate(dataloader, 0):
    #     batch = batch[0].to(device)
    # # for i in range(steps):
    #     opt.zero_grad(set_to_none=True)
    #     batch = model.generate_batch(n_batch)
    #     out, _ = model(batch)
    #     error = (model.importance*(batch.abs() - out)**2)
    #     loss = error.mean()

    #     loss.backward()
    #     opt.step()

    #     # all_batches += batch.detach().cpu().numpy().tolist()
        
    #     # if hooks:
    #     #     hook_data = dict(model=model,
    #     #                     step=step, 
    #     #                     opt=opt,
    #     #                     error=error,
    #     #                     loss=loss,
    #     #                     lr=step_lr)
    #     #     for h in hooks:
    #     #     h(hook_data)
    #     if i % print_freq == 0 or (i + 1 == steps):
    #         print(loss.item())

    # # np.save(os.path.join(savedir, 'dataset.npy'), np.array(all_batches))
    # print(model.W[:10, :10])
    # # exit()
    # model.eval()
    # #   TODO: collect activations of neurons to shared data (save data generated from one trial).  Then proceed
    # #         to train a vanilla SAE over a portion.  Then pass other portion to trained SAE to get latents.
    # #         Repeat for other model (save raw neuron and sae latent acts).  Then perform pairwise corrs over this.
    # all_acts = []
    # for i, batch in enumerate(dataloader, 0):
    #     batch = batch[0].to(device)
    #     _, acts = model(batch)
    #     all_acts += acts.detach().cpu().numpy().tolist()

    # np.save(os.path.join(savedir, 'model2_raw_neuron_acts.npy'), np.array(all_acts))

    # #   Train SAE.  Seed=2 for good results on Model 1.
    # train_split = 0.8
    # n_batch=1024
    # l1_lambda = 1e-8
    # aux_alpha = 1
    # topk = 32
    # print_freq=1000

    # data = np.load(os.path.join(savedir, 'model2_raw_neuron_acts.npy'))
    # dataset = TensorDataset(torch.tensor(data[:int(data.shape[0]*0.8)], dtype=torch.float))
    # dataloader = DataLoader(dataset, batch_size=n_batch, shuffle=False, drop_last=False)

    # sae = ToySAE(n_neurons, n_feats, device)
    # # sae = SAE(n_neurons, expansion=n_feats//n_neurons, device=device, topk=topk, auxk=128, dead_steps_threshold=1000)
    # opt = torch.optim.AdamW(list(sae.parameters()), lr=1e-3)
    # mse = torch.nn.functional.mse_loss
    # dead_acts_recon = None
    # for i, batch in enumerate(dataloader, 0):
    #     batch = batch[0].to(device)
    #     opt.zero_grad(set_to_none=True)

    #     out, latents = sae(batch)
    #     # latents, out, dead_acts_recon, num_dead = sae(batch)

    #     error = (batch.abs() - out)**2
    #     loss = error.mean()

    #     #  Auxiliary loss (prevents dead latents for topk)
    #     if dead_acts_recon is not None:
    #         error = batch - out
    #         aux_mse = mse(dead_acts_recon, error, reduction="none").sum(-1) / mse(error.mean(dim=0)[None, :].broadcast_to(error.shape), error, reduction="none").sum(-1)
    #         aux_loss = aux_alpha * aux_mse.nan_to_num(0)

    #         loss += aux_loss.mean()

    #     weighted_latents = latents * sae.W_dec.norm(dim=1)
    #     sparsity = weighted_latents.norm(p=1, dim=-1)

    #     loss += l1_lambda * sparsity.mean()

    #     loss.backward()
    #     opt.step()

    #     if i % print_freq == 0:
    #         print(f'MSE:  {error.mean().item()}')
    #         print(f'L1:  {sparsity.mean().item()}')
    #         # print(f'Num dead: {num_dead}')

    # # print(sae.W_enc)
    # # print(sae.W_dec)
    # sae.eval()

    # dataset = TensorDataset(torch.tensor(data[-int(data.shape[0]*(1-train_split)):], dtype=torch.float))
    # dataloader = DataLoader(dataset, batch_size=n_batch, shuffle=False, drop_last=False)

    # all_acts = []
    # for i, batch in enumerate(dataloader, 0):
    #     batch = batch[0].to(device)

    #     # latents = F.relu(batch @ sae.W_dec.T)
    #     _, latents = sae(batch)
    #     all_acts += latents.detach().cpu().numpy().tolist()

    # np.save(os.path.join(savedir, f'model2_1e-8lambda_sae_acts.npy'), np.array(all_acts))
    # exit()

    #   Pairwise pearson correlation and jaccard scores.
    train_split = 0.8

    # acts1 = np.load(os.path.join(savedir, 'model1_raw_neuron_acts.npy'))
    # acts1 = torch.tensor(acts1[-int(acts1.shape[0]*(1-train_split)):], device=device)
    # acts2 = np.load(os.path.join(savedir, 'model2_raw_neuron_acts.npy'))
    # acts2 = torch.tensor(acts2[-int(acts2.shape[0]*(1-train_split)):], device=device)

    # print('---Base Neurons---')
    # results = pairwise_corr(acts1, acts2)
    # print(f'Pearson pairwise: {torch.mean(torch.max(results, 1)[0])}')
    # print(f'Pearson pairwise: {torch.mean(torch.max(results, 0)[0])}')
    # score = pairwise_jaccard(acts1, acts2)
    # print(f'Jaccard pairwise: {score}')

    acts1 = np.load(os.path.join(savedir, 'model1_1e-8lambda_sae_acts.npy'))
    acts1 = torch.tensor(acts1[-int(acts1.shape[0]*(1-train_split)):], device=device)
    acts2 = np.load(os.path.join(savedir, 'model2_1e-8lambda_sae_acts.npy'))
    acts2 = torch.tensor(acts2[-int(acts2.shape[0]*(1-train_split)):], device=device)

    print('---SAE Latents---')
    results = pairwise_corr(acts1, acts2)
    print(f'Pearson pairwise: {torch.mean(torch.max(results, 1)[0])}')
    print(f'Pearson pairwise: {torch.mean(torch.max(results, 0)[0])}')
    score = pairwise_jaccard(acts1, acts2)
    print(f'Jaccard pairwise: {score}')


    #   TODO:  Linear mapping between models.