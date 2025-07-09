import torch
from torch import nn
import numpy as np
import math


def LN(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


#   Archetypal code source:  https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/sae/archetypal_dictionary.py
class SAE(nn.Module):
    def __init__(self, input_dims, expansion=32, dtype=torch.float, device="cuda", topk=None, auxk=None, dead_steps_threshold=None, enc_data=None, archetypes=None):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = input_dims
        self.num_latents = input_dims * expansion
        self.dtype = dtype
        self.device = device
        self.topk = topk
        self.archetypes = archetypes

        self.auxk = auxk
        self.dead_steps_threshold = dead_steps_threshold

        def auxk_mask_fn(x):
            dead_mask = self.stats_last_nonzero > self.dead_steps_threshold
            x.data *= dead_mask
            return x

        self.auxk_mask_fn = auxk_mask_fn
        self.register_buffer("stats_last_nonzero", torch.zeros(self.num_latents, dtype=torch.long, device=self.device))

        use_archetype = self.archetypes is not None
        self.decode = self.archetype_decode if use_archetype else self.vanilla_decode
        self.init_weights(use_archetype=use_archetype, enc_data=enc_data)

        # self.enc_bn = nn.BatchNorm1d(self.num_latents)

    def init_weights(self, use_archetype, enc_data=None):
        if use_archetype:
            #   NOTE: should I pass archetypes through LN?  I believe paper mentioned doing something to this effect.
            #   If I still have trouble with getting good results here, perhaps I can lower archetype_k  to be proportional
            #   to the smaller dataset I use (or just take rand subset from this already extracted list?).
            # self.archetypes, _, _ = LN(self.archetypes)
            self.register_buffer("C", self.archetypes)
            self.W = nn.Parameter(torch.eye(self.num_latents, self.archetypes.shape[0], dtype=self.dtype, device=self.device))
            self.Relax = nn.Parameter(torch.zeros(self.num_latents, self.input_dims, dtype=self.dtype, device=self.device))
            self.delta = 1# / self.num_latents

            self._fused_dictionary = None
        else:
            self.W_dec = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(
                        self.num_latents, self.output_dims, dtype=self.dtype, device=self.device
                    )
                )
            )

        self.b_dec = nn.Parameter(
            torch.zeros(self.output_dims, dtype=self.dtype, device=self.device)
        )

        #   enc_data is a tensor subset of training data ideally with shape: (num_latents, input_dims).
        #   Subset would be taken in training code.  If dataset is too small, the entire set should be passed,
        #   and it will be duplicated to match the number of sae latents.
        if enc_data is not None:
            # kaiming_bound = torch.max(self.W_dec)
            kaiming_bound = 0.1083
            # enc_data, _, _ = LN(enc_data)

            data_min = torch.min(enc_data, dim=1).values
            data_min = data_min[:, None].broadcast_to(enc_data.shape)

            data_max = torch.max(enc_data, dim=1).values
            data_max = data_max[:, None].broadcast_to(enc_data.shape)

            enc_data = (((enc_data - data_min) / (data_max - data_min)) * kaiming_bound) - kaiming_bound
            if enc_data.shape[0] < self.num_latents:
                #  duplicate enc_data examples to match num_latents
                enc_data = enc_data[None, :, :].broadcast_to((math.ceil(self.num_latents / enc_data.shape[0]), enc_data.shape[0], enc_data.shape[1]))
                self.W_enc = nn.Parameter(torch.flatten(enc_data, end_dim=1)[:self.num_latents].T)
            else:
                self.W_enc = nn.Parameter(enc_data.T)
        else:
            self.W_enc = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(
                        self.input_dims, self.num_latents, dtype=self.dtype, device=self.device
                    )
                )
            )

        self.b_enc = nn.Parameter(
            torch.zeros(self.num_latents, dtype=self.dtype, device=self.device)
        )

    def encode(self, x):
        preact_feats = x @ self.W_enc + self.b_enc
        # preact_feats = self.enc_bn(preact_feats)
        top_dead_acts = None
        num_dead = torch.tensor(0)

        if self.topk is not None:
            topk_res = torch.topk(preact_feats, k=self.topk, dim=-1)
            values = nn.ReLU()(topk_res.values)
            x = torch.zeros_like(preact_feats, device=self.device)
            x.scatter_(-1, topk_res.indices, values)

            # self.stats_last_nonzero *= (x == 0).all(dim=0).long()
            # self.stats_last_nonzero += 1

            # auxk_acts = self.auxk_mask_fn(preact_feats)
            # #   TODO: ensure all batch entries have at least auxk nonzero (dead) acts.
            # #   On second thought, do I even need to check this?  As there are so many latents,
            # #   and only k can fire each batch, perhaps it's fairly likely that at least 512
            # #   do not fire for the first epoch.
            # if (torch.sum(auxk_acts != 0, dim=-1) >= self.auxk).all(dim=0):
            #     num_dead = torch.mean(torch.sum(auxk_acts != 0, dim=-1).type(torch.float))
            #     deadk_res = torch.topk(auxk_acts, k=self.auxk, dim=-1)
            #     top_dead_acts = torch.zeros_like(auxk_acts)
            #     top_dead_acts.scatter_(-1, deadk_res.indices, deadk_res.values)
            #     top_dead_acts = nn.ReLU()(top_dead_acts)
        else:
            x = preact_feats

        return nn.ReLU()(x), preact_feats, num_dead

    def vanilla_decode(self, x, mu, std):
        x = x @ self.W_dec + self.b_dec
        # x = x * std + mu
        return x

    def archetype_decode(self, x, mu, std):
        dictionary = self.get_dict()
        x = x @ dictionary + self.b_dec
        # x = x * std + mu
        return x

    def get_dict(self):
        if self.training:
            # we are in .train() mode, compute the dictionary on the fly
            with torch.no_grad():
                # ensure W remains row-stochastic (positive and row sum to one)
                W = torch.relu(self.W)
                W /= (W.sum(dim=-1, keepdim=True) + 1e-8)
                self.W.data = W

                # enforce the norm constraint on Lambda to limit deviation from conv(C)
                norm_Lambda = self.Relax.norm(dim=-1, keepdim=True)  # norm per row
                scaling_factor = torch.clamp(self.delta / norm_Lambda, max=1)  # safe scaling factor
                self.Relax.data *= scaling_factor  # scale Lambda to satisfy ||Lambda|| < delta

            # compute the dictionary as a convex combination plus relaxation
            D = self.W @ self.C + self.Relax
            return D# * torch.exp(self.multiplier)
        else:
            # we are in .eval() mode, return the fused dictionary
            assert self._fused_dictionary is not None, "Dictionary is not initialized."
            return self._fused_dictionary

    def forward(self, x):
        dead_acts_recon = None
        # x, mu, std = LN(x)
        mu = std = None

        latents, preact_feats, num_dead = self.encode(x)
        out = self.decode(latents, mu, std)

        # if top_dead_acts is not None:
        #     dead_acts_recon = self.decode(top_dead_acts, mu, std)

        return latents, out, preact_feats, num_dead

    def train(self, mode=True):
        """
        Hook called when switching between training and evaluation mode.
        We use it to fuse W, C, Relax and multiplier into a single dictionary.

        Parameters
        ----------
        mode : bool, optional
            Whether to set the model to training mode or not, by default True.
        """
        if not mode:
            # we are in .eval() mode, fuse the dictionary
            if self.archetypes is not None:
                self._fused_dictionary = self.get_dict()
        super().train(mode)


#   TODO:  SCRAPPED UNTIL RESIDUAL STREAM HAS BEEN IMPLEMENTED (is it not already implemented within model.layerX modules?).
#
#   Wrapper so that we have control over the forward pass.
# class ModelWrapper(nn.Module):
#     target_neuron = None
#     all_acts = []

#     def __init__(self, model, expansion, device, use_sae=False, input_dims=None):
#         super().__init__()
#         self.device = device
#         self.block_output = None

#         self.block_input = nn.Sequential()
#         self.block_input.append(model.conv1)
#         self.block_input.append(model.bn1)
#         self.block_input.append(model.relu)
#         self.block_input.append(model.maxpool)
#         self.block_input.append(model.layer1)
#         self.block_input.append(model.layer2)

#         # self.block_input.append(model.layer3[:5])
#         # self.block_output = nn.Sequential()
#         # self.block_output.append(model.layer3[5].conv1)
#         # self.block_output.append(model.layer3[5].bn1)
#         # self.block_output.append(model.layer3[5].relu)
#         # self.block_output.append(model.layer3[5].conv2)
#         # self.block_output.append(model.layer3[5].bn2)
#         # self.block_output.append(model.layer3[5].relu)
#         # self.block_output.append(model.layer3[5].conv3)
#         # self.block_output.append(model.layer3[5].bn3)

#         self.block_input.append(model.layer3)
#         self.block_input.append(model.layer4[0])
#         self.block_input.append(model.layer4[1].conv1)
#         self.block_input.append(model.layer4[1].bn1)
#         self.block_input.append(model.layer4[1].relu)
#         self.block_input.append(model.layer4[1].conv2)
#         self.block_input.append(model.layer4[1].bn2)

#         self.use_sae = use_sae
#         if self.use_sae:
#             assert input_dims != None
#             self.map = nn.Linear(input_dims, input_dims*expansion, bias=False)

#     def forward(self, x, relu=False, center=False):
#         x = x.to(self.device)
#         x = self.block_input(x)

#         if self.block_output is not None:
#             presum_out = self.block_output(x)
#             x = x + presum_out

#         if relu:
#             x = nn.ReLU(inplace=True)(x)

#         # x = torch.mean(torch.flatten(x, start_dim=-2), dim=-1)
#         if center:
#             center_coord = x.shape[-1] // 2
#             x = x[:, :, center_coord, center_coord]

#             if self.use_sae:
#                 x = self.map(x)
#                 x = x[:, :, None, None]

#         return x