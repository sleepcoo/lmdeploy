# Copyright (c) OpenMMLab. All rights reserved.
import os
import re

import ffn_4
import ffn_23
import torch
from einops import rearrange
from torch import nn
from transformers.models.llama.modeling_llama import LlamaMLP

from lmdeploy.pytorch.weight_utils import hf_model_weights_iterator


def modify_string(original_string: str):
    modified_string = re.sub(r'_\d+', '', original_string)
    return 'sp.' + modified_string


class ParallelSP(nn.Module):

    def __init__(
        self,
        input_dim: int = 5120,
        output_dim: int = 13824,
        low_rank_dim: int = 1000,
    ) -> None:
        super().__init__()

        self.embed_dim = input_dim
        self.low_rank_dim = low_rank_dim
        self.sp_fcx = nn.Linear(input_dim, low_rank_dim, bias=False)
        self.sp_fcy = nn.Linear(low_rank_dim, output_dim, bias=False)
        self.acf_fn = torch.nn.SiLU()

    def forward(self, x):
        idx = self.sp_fcx(x.view(self.embed_dim))
        idx = self.sp_fcy(self.acf_fn(idx))
        return idx


class LlamaSparseMLP(LlamaMLP):

    def __init__(self, config, model_path, torch_dtype, idx=0, llama_mlp=None):
        super().__init__(config)
        self.sp = ParallelSP(config.hidden_size, config.intermediate_size)
        if isinstance(llama_mlp, LlamaMLP):
            own_state = self.state_dict()
            for name, param in llama_mlp.named_parameters():
                if name in own_state:
                    own_state[name] = own_state[name].to(dtype=param.dtype)
                    own_state[name].copy_(param)
                else:
                    raise KeyError(
                        f"Parameter '{name}' not found in LlamaSparseMLP")
            if hasattr(llama_mlp, 'act_fn'):
                self.act_fn = llama_mlp.act_fn

            #self._load_sp_weight(own_state, model_path, idx)
            self.load_state_dict(own_state)
            self.down_weight_t = self.down_proj.weight.t().contiguous()
            self.down_weight_t = self.down_weight_t.to(
                'cuda:0').half().t().contiguous()
            self.to(dtype=torch_dtype)
            self.threshold = 0

            #test code
            self.vec_sparse = torch.rand(self.intermediate_size,
                                         device="cuda:0",
                                         dtype=torch.float16)
            self.vec_sparse = torch.relu(self.vec_sparse - 8 / 10)
            print(
                ">>> act_rate:",
                round(
                    torch.sum(self.vec_sparse > 0).item() * 100 /
                    self.vec_sparse.numel(), 2))

    def _load_sp_weight(self, own_state, model_path, idx):
        sp_model_path = os.path.join(model_path, 'sparse_predictor')
        for name, loaded_weight in hf_model_weights_iterator(sp_model_path):
            cur_idx = re.findall(r'\d+', name)[0]
            if 'sp_fc' in name and int(cur_idx) == idx:
                own_state[modify_string(name)].copy_(loaded_weight)

    def mlp_sparse(self,
                   idx,
                   x,
                   gate_weight,
                   up_weight,
                   down_weight,
                   threshold,
                   act_fn,
                   row=4096,
                   col=11008):
        cuda_res = torch.empty(col + col + row,
                               device='cuda:0',
                               dtype=torch.float16)

        ffn_23.torch_launch_ffn_fuse_23(idx, x, gate_weight, cuda_res[0:col],
                                        row, col, threshold)
        ffn_23.torch_launch_ffn_fuse_23(idx, x, up_weight,
                                        cuda_res[col:col + col], row, col,
                                        threshold)
        self.down_proj(cuda_res[0:col] * cuda_res[col:col + col])
        return cuda_res[col + col:].unsqueeze_(-2).unsqueeze_(-2)

    def sparse_mlp(self, x):
        #idx = self.sp(x)
        out = self.mlp_sparse(self.vec_sparse, x.squeeze_(),
                              self.gate_proj.weight, self.up_proj.weight,
                              self.down_weight_t, 0, self.act_fn,
                              self.hidden_size, self.intermediate_size)
        return out

    def forward(self, x):
        if x.size(1) == 1 and x.size(0) == 1:
            self.sparse_mlp(x)
        else:
            x = self.down_proj(
                self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return x
