import os
import torch
import torch.distributed as dist
from torch import Tensor

import functools
import gc
import math
import random
import string
from typing import List, Optional, Tuple, Callable, Union

import numpy as np
import torch
from torch.backends import cudnn, opt_einsum
from torch.utils._pytree import tree_map

from torch.distributed.tensor import distribute_tensor, DTensor

def to_dist(x, from_local=False, **meta):
    if from_local:
        return DTensor.from_local(
            x,
            device_mesh=meta["device_mesh"],
            placements=meta["placements"],
            shape=meta["shape"],
            stride=meta["stride"],
        )
    else:
        return distribute_tensor(x, device_mesh=meta["device_mesh"], placements=meta["placements"])


def to_local(x, keep_sharded=False):
    if isinstance(x, DTensor):
        meta = dict(
            device_mesh=x.device_mesh,
            placements=x.placements,
            shape=x.shape,
            stride=x.stride(),
        )
        if keep_sharded:
            return x.to_local(), meta
        else:
            return x.full_tensor(), meta

    return x, None


def local_op(x, fn, keep_sharded=False):
    """
    converts to Tensor, does a thing, then back to Dtensor
    """
    x, meta = to_local(x, keep_sharded)
    x = fn(x)
    if meta is not None:
        x = to_dist(x, from_local=keep_sharded, **meta)
    return x
    

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]

            for base_i in range(len(params))[::self.world_size]:
                # every device will be responsible for a different param, but all devices need to share their shards and participate here
                grads = [None] * self.world_size
                metas = [None] * self.world_size
                for i in range(self.world_size):
                    if base_i + i < len(params):
                        # every device needs to be around for the gather
                        grads[i], metas[i] = to_local(params[base_i + i].grad, keep_sharded=False)
                        # but only one device is responsible for the update, free memory 
                        if not i == self.rank:
                            grads[i] = None

                # gradient is fully materialized, do the whitening
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = grads[self.rank]
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    if g.ndim == 4: # for the case of conv filters
                        g = g.view(len(g), -1)
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()


                # shard the update
                # first create empty tensor of its dimensions, and then broadcast it to all devices
                for i in range(self.world_size):
                    if base_i + i < len(params):
                        g_world = torch.empty_like(g)
                        dist.broadcast(g_world, src=self.rank)
                        # now distribute
                        g_world = to_dist(g_world, device_mesh=metas[i]["device_mesh"], placements=metas[i]["placements"])
                        # now do the update
                        params[base_i + i].add_(g_world, alpha=-group["lr"] * max(1, g.size(-2) / g.size(-1))**0.5)

