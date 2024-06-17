# Copyright (c) 2023, Albert Gu, Tri Dao.

from functools import partial

from collections import namedtuple

from mamba_ssm.models.mixer_seq_simple import _init_weights, create_block
from mamba_ssm.utils.generation import GenerationMixin
import pathlib

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import torch
import torch.nn.functional as F

from torch import nn
from x_transformers.autoregressive_wrapper import (
    ENTMAX_ALPHA,
    entmax,
    exists,
    top_a,
    top_k,
    top_p,
)
from x_transformers.x_transformers import exists


class MixerModel(nn.Module):
    def __init__(
            self,
            encoding,
            max_seq_len,
            d_model: int,
            n_layer: int,
            prompt_enable,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            device=None,
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.max_seq_len = max_seq_len
        self.prompt_enable = prompt_enable
        n_tokens = encoding["n_tokens"]
        self.residual_in_fp32 = residual_in_fp32
        self.embedding = nn.ModuleList([nn.Embedding(n, d_model, **factory_kwargs) for n in n_tokens])
        if self.prompt_enable:
            # LLAMA_checkpoint = torch.load(pathlib.Path('model/7B/consolidated.00.pth'))
            # self.embedding_prompt = nn.Embedding.from_pretrained(
            #     LLAMA_checkpoint['tok_embeddings.weight'][:, :d_model])
            # del LLAMA_checkpoint
            # self.embedding_prompt.weight.requires_grad = prompt_enable
            self.embedding_prompt = nn.Embedding(32000, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.lm_head = nn.ModuleList([nn.Linear(d_model, n, **factory_kwargs) for n in n_tokens])

        if self.prompt_enable:
            self.lm_head_prompt = nn.Linear(d_model, 32000)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, prompt, inference_params=None, num_last_tokens=0):
        hidden_states = sum(emb(input_ids[..., i]) for i, emb in enumerate(self.embedding))
        if self.prompt_enable:
            hidden_states = hidden_states + self.embedding_prompt(prompt)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = [to_logit(hidden_states) for to_logit in self.lm_head]

        CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
        Casual_logits = [CausalLMOutput(val) for val in lm_logits]
        out_prompt = None
        if self.prompt_enable:
            out_prompt = self.lm_head_prompt(hidden_states)

        return lm_logits, out_prompt


def sample(logits, kind, threshold, temperature, min_p_pow, min_p_ratio):
    """Sample from the logits with a specific sampling strategy."""
    if kind == "top_k":
        probs = F.softmax(top_k(logits, thres=threshold) / temperature, dim=-1)
    elif kind == "top_p":
        probs = F.softmax(top_p(logits, thres=threshold) / temperature, dim=-1)
    elif kind == "top_a":
        probs = F.softmax(
            top_a(logits, min_p_pow=min_p_pow, min_p_ratio=min_p_ratio)
            / temperature,
            dim=-1,
        )
    elif kind == "entmax":
        probs = entmax(logits / temperature, alpha=ENTMAX_ALPHA, dim=-1)
    else:
        raise ValueError(f"Unknown sampling strategy: {kind}")

    return torch.multinomial(probs, 1)


class MusicAutoregressiveWrapper(nn.Module):
    def __init__(self, net, encoding, ignore_index=-100, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        # Get the type codes
        self.sos_type_code = encoding["type_code_map"]["start-of-song"]
        self.eos_type_code = encoding["type_code_map"]["end-of-song"]
        self.son_type_code = encoding["type_code_map"]["start-of-notes"]
        self.instrument_type_code = encoding["type_code_map"]["instrument"]
        self.note_type_code = encoding["type_code_map"]["note"]

        # Get the dimension indices
        self.dimensions = {
            key: encoding["dimensions"].index(key)
            for key in (
                "type",
                "beat",
                "position",
                "pitch",
                "duration",
                "instrument",
                "timesignaturenumerator",
                "timesignaturedenominator",
                "tempo"
            )
        }
        assert self.dimensions["type"] == 0

    @torch.no_grad()
    def generate(
            self,
            start_tokens,  # shape : (b, n, d)
            prompt,
            seq_len,
            eos_token=None,
            temperature=1.0,  # int or list of int
            filter_logits_fn="top_k",  # str or list of str
            filter_thres=0.9,  # int or list of int
            min_p_pow=2.0,
            min_p_ratio=0.02,
            monotonicity_dim=None,
            return_attn=False,
            **kwargs,
    ):
        _, t, dim = start_tokens.shape

        if isinstance(temperature, (float, int)):
            temperature = [temperature] * dim
        elif len(temperature) == 1:
            temperature = temperature * dim
        else:
            assert (
                    len(temperature) == dim
            ), f"`temperature` must be of length {dim}"

        if isinstance(filter_logits_fn, str):
            filter_logits_fn = [filter_logits_fn] * dim
        elif len(filter_logits_fn) == 1:
            filter_logits_fn = filter_logits_fn * dim
        else:
            assert (
                    len(filter_logits_fn) == dim
            ), f"`filter_logits_fn` must be of length {dim}"

        if isinstance(filter_thres, (float, int)):
            filter_thres = [filter_thres] * dim
        elif len(filter_thres) == 1:
            filter_thres = filter_thres * dim
        else:
            assert (
                    len(filter_thres) == dim
            ), f"`filter_thres` must be of length {dim}"

        if isinstance(monotonicity_dim, str):
            monotonicity_dim = [self.dimensions[monotonicity_dim]]
        else:
            monotonicity_dim = [self.dimensions[d] for d in monotonicity_dim]

        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 2:
            start_tokens = start_tokens[None, :, :]

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop("mask", None)

        if mask is None:
            mask = torch.ones(
                (out.shape[0], out.shape[1]),
                dtype=torch.bool,
                device=out.device,
            )

        if monotonicity_dim is not None:
            current_values = {
                d: torch.max(start_tokens[:, :, d], 1)[0]
                for d in monotonicity_dim
            }
        else:
            current_values = None

        instrument_dim = self.dimensions["instrument"]
        type_dim = self.dimensions["type"]
        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]
            # cut the prompt
            if return_attn:
                logits, attn = self.net(
                    x
                )
                logits = [l[:, -1, :] for l in logits]
            else:
                # print(self.net(x, prompt)[0])
                logits = [
                    l[:, -1, :] for l in self.net(x, prompt)[0]
                ]

            # Enforce monotonicity
            if monotonicity_dim is not None and 0 in monotonicity_dim:
                for i, v in enumerate(current_values[0]):
                    logits[0][i, :v] = -float("inf")

            # Filter out sos token
            logits[0][type_dim, 0] = -float("inf")

            # Sample from the logits
            sample_type = sample(
                logits[0],
                filter_logits_fn[0],
                filter_thres[0],
                temperature[0],
                min_p_pow,
                min_p_ratio,
            )

            # Update current values
            if monotonicity_dim is not None and 0 in monotonicity_dim:
                current_values[0] = torch.maximum(
                    current_values[0], sample_type.reshape(-1)
                )

            # Iterate after each sample
            samples = [[s_type] for s_type in sample_type]
            for idx, s_type in enumerate(sample_type):
                # A start-of-song, end-of-song or start-of-notes code
                if s_type in (
                        self.sos_type_code,
                        self.eos_type_code,
                        self.son_type_code,
                ):
                    samples[idx] += [torch.zeros_like(s_type)] * (
                            len(logits) - 1
                    )
                # An instrument code
                elif s_type == self.instrument_type_code:
                    samples[idx] += [torch.zeros_like(s_type)] * (
                            len(logits) - 2
                    )
                    logits[instrument_dim][:, 0] = -float("inf")  # avoid none
                    sampled = sample(
                        logits[instrument_dim][idx: idx + 1],
                        filter_logits_fn[instrument_dim],
                        filter_thres[instrument_dim],
                        temperature[instrument_dim],
                        min_p_pow,
                        min_p_ratio,
                    )[0]
                    samples[idx].append(sampled)
                # A note code
                elif s_type == self.note_type_code:
                    for d in range(1, dim):
                        # Enforce monotonicity
                        if (
                                monotonicity_dim is not None
                                and d in monotonicity_dim
                        ):
                            logits[d][idx, : current_values[d][idx]] = -float(
                                "inf"
                            )

                        # Sample from the logits
                        logits[d][:, 0] = -float("inf")  # avoid none
                        sampled = sample(
                            logits[d][idx: idx + 1],
                            filter_logits_fn[d],
                            filter_thres[d],
                            temperature[d],
                            min_p_pow,
                            min_p_ratio,
                        )[0]
                        samples[idx].append(sampled)

                        # Update current values
                        if (
                                monotonicity_dim is not None
                                and d in monotonicity_dim
                        ):
                            current_values[d][idx] = torch.max(
                                current_values[d][idx], sampled
                            )[0]
                else:
                    raise ValueError(f"Unknown event type code: {s_type}")

            stacked = torch.stack(
                [torch.cat(s).expand(1, -1) for s in samples], 0
            )
            out = torch.cat((out, stacked), dim=1)
            mask = F.pad(mask, (0, 1), value=True)

            if exists(eos_token):
                is_eos_tokens = out[..., 0] == eos_token

                # Mask out everything after the eos tokens
                if is_eos_tokens.any(dim=1).all():
                    for i, is_eos_token in enumerate(is_eos_tokens):
                        idx = torch.argmax(is_eos_token.byte())
                        out[i, idx + 1:] = self.pad_value
                    break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)

        if return_attn:
            return out, attn

        return out

    def forward(self, x, prompt, attribute, return_list=False):
        xi = x[:, :-1]
        xo = x[:, 1:]

        out, out_prompt = self.net(xi, prompt[:, :-1])
        losses = [
            F.cross_entropy(
                out[i].transpose(1, 2),
                xo[..., i],
                ignore_index=self.ignore_index,
            )
            for i in range(len(out))
        ]
        loss = sum(losses)
        if self.net.prompt_enable:
            loss_prompt = F.cross_entropy(out_prompt.transpose(1, 2), attribute[:, :-1])
            loss += loss_prompt
            losses += [loss_prompt]

        if return_list:
            return loss, losses
        return loss


class MusicXTransformer(nn.Module):
    def __init__(self, *, dim, encoding, prompt_enable, **kwargs):
        super().__init__()

        self.decoder = MixerModel(
            encoding=encoding,
            prompt_enable=prompt_enable,
            max_seq_len=kwargs.pop("max_seq_len"),
            d_model=dim,
            n_layer=kwargs.pop("layers")

        )
        self.decoder = MusicAutoregressiveWrapper(
            self.decoder, encoding=encoding
        )

    @torch.no_grad()
    def generate(self, seq_in, prompt, seq_len, **kwargs):
        return self.decoder.generate(seq_in, prompt, seq_len, **kwargs)

    def forward(self, seq, prompt, attribute, return_list=False):
        return self.decoder(seq, prompt, attribute, return_list=return_list)
