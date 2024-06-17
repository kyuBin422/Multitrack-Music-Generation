import math
import pathlib

import torch
import torch.nn.functional as F
from fairscale.nn import checkpoint_wrapper, wrap
from torch import nn
from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import DecoderLayer, RetNetRelPos
from torchscale.component.rms_norm import RMSNorm
from x_transformers.autoregressive_wrapper import (
    ENTMAX_ALPHA,
    entmax,
    exists,
    top_a,
    top_k,
    top_p,
)
from x_transformers.x_transformers import (
    TokenEmbedding,
    exists,
)


class RetNetDecoder(nn.Module):
    def __init__(
            self,
            args,
            encoding,
            max_seq_len,
            embed_tokens=None,
            prompt_enable=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.prompt_enable = prompt_enable
        self.args = args
        self.decoder_value_embed_dim = self.args.decoder_value_embed_dim

        self.dropout_module = torch.nn.Dropout(args.dropout)

        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)
        if self.prompt_enable:
            # LLAMA_checkpoint = torch.load(pathlib.Path('model/7B/consolidated.00.pth'))
            # self.embed_tokens_prompt = nn.Embedding.from_pretrained(
            #     LLAMA_checkpoint['tok_embeddings.weight'][:, :embed_dim])
            # self.embed_tokens_prompt.weight.requires_grad = prompt_enable
            self.embed_tokens_prompt = TokenEmbedding(embed_dim, 32000)

        self.embed_tokens = embed_tokens
        n_tokens = encoding["n_tokens"]
        self.embed_tokens = nn.ModuleList(
            [
                TokenEmbedding(embed_dim, n)
                for n in n_tokens
            ]
        )
        self.output_projection = self.build_output_projection(args, encoding)
        if args.layernorm_embedding:
            self.layernorm_embedding = RMSNorm(embed_dim, eps=args.layernorm_eps)
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([])

        moe_freq = args.moe_freq
        for i in range(args.decoder_layers):
            is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
            self.layers.append(
                self.build_decoder_layer(
                    args,
                    depth=i,
                    is_moe_layer=is_moe_layer,
                )
            )

        self.num_layers = len(self.layers)

        if args.decoder_normalize_before:
            self.layer_norm = RMSNorm(embed_dim, eps=args.layernorm_eps)
        else:
            self.layer_norm = None

        self.retnet_rel_pos = RetNetRelPos(args)
        self.chunkwise_recurrent = args.chunkwise_recurrent
        self.recurrent_chunk_size = args.recurrent_chunk_size

        if args.deepnorm:
            init_scale = math.pow(8.0 * args.decoder_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                        "fc1" in name
                        or "fc2" in name
                        or "out_proj" in name
                        or "v_proj" in name
                ):
                    p.data.div_(init_scale)
        if self.prompt_enable:
            self.output_projection_prompt = nn.Linear(embed_dim, 32000)

    def build_output_projection(
            self,
            args,
            encoding,
    ):
        if args.share_decoder_input_output_embed:
            output_projection = torch.nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = self.embed_tokens.weight
        else:
            n_tokens = encoding["n_tokens"]
            output_projection = nn.ModuleList([nn.Linear(args.decoder_embed_dim, n, bias=False) for n in n_tokens])
            for output in output_projection:
                torch.nn.init.normal_(
                    output.weight, mean=0, std=args.decoder_embed_dim ** -0.5
                )
        return output_projection

    def build_decoder_layer(
            self, args, depth, is_moe_layer=False
    ):
        layer = DecoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer

    def forward_embedding(
            self,
            tokens,
            prompt,
            token_embedding=None,
            incremental_state=None,
    ):
        if incremental_state is not None and not self.is_first_step(incremental_state):
            tokens = tokens[:, -1:]

        if token_embedding is None:
            token_embedding = sum(
                emb(tokens[..., i]) for i, emb in enumerate(self.embed_tokens)
            )
        if self.prompt_enable:
            token_embedding = token_embedding + self.embed_tokens_prompt(prompt)
        x = embed = self.embed_scale * token_embedding

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        return x, embed

    def is_first_step(self, incremental_state):
        if incremental_state is None:
            return False
        return incremental_state.get("is_first_step", False)

    def forward(
            self,
            prev_output_tokens,
            prompt,
            incremental_state=None,
            features_only=False,
            return_all_hiddens=False,
            token_embeddings=None,
            **kwargs
    ):
        # embed tokens
        x, _ = self.forward_embedding(
            prev_output_tokens, prompt, token_embeddings, incremental_state
        )
        is_first_step = self.is_first_step(incremental_state)

        if self.chunkwise_recurrent and prev_output_tokens.size(1) % self.recurrent_chunk_size != 0:
            padding_len = self.recurrent_chunk_size - prev_output_tokens.size(1) % self.recurrent_chunk_size
            slen = prev_output_tokens.size(1) + padding_len
            x = F.pad(x, (0, 0, 0, padding_len))
        else:
            slen = prev_output_tokens.size(1)
        # relative position
        retention_rel_pos = self.retnet_rel_pos(slen, incremental_state is not None and not is_first_step,
                                                chunkwise_recurrent=self.chunkwise_recurrent)
        # decoder layers
        inner_states = [x]

        l_aux = []

        for idx, layer in enumerate(self.layers):
            if incremental_state is None or is_first_step:
                if is_first_step and incremental_state is not None:
                    if idx not in incremental_state:
                        incremental_state[idx] = {}
            else:
                if idx not in incremental_state:
                    incremental_state[idx] = {}

            x, l_aux_i = layer(
                x,
                incremental_state[idx] if incremental_state is not None else None,
                retention_rel_pos=retention_rel_pos,
                chunkwise_recurrent=self.chunkwise_recurrent,
            )
            l_aux.append(l_aux_i)
            inner_states.append(x)

        if self.chunkwise_recurrent and prev_output_tokens.size(1) % self.recurrent_chunk_size != 0:
            x = x[:, :prev_output_tokens.size(1), :]

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if not features_only:
            x = self.output_layer(x)

        return x, {
            "inner_states": inner_states,
            "l_aux": l_aux,
            "attn": None,
        }

    def output_layer(self, features):
        out_prompt = None
        if self.prompt_enable:
            out_prompt = self.output_projection_prompt(features)
        return [to_logit(features) for to_logit in self.output_projection], out_prompt


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
            prompt = prompt[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]
            # cut the prompt
            if return_attn:
                logits, attn = self.net(
                    x
                )
                logits = [l[:, -1, :] for l in logits]
            else:
                logits = [
                    l[:, -1, :] for l in self.net(x, prompt)[0][0]
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

        (out, out_prompt), _ = self.net(xi, prompt[:, :-1])
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

        self.config = RetNetConfig(decoder_embed_dim=dim,
                                   decoder_retention_heads=kwargs.pop("heads"),
                                   decoder_layers=kwargs.pop("layers"),
                                   dropout=kwargs.pop("emb_dropout"),
                                   )
        self.decoder = RetNetDecoder(
            self.config,
            encoding=encoding,
            max_seq_len=kwargs.pop("max_seq_len"),
            prompt_enable=prompt_enable,
        )
        self.decoder = MusicAutoregressiveWrapper(
            self.decoder, encoding=encoding
        )

    @torch.no_grad()
    def generate(self, seq_in, prompt, seq_len, **kwargs):
        return self.decoder.generate(seq_in, prompt, seq_len, **kwargs)

    def forward(self, seq, prompt, attribute, return_list=False):
        return self.decoder(seq, prompt, attribute, return_list=return_list)
