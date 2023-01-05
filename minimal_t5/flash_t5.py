import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import flash_attn.flash_attn_triton as flash_attn_triton


@dataclass
class T5Config:
    hidden_dim: int
    ff_dim: int
    num_heads: int
    num_encoder_layers: int
    num_decoder_layers: int
    head_dim: int = 64
    dropout_rate: float = 0.1
    vocab_size: int = 32128
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    pad_token_id: int = 0
    layer_norm_epsilon = 1e-6
    dtype = torch.float32

    @property
    def hidden_size(self):
        return self.hidden_dim

    def to_dict(self):
        return asdict(self)


T5BaseConfig = T5Config(
    hidden_dim=768,
    ff_dim=2048,
    num_heads=12,
    num_encoder_layers=12,
    num_decoder_layers=12,
)
T5LargeConfig = T5Config(
    hidden_dim=1024,
    ff_dim=2816,
    num_heads=16,
    num_encoder_layers=24,
    num_decoder_layers=24,
)
T5XLConfig = T5Config(
    hidden_dim=2048,
    ff_dim=5120,
    num_heads=32,
    num_encoder_layers=24,
    num_decoder_layers=24,
)
T5XXLConfig = T5Config(
    hidden_dim=4096,
    ff_dim=10240,
    num_heads=64,
    num_encoder_layers=24,
    num_decoder_layers=24,
)


class T5Model(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config

        # modules
        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.encoder = T5EncoderStack(config)
        self.decoder = T5DecoderStack(config)
        self.embed_out = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self,
                encoder_input_ids,
                decoder_input_ids):
        """Forward pass (with full decode sequence, intended for training or loss-scoring)

        :param encoder_input_ids: [batch_size, enc_seq_len] (pad with padding_idx)
        :param decoder_input_ids: [batch_size, dec_seq_len]
        :return: logits [batch_size, dec_seq_len]
        """

        # 1) Generate masks
        # encoder / cross-attention mask
        # [batch_size, num_heads=1, q_len=1, kv_len=enc_seq_len]
        encoder_attention_mask = create_encoder_attention_mask(
            encoder_input_ids=encoder_input_ids,
            pad_token_id=self.config.pad_token_id,
            dtype=self.config.dtype,
        )
        # decoder mask
        # [batch_size, num_heads=1, q_len=dec_seq_len, kv_len=dec_seq_len]
        decoder_attention_mask = create_decoder_attention_mask(
            decoder_input_ids=decoder_input_ids,
            dtype=self.config.dtype,
        )

        # 2) Forward pass
        # [batch_size, enc_seq_len, hidden_dim]
        encoder_embedding = self.embed_in(encoder_input_ids)
        # [batch_size, enc_seq_len, hidden_dim]
        encoder_hidden_states = self.encoder(
            hidden_states=encoder_embedding,
            attention_mask=encoder_attention_mask,
        )
        # [batch_size, dec_seq_len, hidden_dim]
        decoder_embedding = self.embed_in(decoder_input_ids)
        # dict(
        #   hidden_states = [batch_size, dec_seq_len, hidden_dim]
        # )
        decoder_out = self.decoder(
            hidden_states=decoder_embedding,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_decoder_attention_mask=encoder_attention_mask,
        )
        # [batch_size, dec_seq_len, vocab_size]
        logits = self.embed_out(decoder_out["hidden_states"])
        return logits

    def init_kv_cache(self, decoder_input_ids, dtype):
        # noinspection GrazieInspection
        """Initialize KV cache for decoding.

        A KV cache consists of a list of dicts (one per layer):
            dict(
              key = [batch_size, kv_seq_len=0, num_heads, head_dim]
              value = [batch_size, kv_seq_len=0, num_heads, head_dim]
            )

        :param decoder_input_ids: [batch_size, dec_seq_len]
        :param dtype: dtype
        :return: 0-length kv_cache
        """
        kv_cache = []
        batch_size = decoder_input_ids.shape[0]
        num_heads = self.config.num_heads
        head_dim = self.config.head_dim
        for layer in self.decoder.layers:
            device = layer.self_layer_norm.weight.device
            kv_cache.append({
                "key": torch.zeros([batch_size, 0, num_heads, head_dim]).to(device=device, dtype=dtype),
                "value": torch.zeros([batch_size, 0, num_heads, head_dim]).to(device=device, dtype=dtype),
            })
        return kv_cache

    def generate(self, encoder_input_ids, generation_length):
        """Generate tokens with efficient caching of KV.

        TODO: Add stopping conditions
        TODO: Add sampling capabilities
        (No planned support for decoder-conditional generation)

        :param encoder_input_ids: [batch_size, enc_seq_len]
        :param generation_length: int
        :return: [batch_size, generation_length]
        """
        batch_size = encoder_input_ids.shape[0]

        # 1) Encoder
        # [batch_size, enc_seq_len, hidden_dim]
        encoder_embedding = self.embed_in(encoder_input_ids)
        # [batch_size, num_heads=1, q_len=1, kv_len=enc_seq_len]
        encoder_attention_mask = create_encoder_attention_mask(
            encoder_input_ids=encoder_input_ids,
            pad_token_id=self.config.pad_token_id,
            dtype=self.config.dtype,
        )
        # [batch_size, enc_seq_len, hidden_dim]
        encoder_hidden_states = self.encoder(
            hidden_states=encoder_embedding,
            attention_mask=encoder_attention_mask,
        )

        # 1.1) Pre-compute KV outputs for encoder hidden states
        precomputed_kv_hidden_states = []
        for layer in self.decoder.layers:
            out_shape = batch_size, encoder_hidden_states.size(1), self.config.num_heads, self.config.head_dim
            precomputed_kv_hidden_states.append({
                "key": layer.cross_attention.k(encoder_hidden_states).view(*out_shape),
                "value": layer.cross_attention.v(encoder_hidden_states).view(*out_shape),
            })

        # 2) Decoder
        # [batch_size, dec_seq_len=1]
        decoder_input_ids = torch.LongTensor(
            [[self.config.pad_token_id]] * batch_size
        ).to(encoder_input_ids.device)
        generated_token_ids_list = [decoder_input_ids]
        # See: init_kv_cache. list[dict]
        kv_cache = self.init_kv_cache(decoder_input_ids, dtype=encoder_hidden_states.dtype)
        for decode_step in range(generation_length):
            # [batch_size, dec_seq_len=1]
            decoder_embedding = self.embed_in(decoder_input_ids)
            # [batch_size=1, num_heads=1, q_len=1, kv_len=1]
            decoder_attention_mask = create_decoder_attention_mask(
                decoder_input_ids=decoder_input_ids,
                dtype=self.config.dtype,
            )
            # dict(
            #   hidden_states = [batch_size, dec_seq_len=decode_step+1, hidden_dim]
            #   kv_cache = list[dict(
            #     key = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
            #     value = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
            #   )]
            # )
            decoder_out = self.decoder(
                hidden_states=decoder_embedding,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=None,  # Use pre-computed KV
                encoder_decoder_attention_mask=encoder_attention_mask,
                precomputed_kv_hidden_states=precomputed_kv_hidden_states,
                kv_cache=kv_cache,
            )
            # [batch_size, dec_seq_len=1, vocab_size]
            logits = self.embed_out(decoder_out["hidden_states"])
            kv_cache = decoder_out["kv_cache"]
            # [batch_size, dec_seq_len=1]
            new_decoder_input_ids = logits.argmax(-1)
            decoder_input_ids = new_decoder_input_ids
            generated_token_ids_list.append(logits.argmax(-1))
        return torch.cat(generated_token_ids_list, dim=1)


class T5EncoderStack(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.relative_attention_bias = T5RelationAttentionBias(
            config,
            bidirectional=True,
        )
        self.layers = nn.ModuleList([
            T5EncoderLayer(config)
            for _ in range(self.config.num_encoder_layers)
        ])
        self.final_layer_norm = T5LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, attention_mask):
        # [batch_size=1, num_heads, enc_seq_len, enc_seq_len]
        position_bias = self.relative_attention_bias(
            query_length=hidden_states.shape[1],
            key_length=hidden_states.shape[1],
        )
        for layer in self.layers:
            # [batch_size, enc_seq_len, hidden_dim]
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                attention_bias=position_bias,
            )
        # [batch_size, enc_seq_len, hidden_dim]
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class T5DecoderStack(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.relative_attention_bias = T5RelationAttentionBias(
            config=config,
            bidirectional=False,
        )
        self.layers = nn.ModuleList([
            T5DecoderLayer(config)
            for _ in range(self.config.num_decoder_layers)
        ])
        self.final_layer_norm = T5LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self,
                hidden_states, attention_mask,
                encoder_hidden_states, encoder_decoder_attention_mask,
                precomputed_kv_hidden_states=None,
                kv_cache=None):
        """
        :param hidden_states: [batch_size, dec_seq_len, hidden_dim]
        :param attention_mask: [batch_size=1, num_heads=1, dec_seq_len, dec_seq_len]
        :param encoder_hidden_states: [batch_size, enc_seq_len, hidden_dim]
        :param encoder_decoder_attention_mask: [batch_size=1, num_heads=1, dec_seq_len=1, enc_seq_len]
        :param precomputed_kv_hidden_states: list[dict] of pre-computed KV encoder hidden states
            for decoding
        :param kv_cache: See init_kv_cache.
            We use the presence of kv_cache to determine if we're generating
        """
        new_kv_cache = []

        if kv_cache:
            kv_len = kv_cache[0]["key"].size(1)
            q_len = hidden_states.size(1)
            assert q_len == 1  # should always be 1 for generation
            # [batch_size=1, num_heads, kv_len + q_len, kv_len + q_len]
            self_position_bias = self.relative_attention_bias(
                query_length=kv_len + q_len,
                key_length=kv_len + q_len,
            )
            # Only take the relevant position biases (i.e. last token)
            self_position_bias = self_position_bias[:, :, -q_len:, :]
        else:
            # [batch_size=1, num_heads, dec_seq_len, dec_seq_len]
            self_position_bias = self.relative_attention_bias(
                query_length=hidden_states.shape[1],
                key_length=hidden_states.shape[1],
            )
        for layer_i, layer in enumerate(self.layers):
            if kv_cache:
                # dict(
                #   key = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
                #   value = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
                # )
                layer_kv_cache = kv_cache[layer_i]
            else:
                layer_kv_cache = None

            if precomputed_kv_hidden_states:
                layer_precomputed_kv_hidden_states = precomputed_kv_hidden_states[layer_i]
            else:
                layer_precomputed_kv_hidden_states = None
            # dict(
            #   hidden_states = [batch_size, dec_seq_len=decode_step+1, hidden_dim]
            #   kv_cache = dict(
            #     key = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
            #     value = [batch_size, num_heads, kv_seq_len=decode_step+1, head_dim]
            #   )
            # )
            layer_out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_decoder_attention_mask=encoder_decoder_attention_mask,
                self_attention_bias=self_position_bias,
                kv_cache=layer_kv_cache,
                precomputed_kv_hidden_states=layer_precomputed_kv_hidden_states,
            )
            hidden_states = layer_out["hidden_states"]
            if kv_cache:
                new_kv_cache.append(layer_out["kv_cache"])
        # [batch_size, enc_seq_len, hidden_dim]
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if kv_cache:
            return {
                "hidden_states": hidden_states,
                "kv_cache": new_kv_cache,
            }
        else:
            return {
                "hidden_states": hidden_states
            }


class T5RelationAttentionBias(nn.Module):
    """Taken from HF implementation."""
    def __init__(self, config, bidirectional):
        super().__init__()
        self.config = config
        self.bidirectional = bidirectional
        self.attention_biases = nn.Embedding(
            config.relative_attention_num_buckets,
            config.num_heads,
        )

    def compute_relative_position_bucket(self, relative_position, num_buckets=32, max_distance=128):
        relative_buckets = 0
        if self.bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def forward(self, query_length, key_length):
        device = self.attention_biases.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self.compute_relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            num_buckets=self.config.relative_attention_num_buckets,
            max_distance=self.config.relative_attention_max_distance,
        )
        values = self.attention_biases(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.hidden_dim, config.ff_dim, bias=False)
        self.wi_1 = nn.Linear(config.hidden_dim, config.ff_dim, bias=False)
        self.wo = nn.Linear(config.ff_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = NewGELUActivation()

    def forward(self, hidden_states):
        # [batch_size, seq_len, ff_dim]
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # [batch_size, seq_len, ff_dim]
        hidden_linear = self.wi_1(hidden_states)
        # [batch_size, seq_len, ff_dim]
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        # [batch_size, seq_len, hidden_dim]
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus variance is calculated
        # w/o mean and there is no bias. Additionally, we want to make sure that the accumulation for
        # half-precision inputs is done in fp32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


class T5FFLayer(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(config)
        self.layer_norm = T5LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        # [batch_size, seq_len, hidden_dim]
        forwarded_states = self.layer_norm(hidden_states)
        # [batch_size, seq_len, hidden_dim]
        forwarded_states = self.DenseReluDense(forwarded_states)
        # [batch_size, seq_len, hidden_dim]
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    # noinspection PyMethodMayBeStatic
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * torch.pow(x, 3.0))
            ))
        )


class T5Attention(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        inner_dim = config.num_heads * config.head_dim
        self.q = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.k = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.v = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.o = nn.Linear(inner_dim, config.hidden_dim, bias=False)

    def forward(self,
                q_hidden_states, kv_hidden_states, attention_bias, attention_mask,
                precomputed_kv_hidden_states=None,
                kv_cache=None):
        assert precomputed_kv_hidden_states is None or kv_cache is None
        batch_size, q_seq_len, hidden_dim = q_hidden_states.shape
        num_heads, head_dim = self.config.num_heads, self.config.head_dim
        # [batch_size, q_seq_len, num_heads, head_dim]
        query_states = self.q(q_hidden_states).view(
            batch_size, q_seq_len, num_heads, head_dim,
        )
        if precomputed_kv_hidden_states:
            assert kv_hidden_states is None
            key_states = precomputed_kv_hidden_states["key"]
            value_states = precomputed_kv_hidden_states["value"]
        else:
            _, kv_seq_len, _ = kv_hidden_states.shape
            key_states = self.k(kv_hidden_states).view(
                batch_size, kv_seq_len, num_heads, head_dim,
            )
            value_states = self.v(kv_hidden_states).view(
                batch_size, kv_seq_len, num_heads, head_dim,
            )
            if kv_cache:
                key_states = torch.cat([kv_cache["key"], key_states], dim=1)
                value_states = torch.cat([kv_cache["value"], value_states], dim=1)

        if attention_bias is None:
            # Enc-Dec
            final_attn_bias = attention_mask
        else:
            final_attn_bias = attention_bias + attention_mask
        attn_output = compute_flash_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_bias=final_attn_bias,
        )

        attn_output = attn_output.contiguous().view(
            batch_size, q_seq_len, hidden_dim,
        )
        attn_output = self.o(attn_output)
        if kv_cache:
            new_kv_cache = {"key": key_states, "value": value_states}
            return {"attn_output": attn_output, "kv_cache": new_kv_cache}
        else:
            return {"attn_output": attn_output}


class T5EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_norm = T5LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.self_attention = T5Attention(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.ffn = T5FFLayer(config)

    def forward(
            self,
            hidden_states,
            attention_mask,
            attention_bias,
    ):
        # [batch_size, enc_seq_len, hidden_dim]
        normed_hidden_states = self.layer_norm(hidden_states)
        # dict(
        #   attn_output = [batch_size, dec_seq_len=decode_step+1, hidden_dim]
        # )
        attn_output = self.self_attention(
            q_hidden_states=normed_hidden_states,
            kv_hidden_states=normed_hidden_states,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
        )["attn_output"]
        # [batch_size, enc_seq_len, hidden_dim]
        hidden_states = hidden_states + self.dropout(attn_output)
        # [batch_size, enc_seq_len, hidden_dim]
        hidden_states = stabilize_hidden_states(hidden_states)
        # [batch_size, enc_seq_len, hidden_dim]
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class T5DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.self_layer_norm = T5LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.self_attention = T5Attention(config)
        self.self_dropout = nn.Dropout(config.dropout_rate)
        self.cross_layer_norm = T5LayerNorm(config.hidden_dim, eps=config.layer_norm_epsilon)
        self.cross_attention = T5Attention(config)
        self.cross_dropout = nn.Dropout(config.dropout_rate)
        self.ffn = T5FFLayer(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states,
        encoder_decoder_attention_mask,
        self_attention_bias,
        precomputed_kv_hidden_states=None,
        kv_cache=None,
    ):
        # 1) Self-attention
        # [batch_size, dec_seq_len, hidden_dim]
        normed_hidden_states = self.self_layer_norm(hidden_states)
        # dict(
        #   attn_output = [batch_size, dec_seq_len=dec_seq_len, hidden_dim]
        #   kv_cache = dict(
        #     key = [batch_size, num_heads, kv_seq_len, head_dim]
        #     value = [batch_size, num_heads, kv_seq_len, head_dim]
        #   )
        # )
        raw_self_attn_output = self.self_attention(
            q_hidden_states=normed_hidden_states,
            kv_hidden_states=normed_hidden_states,
            attention_mask=attention_mask,
            attention_bias=self_attention_bias,
            kv_cache=kv_cache,
        )
        attn_output = raw_self_attn_output["attn_output"]
        # [batch_size, dec_seq_len, hidden_dim]
        hidden_states = hidden_states + self.self_dropout(attn_output)
        hidden_states = stabilize_hidden_states(hidden_states)

        # 2) Cross-attn
        # [batch_size, dec_seq_len, hidden_dim]
        normed_hidden_states = self.cross_layer_norm(hidden_states)
        # dict(
        #   attn_output = [batch_size, dec_seq_len=dec_seq_len, hidden_dim]
        # )
        attn_output = self.cross_attention(
            q_hidden_states=normed_hidden_states,
            kv_hidden_states=encoder_hidden_states,
            attention_mask=encoder_decoder_attention_mask,
            precomputed_kv_hidden_states=precomputed_kv_hidden_states,
            attention_bias=None,
        )["attn_output"]
        # [batch_size, dec_seq_len, hidden_dim]
        hidden_states = hidden_states + self.cross_dropout(attn_output)
        hidden_states = stabilize_hidden_states(hidden_states)

        # 3) FFN
        # [batch_size, dec_seq_len, hidden_dim]
        hidden_states = self.ffn(hidden_states)
        if kv_cache:
            return {
                "hidden_states": hidden_states,
                "kv_cache": raw_self_attn_output["kv_cache"],
            }
        else:
            return {
                "hidden_states": hidden_states
            }


def stabilize_hidden_states(hidden_states):
    if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    return hidden_states


def convert_hf_state_dict(hf_state_dict, config: T5Config):
    # noinspection PyDictCreation
    state_dict = {}

    # Encoder
    state_dict["embed_in.weight"] = hf_state_dict["shared.weight"]
    state_dict["encoder.relative_attention_bias.attention_biases.weight"] = \
        hf_state_dict["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]
    state_dict["encoder.final_layer_norm.weight"] = \
        hf_state_dict["encoder.final_layer_norm.weight"]
    for layer_i in range(config.num_encoder_layers):
        prefix = f"encoder.layers.{layer_i}"
        in_prefix = f"encoder.block.{layer_i}.layer"
        state_dict[f"{prefix}.layer_norm.weight"] = \
            hf_state_dict[f"{in_prefix}.0.layer_norm.weight"]
        state_dict[f"{prefix}.self_attention.q.weight"] = \
            hf_state_dict[f"{in_prefix}.0.SelfAttention.q.weight"]
        state_dict[f"{prefix}.self_attention.k.weight"] = \
            hf_state_dict[f"{in_prefix}.0.SelfAttention.k.weight"]
        state_dict[f"{prefix}.self_attention.v.weight"] = \
            hf_state_dict[f"{in_prefix}.0.SelfAttention.v.weight"]
        state_dict[f"{prefix}.self_attention.o.weight"] = \
            hf_state_dict[f"{in_prefix}.0.SelfAttention.o.weight"]
        state_dict[f"{prefix}.ffn.DenseReluDense.wi_0.weight"] = \
            hf_state_dict[f"{in_prefix}.1.DenseReluDense.wi_0.weight"]
        state_dict[f"{prefix}.ffn.DenseReluDense.wi_1.weight"] = \
            hf_state_dict[f"{in_prefix}.1.DenseReluDense.wi_1.weight"]
        state_dict[f"{prefix}.ffn.DenseReluDense.wo.weight"] = \
            hf_state_dict[f"{in_prefix}.1.DenseReluDense.wo.weight"]
        state_dict[f"{prefix}.ffn.layer_norm.weight"] = \
            hf_state_dict[f"{in_prefix}.1.layer_norm.weight"]

    # Decoder
    state_dict["decoder.relative_attention_bias.attention_biases.weight"] = \
        hf_state_dict["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]
    state_dict["decoder.final_layer_norm.weight"] = \
        hf_state_dict["decoder.final_layer_norm.weight"]
    for layer_i in range(config.num_decoder_layers):
        prefix = f"decoder.layers.{layer_i}"
        in_prefix = f"decoder.block.{layer_i}.layer"
        state_dict[f"{prefix}.self_layer_norm.weight"] = \
            hf_state_dict[f"{in_prefix}.0.layer_norm.weight"]
        state_dict[f"{prefix}.self_attention.q.weight"] = \
            hf_state_dict[f"{in_prefix}.0.SelfAttention.q.weight"]
        state_dict[f"{prefix}.self_attention.k.weight"] = \
            hf_state_dict[f"{in_prefix}.0.SelfAttention.k.weight"]
        state_dict[f"{prefix}.self_attention.v.weight"] = \
            hf_state_dict[f"{in_prefix}.0.SelfAttention.v.weight"]
        state_dict[f"{prefix}.self_attention.o.weight"] = \
            hf_state_dict[f"{in_prefix}.0.SelfAttention.o.weight"]
        state_dict[f"{prefix}.cross_layer_norm.weight"] = \
            hf_state_dict[f"{in_prefix}.1.layer_norm.weight"]
        state_dict[f"{prefix}.cross_attention.q.weight"] = \
            hf_state_dict[f"{in_prefix}.1.EncDecAttention.q.weight"]
        state_dict[f"{prefix}.cross_attention.k.weight"] = \
            hf_state_dict[f"{in_prefix}.1.EncDecAttention.k.weight"]
        state_dict[f"{prefix}.cross_attention.v.weight"] = \
            hf_state_dict[f"{in_prefix}.1.EncDecAttention.v.weight"]
        state_dict[f"{prefix}.cross_attention.o.weight"] = \
            hf_state_dict[f"{in_prefix}.1.EncDecAttention.o.weight"]
        state_dict[f"{prefix}.ffn.DenseReluDense.wi_0.weight"] = \
            hf_state_dict[f"{in_prefix}.2.DenseReluDense.wi_0.weight"]
        state_dict[f"{prefix}.ffn.DenseReluDense.wi_1.weight"] = \
            hf_state_dict[f"{in_prefix}.2.DenseReluDense.wi_1.weight"]
        state_dict[f"{prefix}.ffn.DenseReluDense.wo.weight"] = \
            hf_state_dict[f"{in_prefix}.2.DenseReluDense.wo.weight"]
        state_dict[f"{prefix}.ffn.layer_norm.weight"] = \
            hf_state_dict[f"{in_prefix}.2.layer_norm.weight"]

    state_dict["embed_out.weight"] = hf_state_dict["lm_head.weight"]
    return state_dict


def get_hf_state_dict(hf_model_name, model_source=None):
    import transformers
    if model_source is None:
        model_source = hf_model_name
    config = get_config(hf_model_name)
    hf_model = transformers.T5ForConditionalGeneration.from_pretrained(model_source)
    loaded_state_dict = hf_model.state_dict()
    return convert_hf_state_dict(hf_state_dict=loaded_state_dict, config=config)


def get_config(hf_model_name) -> T5Config:
    if hf_model_name.startswith("google/"):
        hf_model_name = hf_model_name.replace("google/", "")
    return {
        "t5-base-lm-adapt": T5BaseConfig,
        "t5-large-lm-adapt": T5LargeConfig,
        "t5-xl-lm-adapt": T5XLConfig,
        "t5-xxl-lm-adapt": T5XXLConfig,
        "flan-t5-base": T5BaseConfig,
        "flan-t5-large": T5LargeConfig,
        "flan-t5-xl": T5XLConfig,
        "flan-t5-xxl": T5XXLConfig,
    }[hf_model_name]


def create_encoder_attention_mask(encoder_input_ids,
                                  pad_token_id=0,
                                  dtype=torch.float32,
                                  return_soft_mask=True):
    """Create mask for encoder (and encoder-decoder) attention.

    1) Note that we can broadcast over the q_len, since all tokens can
       attend to the same tokens in a given example.
    2) We can also broadcast over num_heads.
    3) Because of this broadcasting, the encoder attention mask also works for
       decoder attention (all tokens can attend to all valid encoder tokens)

    :param encoder_input_ids: [batch_size, seq_len]
    :param pad_token_id: int
    :param dtype: dtype
    :param return_soft_mask: whether to return mask or logits-mask
    :return: float [batch_size=1, num_heads=1, q_len=1, kv_len=seq_len]
    """
    # [batch_size, seq_len]
    is_valid_token = (encoder_input_ids != pad_token_id).to(dtype=dtype)
    # [batch_size, num_heads=1, q_len=1, seq_len]
    attention_mask = is_valid_token[:, None, None, :]
    if return_soft_mask:
        return convert_mask_to_soft_mask(attention_mask, dtype=dtype)
    else:
        return attention_mask


def create_decoder_attention_mask(decoder_input_ids,
                                  dtype=torch.float32,
                                  return_soft_mask=True):
    """Create mask for decoder attention.

    Decoder masks have two use-cases:

    1) Training, where we see the full decoder sequence. In that case,
       we want a causal mask.

    2) Generation, where we only see one token at once. In that case,
       it doesn't really matter what we give, we can just give a 1.
       (i.e. seq_len = 1)

    Note that in both cases we do not care about which decoder_input_ids
    are valid, and also we can always simply broadcast over the batch size
    and heads.

    :param decoder_input_ids: [batch_size, seq_len]
    :param dtype: dtype
    :param return_soft_mask: whether to return mask or logits-mask
    :return: float [batch_size=1, num_heads=1, q_len=seq_len, kv_len=seq_len]
    """
    batch_size, seq_length = decoder_input_ids.shape
    # [seq_len]
    seq_ids = torch.arange(seq_length, device=decoder_input_ids.device)
    # [seq_len, seq_len]
    causal_mask = seq_ids[None, :].repeat(seq_length, 1) <= seq_ids[:, None]
    # [batch_size=1, num_heads=1, seq_len, seq_len]
    causal_mask = causal_mask[None, None, :, :]
    if return_soft_mask:
        return convert_mask_to_soft_mask(causal_mask, dtype=dtype)
    else:
        return causal_mask


def convert_mask_to_soft_mask(mask, dtype):
    """Convert binary mask to mask that can be added to logits.

    (i.e. 0 for attention, large negative for masked)
    """
    mask = mask.to(dtype=dtype)
    mask = (1.0 - mask) * torch.finfo(dtype).min
    return mask


def shift_right(input_ids):
    decoder_start_token_id = 0
    pad_token_id = 0

    assert decoder_start_token_id is not None, (
        "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
        " See T5 docs for more information"
    )
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

    return shifted_input_ids


def compute_logits_loss(lm_logits, labels):
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(lm_logits.view(
        -1, lm_logits.size(-1)), labels.view(-1)
    )
    return loss


def compute_flash_attention(query_states, key_states, value_states, attention_bias):
    """Flash Attention (Triton version)

    :param query_states: [batch_size, q_seq_len, num_heads, head_size]
    :param key_states: [batch_size, kv_seq_len, num_heads, head_size]
    :param value_states: [batch_size, kv_seq_len, num_heads, head_size]
    :param attention_bias: [batch_size, num_heads/1, q_seq_len/1, kv_seq_len/1]
    :return: attn_out: [batch_size, q_seq_len, num_heads, head_size]
    """
    batch_size, q_seq_len, num_heads, _ = query_states.shape
    batch_size, kv_seq_len, _, _ = key_states.shape
    attention_bias = attention_bias.expand(batch_size, num_heads, q_seq_len, kv_seq_len)
    return flash_attn_triton.flash_attn_func(
        query_states,
        key_states,
        value_states,
        attention_bias,
        False,
        1.0,
    )
