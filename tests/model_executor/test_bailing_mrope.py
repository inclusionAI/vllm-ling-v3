# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.bailing_mrope import (
    BailingMRotaryEmbedding,
)
from vllm.platforms import current_platform


def test_bailing_mrope_selects_alternating_spatial_then_temporal():
    with set_current_vllm_config(VllmConfig()):
        rotary_emb = BailingMRotaryEmbedding(
            head_size=12,
            rotary_dim=12,
            max_position_embeddings=8,
            base=10_000,
            is_neox_style=False,
            dtype=torch.float32,
            mrope_section=[2, 2, 2],
        )

    positions = torch.tensor([[1, 2], [3, 4], [5, 6]])
    frequencies = torch.arange(6).expand(8, -1)
    position_values = torch.arange(8).unsqueeze(1) * 10
    cos = position_values + frequencies
    sin = cos + 100
    cos_sin_cache = torch.cat((cos, sin), dim=-1)

    actual_cos, actual_sin = rotary_emb.select_cos_sin(
        positions,
        cos_sin_cache,
    )

    expected_cos = torch.tensor(
        [
            [30, 51, 32, 53, 14, 15],
            [40, 61, 42, 63, 24, 25],
        ]
    )
    torch.testing.assert_close(actual_cos, expected_cos)
    torch.testing.assert_close(actual_sin, expected_cos + 100)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Skipping CUDA/ROCm-only test.",
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("position_kind", ["text", "multimodal"])
@torch.inference_mode()
def test_bailing_mrope_cuda_matches_native_for_strided_mla_views(
    dtype: torch.dtype,
    position_kind: str,
):
    device = torch.device("cuda")
    num_tokens = 17
    num_query_heads = 8
    with set_current_vllm_config(VllmConfig()):
        rotary_emb = BailingMRotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=64,
            base=6_000_000,
            is_neox_style=False,
            dtype=dtype,
            mrope_section=[8, 12, 12],
        ).to(device)

    torch.manual_seed(0)
    multimodal_positions = torch.randint(
        0,
        64,
        (3, 2 * num_tokens),
        dtype=torch.int32,
        device=device,
    )[:, ::2]
    if position_kind == "multimodal":
        positions = multimodal_positions
        reference_positions = positions
    else:
        positions = torch.randint(
            0,
            64,
            (2 * num_tokens,),
            dtype=torch.int64,
            device=device,
        )[::2]
        reference_positions = positions.expand(3, -1)
    query_buffer = torch.randn(
        num_tokens,
        num_query_heads,
        192,
        dtype=dtype,
        device=device,
    )
    key_buffer = torch.randn(
        num_tokens,
        576,
        dtype=dtype,
        device=device,
    )
    query = query_buffer[..., 128:]
    key = key_buffer[:, 512:].unsqueeze(1)
    assert not query.is_contiguous()
    assert not key.is_contiguous()

    query_prefix = query_buffer[..., :128].clone()
    key_prefix = key_buffer[:, :512].clone()
    expected_query, expected_key = rotary_emb.forward_native(
        reference_positions,
        query.clone(),
        key.clone(),
    )
    actual_query, actual_key = rotary_emb.forward_cuda(positions, query, key)
    assert expected_key is not None
    assert actual_key is not None

    torch.testing.assert_close(actual_query, expected_query, atol=1e-2, rtol=1.6e-2)
    torch.testing.assert_close(actual_key, expected_key, atol=1e-2, rtol=1.6e-2)
    assert torch.equal(query_buffer[..., :128], query_prefix)
    assert torch.equal(key_buffer[:, :512], key_prefix)
    if position_kind == "multimodal":
        assert actual_query.data_ptr() == query.data_ptr()
        assert actual_key.data_ptr() == key.data_ptr()
