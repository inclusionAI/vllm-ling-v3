# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from transformers import AutoConfig

from vllm.transformers_utils.config import (
    _CONFIG_REGISTRY,
    get_config,
    uses_mrope,
)
from vllm.transformers_utils.configs.bailing_moe_v3_vl import (
    BailingMoeV3TextConfig,
    BailingMoeV3VisionConfig,
    BailingMoeV3VLConfig,
)


def _bailing_v3_vl_config() -> dict[str, Any]:
    return {
        "architectures": ["BailingMoeV3VLForConditionalGeneration"],
        "auto_map": {
            "AutoConfig": ("configuration_bailing_moe_v3_vl.BailingMoeV3VLConfig")
        },
        "model_type": "bailing_moe_v3_vl",
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
        "norm_query_embeds": False,
        "image_token_id": 157157,
        "video_token_id": 156909,
        "vision_start_token_id": 157158,
        "vision_end_token_id": 157159,
        "mrope_section": [8, 12, 12],
        "text_config": {
            "vocab_size": 157184,
            "hidden_size": 2560,
            "intermediate_size": 6144,
            "num_hidden_layers": 42,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "head_dim": 128,
            "max_position_embeddings": 131072,
            "layer_group_size": 6,
            "rope_theta": 6_000_000,
            "partial_rotary_factor": 0.5,
            "rms_norm_eps": 1e-6,
            "num_experts": 512,
            "num_experts_per_tok": 8,
            "kv_lora_rank": 512,
            "no_kda_lora": True,
        },
        "vision_config": {
            "model_type": "qwen3_moe_vit",
            "depth": 27,
            "hidden_size": 1152,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 4304,
            "num_heads": 16,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 4096,
            "num_position_embeddings": 2304,
            "disable_merger_proj": True,
        },
    }


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_bailing_v3_vl_config_loads_without_remote_code(tmp_path: Path):
    model_path = tmp_path / "model"
    _write_config(model_path, _bailing_v3_vl_config())

    assert _CONFIG_REGISTRY["bailing_moe_v3_vl"] is BailingMoeV3VLConfig
    config = get_config(model_path, trust_remote_code=False)

    assert isinstance(config, BailingMoeV3VLConfig)
    assert isinstance(config.text_config, BailingMoeV3TextConfig)
    assert isinstance(config.vision_config, BailingMoeV3VisionConfig)
    assert isinstance(
        AutoConfig.from_pretrained(model_path, trust_remote_code=False),
        BailingMoeV3VLConfig,
    )
    assert config.architectures == ["BailingMoeV3VLForConditionalGeneration"]
    assert config.image_token_id == 157157
    assert config.mrope_section == [8, 12, 12]
    assert config.norm_query_embeds is False
    assert uses_mrope(config)

    text_config = config.text_config
    assert text_config.model_type == "bailing_hybrid"
    assert text_config.architectures == ["BailingMoeV3ForCausalLM"]
    assert text_config.rope_parameters["mrope_section"] == [8, 12, 12]
    assert text_config.rope_parameters["rope_theta"] == 6_000_000
    assert text_config.use_bias is False
    assert text_config.num_shared_experts == 1
    assert text_config.num_experts == 512
    assert text_config.kv_lora_rank == 512
    assert text_config.no_kda_lora is True
    assert text_config.layer_types.count("linear_attention") == 35
    assert [
        layer_idx
        for layer_idx, layer_type in enumerate(text_config.layer_types)
        if layer_type == "full_attention"
    ] == [5, 11, 17, 23, 29, 35, 41]
    assert config.vision_config.disable_merger_proj is True

    normalized = config.to_dict()
    restored = BailingMoeV3VLConfig.from_dict(normalized)
    assert isinstance(restored.text_config, BailingMoeV3TextConfig)
    assert isinstance(restored.vision_config, BailingMoeV3VisionConfig)
    assert restored.to_dict() == normalized


def test_bailing_v3_vl_config_preserves_explicit_nested_values():
    config_dict = _bailing_v3_vl_config()
    config_dict["architectures"] = ["CustomBailingVLForConditionalGeneration"]
    text_config = config_dict["text_config"]
    text_config["architectures"] = ["CustomBailingForCausalLM"]
    text_config["layer_types"] = [
        "full_attention" if (layer_idx + 1) % 6 == 0 else "linear_attention"
        for layer_idx in range(42)
    ]
    text_config["rope_theta"] = 1_000_000
    text_config["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": 1_000_000,
        "mrope_section": [8, 12, 12],
    }
    original = copy.deepcopy(config_dict)

    config = BailingMoeV3VLConfig(**config_dict)

    assert config_dict == original
    assert config.architectures == ["CustomBailingVLForConditionalGeneration"]
    assert config.text_config.architectures == ["CustomBailingForCausalLM"]
    assert config.text_config.layer_types == text_config["layer_types"]
    assert config.text_config.rope_parameters["rope_theta"] == 1_000_000
    assert config.text_config.rope_parameters["mrope_section"] == [8, 12, 12]


def test_bailing_v3_vl_config_normalizes_config_instance():
    text_config = BailingMoeV3TextConfig()

    config = BailingMoeV3VLConfig(
        text_config=text_config,
        mrope_section=[8, 12, 12],
    )

    assert config.text_config is text_config
    assert text_config.rope_parameters["mrope_section"] == [8, 12, 12]


def test_bailing_v3_vl_config_defaults_and_trailing_layers():
    config = BailingMoeV3VLConfig()
    text_config = BailingMoeV3TextConfig(
        num_hidden_layers=8,
        layer_group_size=3,
    )

    assert config.mrope_section == [12, 10, 10]
    assert config.image_token_id == 151655
    assert config.video_token_id == 151656
    assert config.vision_start_token_id == 151652
    assert config.vision_end_token_id == 151653
    assert text_config.layer_types == [
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "full_attention",
        "full_attention",
    ]

    explicit_block_types = [
        "mamba",
        "mamba",
        "attention",
        "mamba",
        "mamba",
        "attention",
        "attention",
        "attention",
    ]
    text_config = BailingMoeV3TextConfig(
        num_hidden_layers=8,
        layer_group_size=3,
        layers_block_type=explicit_block_types,
    )
    assert text_config.layer_types == [
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "full_attention",
        "full_attention",
    ]


def test_bailing_v3_text_config_accepts_legacy_rope_type():
    text_config = BailingMoeV3TextConfig(rope_scaling={"type": "linear", "factor": 2.0})

    assert text_config.rope_parameters["type"] == "linear"
    assert text_config.rope_parameters["rope_type"] == "linear"


def test_bailing_v3_vl_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="layer_group_size must be positive"):
        BailingMoeV3TextConfig(layer_group_size=0)

    with pytest.raises(ValueError, match="must match the layer_group_size schedule"):
        BailingMoeV3TextConfig(
            num_hidden_layers=8,
            layer_group_size=3,
            layer_types=["full_attention"] * 8,
        )

    with pytest.raises(ValueError, match="conflicting M-RoPE sections"):
        BailingMoeV3VLConfig(
            text_config={
                "rope_parameters": {
                    "rope_type": "default",
                    "mrope_section": [4, 14, 14],
                }
            },
            mrope_section=[8, 12, 12],
        )
