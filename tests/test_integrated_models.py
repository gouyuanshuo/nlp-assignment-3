"""Synthetic interface and mask regressions, not MOSEI experimental results."""

from dataclasses import asdict

import pytest
import torch

from mosei.models import ModelConfig, build_model

DIMS = {"text": 5, "audio": 3, "vision": 2}
NEW_MODELS = ("almt", "graph_mfn")


def config(name, **options):
    values = dict(
        name=name,
        d_model=8,
        hidden_dim=8,
        n_heads=2,
        ff_dim=16,
        dropout=0,
        almt_token_count=2,
        almt_max_input_length=12,
        almt_fusion_depth=1,
        graph_mfn_heads=2,
    )
    values.update(options)
    return ModelConfig(**values)


def batch():
    generator = torch.Generator().manual_seed(41)
    features = {m: torch.randn(3, 4, d, generator=generator) for m, d in DIMS.items()}
    masks = {
        "text": torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0]]).bool(),
        "audio": torch.tensor([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]]).bool(),
        "vision": torch.tensor([[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]).bool(),
    }
    return features, masks


@pytest.mark.parametrize("name", NEW_MODELS)
@pytest.mark.parametrize("missing", [None, "text", "audio", "vision", "all"])
def test_forward_backward_with_partial_and_missing_streams(name, missing):
    torch.manual_seed(3)
    model = build_model(config(name), DIMS)
    features, masks = batch()
    for modality in DIMS:
        if missing in (modality, "all"):
            masks[modality][:] = False
        features[modality].requires_grad_()
    logits = model(features, masks)
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for modality in DIMS:
        assert torch.isfinite(features[modality].grad).all()
        assert features[modality].grad[~masks[modality]].count_nonzero() == 0


@pytest.mark.parametrize("name", NEW_MODELS)
@pytest.mark.parametrize("junk", [1e20, float("nan"), float("inf")])
def test_padding_junk_and_extra_padding_cannot_change_logits_or_gradients(name, junk):
    torch.manual_seed(9)
    model = build_model(config(name), DIMS).eval()
    features, masks = batch()
    expected = model(features, masks)
    altered = {m: x.masked_fill(~masks[m][..., None], junk) for m, x in features.items()}
    actual = model(altered, masks)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    extended = {
        m: torch.cat((x, torch.full_like(x[:, :2], junk)), dim=1) for m, x in altered.items()
    }
    extended_masks = {
        m: torch.cat((mask, torch.zeros(3, 2, dtype=torch.bool)), dim=1)
        for m, mask in masks.items()
    }
    torch.testing.assert_close(model(extended, extended_masks), expected, atol=1e-6, rtol=1e-5)


def test_almt_ahl_really_uses_audio_and_vision_instead_of_masking_every_latent():
    # The teammate code constructed all_valid=zeros then passed ~all_valid,
    # silently masking every audio/vision key. Keep the AHL path observable.
    torch.manual_seed(4)
    model = build_model(config("almt", almt_use_graph_fusion=False), DIMS)
    features, masks = batch()
    logits = model(features, masks)
    logits.square().sum().backward()
    for layer in model.ahl_layers:
        for attention in (layer.language_to_audio, layer.language_to_visual):
            assert attention.in_proj_weight.grad.abs().sum() > 0
    changed = {m: x.clone() for m, x in features.items()}
    changed["audio"][masks["audio"]] += 7
    assert not torch.allclose(model(changed, masks), logits)


@pytest.mark.parametrize(
    "option",
    ["modality_embeddings", "almt_use_ahl", "almt_use_cross_modal_fusion", "almt_use_graph_fusion"],
)
def test_almt_component_ablations_preserve_interface_and_missing_input_safety(option):
    model = build_model(config("almt", **{option: False}), DIMS)
    features, masks = batch()
    features = {m: x.masked_fill(~masks[m][..., None], float("nan")) for m, x in features.items()}
    logits = model(features, masks)
    logits.square().sum().backward()
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize(
    "option",
    [
        "graph_use_dynamic_graph",
        "graph_use_view_memory",
        "graph_use_fusion_memory",
        "graph_use_temporal_attention",
    ],
)
def test_graph_component_ablations_preserve_interface(option):
    model = build_model(config("graph_mfn", **{option: False}), DIMS)
    logits = model(*batch())
    logits.square().sum().backward()
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("name", NEW_MODELS)
def test_disabled_modality_cannot_affect_logits(name):
    model = build_model(config(name, modalities=["text", "vision"]), DIMS).eval()
    features, masks = batch()
    expected = model(features, masks)
    features["audio"] = torch.full_like(features["audio"], float("nan"))
    masks["audio"][:] = True
    torch.testing.assert_close(model(features, masks), expected)


@pytest.mark.parametrize("name", ["text", "concat", "transformer", *NEW_MODELS])
def test_config_and_checkpoint_roundtrip(name, tmp_path):
    cfg = config(name)
    model = build_model(cfg, DIMS).eval()
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_config": asdict(cfg), "model_state": model.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = build_model(ModelConfig(**saved["model_config"]), DIMS).eval()
    restored.load_state_dict(saved["model_state"], strict=True)
    torch.testing.assert_close(restored(*batch()), model(*batch()))


@pytest.mark.parametrize("name", ["text", "concat", "transformer"])
def test_legacy_config_without_new_fields_remains_loadable(name):
    legacy = {
        "name": name,
        "d_model": 8,
        "hidden_dim": 7,
        "n_layers": 1,
        "n_heads": 2,
        "ff_dim": 16,
        "dropout": 0,
        "modality_embeddings": True,
        "positions": True,
        "pooling": "cls",
        "modalities": ["text", "audio", "vision"],
    }
    model = build_model(ModelConfig(**legacy), DIMS)
    assert model(*batch()).shape == (3, 2)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"almt_token_count": 0}, "almt_token_count"),
        ({"almt_max_input_length": 0}, "almt_max_input_length"),
        ({"almt_fusion_depth": 0}, "almt_fusion_depth"),
        ({"graph_mfn_heads": 0}, "graph_mfn_heads"),
        ({"name": "graph_mfn", "hidden_dim": 7}, "hidden_dim"),
    ],
)
def test_invalid_new_model_configuration_fails_clearly(options, match):
    with pytest.raises(ValueError, match=match):
        config(**({"name": "almt"} | options))


def test_almt_overlong_sequence_fails_with_actionable_limit():
    model = build_model(config("almt", almt_max_input_length=2), DIMS)
    with pytest.raises(ValueError, match="maximum 2"):
        model(*batch())


@pytest.mark.parametrize("name", NEW_MODELS)
def test_unaligned_sequences_fail_clearly(name):
    model = build_model(config(name), DIMS)
    features, masks = batch()
    features["audio"] = features["audio"][:, :2]
    masks["audio"] = masks["audio"][:, :2]
    with pytest.raises(ValueError, match="aligned"):
        model(features, masks)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_graph_masked_softmax_is_defined_for_all_missing_rows_in_each_precision(dtype):
    from mosei.graph_mfn import _masked_softmax

    logits = torch.tensor([[0, 2, 3], [2, 0, -1]], dtype=dtype, requires_grad=True)
    valid = torch.tensor([[False, False, False], [True, False, True]])
    weights = _masked_softmax(logits, valid)
    assert torch.isfinite(weights).all()
    assert weights[~valid].count_nonzero() == 0
    torch.testing.assert_close(weights[1].sum(), torch.ones((), dtype=dtype))
    weights.square().sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "name,option,component",
    [
        ("almt", "modality_embeddings", "modality_embedding"),
        ("almt", "almt_use_ahl", "ahl_layers"),
        ("almt", "almt_use_cross_modal_fusion", "fusion_layers"),
        ("almt", "almt_use_graph_fusion", "regressor"),
        ("graph_mfn", "graph_use_dynamic_graph", "dynamic_graph"),
        ("graph_mfn", "graph_use_view_memory", "view_memory_cells"),
        ("graph_mfn", "graph_use_fusion_memory", "fusion_memory"),
        ("graph_mfn", "graph_use_temporal_attention", "temporal_attention"),
    ],
)
def test_each_ablation_switch_actually_bypasses_its_component(name, option, component):
    model = build_model(config(name, **{option: False}), DIMS)
    selected = getattr(model, component)
    modules = (
        list(selected.values())
        if isinstance(selected, torch.nn.ModuleDict)
        else (list(selected) if isinstance(selected, torch.nn.ModuleList) else [selected])
    )
    calls = []
    handles = [module.register_forward_hook(lambda *args: calls.append(True)) for module in modules]
    model(*batch())
    for handle in handles:
        handle.remove()
    assert not calls


@pytest.mark.parametrize("name", NEW_MODELS)
def test_every_present_input_stream_receives_nonzero_gradient(name):
    torch.manual_seed(27)
    model = build_model(config(name), DIMS)
    features, masks = batch()
    for tensor in features.values():
        tensor.requires_grad_()
    model(features, masks).square().sum().backward()
    for modality, tensor in features.items():
        assert tensor.grad[masks[modality]].abs().sum() > 0


def test_graph_node_and_memory_order_stays_text_audio_vision():
    model = build_model(config("graph_mfn"), DIMS)
    assert list(model.input_projection) == ["text", "audio", "vision"]
    assert list(model.unimodal_cells) == ["text", "audio", "vision"]
