import pytest
import torch

from mosei.models import ModelConfig, build_model

DIMS = {"text": 4, "audio": 2, "vision": 1}


def batch():
    generator = torch.Generator().manual_seed(44)
    features = {m: torch.randn(2, 4, d, generator=generator) for m, d in DIMS.items()}
    masks = {m: torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool) for m in DIMS}
    return features, masks


@pytest.mark.parametrize("name", ["text", "concat", "transformer"])
def test_padding_values_and_extra_padding_cannot_change_logits(name):
    torch.manual_seed(0)
    model = build_model(
        ModelConfig(name=name, d_model=8, n_heads=2, ff_dim=16, dropout=0), DIMS
    ).eval()
    features, masks = batch()
    expected = model(features, masks)
    altered = {m: x.masked_fill(~masks[m][..., None], 10000) for m, x in features.items()}
    actual = model(altered, masks)
    torch.testing.assert_close(actual, expected)
    extended = {
        m: torch.cat([x, torch.full_like(x[:, :1], -500)], dim=1) for m, x in features.items()
    }
    extended_masks = {
        m: torch.cat([mask, torch.zeros(2, 1, dtype=torch.bool)], dim=1)
        for m, mask in masks.items()
    }
    torch.testing.assert_close(model(extended, extended_masks), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("name", ["text", "concat", "transformer"])
@pytest.mark.parametrize("missing", ["text", "audio", "vision", "all"])
def test_missing_streams_have_finite_predictions_and_gradients(name, missing):
    model = build_model(ModelConfig(name=name, d_model=8, n_heads=2, ff_dim=16, dropout=0), DIMS)
    features, masks = batch()
    for m in DIMS:
        if m == missing or missing == "all":
            masks[m][:] = False
    logits = model(features, masks)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    altered = {m: x + 100 * (~masks[m])[..., None] for m, x in features.items()}
    torch.testing.assert_close(model(altered, masks), logits)


def test_transformer_gradient_reaches_each_modality_and_attention():
    torch.manual_seed(4)
    model = build_model(ModelConfig(d_model=8, n_heads=2, ff_dim=16, dropout=0), DIMS)
    features, masks = batch()
    for x in features.values():
        x.requires_grad_()
    loss = model(features, masks).square().sum()
    loss.backward()
    for m, x in features.items():
        assert x.grad[masks[m]].abs().sum() > 0
        assert x.grad[~masks[m]].abs().sum() == 0
    assert model.encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0
    assert not torch.equal(
        model.encoder.layers[0].linear1.weight, model.encoder.layers[1].linear1.weight
    )


@pytest.mark.parametrize(
    "option", [{"modality_embeddings": False}, {"positions": False}, {"pooling": "mean"}]
)
def test_ablation_configs_execute(option):
    model = build_model(ModelConfig(d_model=8, n_heads=2, ff_dim=16, **option), DIMS)
    assert torch.isfinite(model(*batch())).all()


def test_dropped_modality_has_no_parameters_and_cannot_change_logits():
    torch.manual_seed(0)
    model = build_model(
        ModelConfig(
            d_model=8, n_heads=2, ff_dim=16, dropout=0, modalities=["text", "vision"]
        ),
        DIMS,
    ).eval()
    features, masks = batch()
    expected = model(features, masks)
    changed = dict(features)
    changed["audio"] = features["audio"] + 50
    torch.testing.assert_close(model(changed, masks), expected)
    assert set(model.projections) == {"text", "vision"}
    with pytest.raises(ValueError):
        ModelConfig(modalities=["text", "missing"])
