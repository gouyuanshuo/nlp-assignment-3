import pytest
import torch

from mosei.evaluate import Condition, corrupt_batch
from mosei.metrics import classification_metrics
from mosei.report import error_examples, paired_bootstrap


def test_noise_is_per_clip_batch_invariant_and_padding_unchanged():
    features = {m: torch.zeros(2, 4, 3) for m in ("text", "audio", "vision")}
    masks = {m: torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool) for m in features}
    condition = Condition("noise", "audio", 0.3, 2026)
    corrupted, out_masks = corrupt_batch(features, masks, ["a", "b"], condition)
    single, _ = corrupt_batch(
        {m: x[1:] for m, x in features.items()},
        {m: x[1:] for m, x in masks.items()},
        ["b"],
        condition,
    )
    torch.testing.assert_close(corrupted["audio"][1:], single["audio"])
    assert not corrupted["audio"][~masks["audio"]].any()
    assert corrupted["audio"][masks["audio"]].abs().sum() > 0
    assert not corrupted["text"].any()
    assert not features["audio"].any()
    assert torch.equal(out_masks["audio"], masks["audio"])


def test_missing_modality_masks_all_tokens_without_mutating_original():
    features = {m: torch.ones(1, 3, 2) for m in ("text", "audio", "vision")}
    masks = {m: torch.ones(1, 3, dtype=torch.bool) for m in features}
    out, valid = corrupt_batch(features, masks, ["a"], Condition("missing", "text"))
    assert not out["text"].any() and not valid["text"].any()
    assert out["audio"].all() and masks["text"].all()


def records(predictions):
    return [
        {"id": str(i), "video_id": "v" + str(i // 2), "label": label, "prediction": pred}
        for i, (label, pred) in enumerate(zip([0, 0, 1, 1], predictions, strict=True))
    ]


def test_metrics_fixed_class_order_and_hand_calculated_f1():
    metrics = classification_metrics(records([0, 1, 1, 1]))
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics["per_class"]["positive"]["recall"] == 1


def test_video_cluster_bootstrap_is_paired_and_reproducible():
    first, second = records([0, 0, 1, 1]), records([1, 1, 0, 0])
    result = paired_bootstrap([first], [second], n_resamples=100, seed=8)
    assert result["mean_difference"] == 1
    assert result["cluster_count"] == 2
    assert result["ci_low"] >= 0.5  # All-correct vs all-wrong even if one class sampled.
    assert result == paired_bootstrap([first], [second], n_resamples=100, seed=8)
    zero = paired_bootstrap([first], [first], n_resamples=30, seed=0)
    assert zero["ci_low"] == zero["ci_high"] == 0


def test_paired_report_refuses_misaligned_or_relabelled_examples():
    left, right = records([0, 0, 1, 1]), records([0, 0, 1, 1])
    right[0]["label"] = 1
    with pytest.raises(ValueError, match="pair|label"):
        paired_bootstrap([left], [right], n_resamples=5)
    with pytest.raises(ValueError, match="pair|ID|id"):
        paired_bootstrap([left], [right[1:]], n_resamples=5)


def test_invalid_corruption_configuration_rejected():
    with pytest.raises(ValueError):
        Condition("noise", "audio", -1)


def test_error_examples_identify_corrections_and_regressions():
    examples = error_examples(records([0, 0, 1, 1]), records([1, 1, 0, 0]))
    assert examples["left_correct_right_wrong_count"] == 4
    assert examples["right_correct_left_wrong_count"] == 0
    assert examples["left_correct_right_wrong"][0]["id"] == "0"
