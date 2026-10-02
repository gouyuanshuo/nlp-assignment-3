"""Hand-calculated timestamp/window examples for training and video inference."""

import numpy as np
import pytest


class WordpieceCounter:
    model_max_length = 512

    def __call__(self, words, **kwargs):
        return {"input_ids": [101] + [1 for w in words for _ in w.split("-")] + [102]}


def test_alignment_weights_overlap_and_excludes_next_boundary():
    from mosei.video_features import TimedFeatures, Word, align_words

    stream = TimedFeatures(
        np.array([[0, 0.5], [0.5, 1.5], [2, 2]]), np.array([[0], [10], [20]]), ["F0"]
    )
    words = [Word("a", 0.25, 1), Word("b", 1, 2), Word("c", 2, 2.5), Word("d", 3, 4)]
    values, valid = align_words(words, stream)
    np.testing.assert_allclose(values[:3, 0], [20 / 3, 10, 20])
    assert valid.tolist() == [True, True, True, False]
    assert np.isnan(values[3, 0])


def test_nonfinite_measurements_do_not_contaminate_available_channels():
    from mosei.video_features import TimedFeatures, Word, align_words

    stream = TimedFeatures(
        np.array([[0, 1], [0, 1]]), np.array([[np.nan, 4], [6, np.inf]]), ["x", "y"]
    )
    values, mask = align_words([Word("a", 0, 1)], stream)
    np.testing.assert_array_equal(values, [[6, 4]])
    assert mask.tolist() == [True]


def test_51_words_keep_late_content_and_positive_durations():
    from mosei.video_features import Word, make_windows

    words = [Word(str(i), i, i + 1) for i in range(51)]
    windows = make_windows(words, WordpieceCounter())
    assert [len(w) for w in windows] == [50, 1]
    assert windows[1][0].text == "50"


def test_token_limit_splits_before_word_limit():
    from mosei.video_features import Word, make_windows

    tokenizer = WordpieceCounter()
    tokenizer.model_max_length = 7
    words = [Word("three-word-pieces", i, i + 1) for i in range(2)]
    assert [len(w) for w in make_windows(words, tokenizer)] == [1, 1]
    tokenizer.model_max_length = 4
    with pytest.raises(ValueError, match="token"):
        make_windows(words, tokenizer)


@pytest.mark.parametrize("start,end", [(0, 0), (2, 1), (float("nan"), 1), (-1, 1)])
def test_invalid_word_intervals_fail(start, end):
    from mosei.video_features import Word

    with pytest.raises(ValueError, match="word|interval"):
        Word("hello", start, end)


def test_manifest_mismatch_fails_before_encoder_runs():
    from mosei.video_features import TimedFeatures, Word, prepare_windows

    stream = TimedFeatures(np.array([[0, 1]]), np.array([[1]]), ["wrong"])
    manifest = {"audio": {"columns": ["F0"]}, "vision": {"columns": ["AU01_r"]}, "max_words": 50}
    with pytest.raises(ValueError, match="column"):
        prepare_windows([Word("hello", 0, 1)], stream, stream, None, manifest)


def test_shared_preparation_masks_padding_and_uses_saved_normalization():
    from mosei.video_features import TimedFeatures, Word, prepare_windows

    class Encoder:
        tokenizer = WordpieceCounter()

        def encode(self, windows):
            return [np.array([[2, 4]] * len(words), dtype=np.float32) for words in windows]

    audio = TimedFeatures(np.array([[0, 1]]), np.array([[10]]), ["F0"])
    vision = TimedFeatures(np.array([[1, 2]]), np.array([[4]]), ["AU01_r"])
    manifest = {"audio": {"columns": ["F0"]}, "vision": {"columns": ["AU01_r"]}, "max_words": 50}
    stats = {
        "text": {"mean": [1, 2], "scale": [1, 2]},
        "audio": {"mean": [8], "scale": [2]},
        "vision": {"mean": [2], "scale": [2]},
    }
    result = prepare_windows(
        [Word("hello", 0, 1), Word("there", 1, 2)], audio, vision, Encoder(), manifest, stats
    )
    assert result["features"]["text"].shape == (1, 50, 2)
    np.testing.assert_array_equal(result["features"]["text"][0, :2], [[1, 1], [1, 1]])
    np.testing.assert_array_equal(result["features"]["audio"][0, :2, 0], [1, 0])
    np.testing.assert_array_equal(result["features"]["vision"][0, :2, 0], [0, 1])
    assert result["masks"]["audio"][0, :3].tolist() == [True, False, False]
    assert result["durations"].tolist() == [2]
    assert result["intervals"].tolist() == [[0, 2]]
    assert all(np.all(a[:, 2:] == 0) for a in result["features"].values())


def test_bert_word_encoder_is_invariant_to_other_batch_padding(tmp_path):
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast

    from mosei.text_features import BertWordEncoder
    from mosei.video_features import Word

    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nthere\nplay\n##ing\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), model_max_length=32)
    tokenizer.save_pretrained(tmp_path)
    torch.manual_seed(1)
    model = BertModel(
        BertConfig(
            vocab_size=9,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
        )
    )
    model.save_pretrained(tmp_path)
    encoder = BertWordEncoder(tmp_path, device="cpu")
    short = [Word("playing", 0, 1)]
    long = [Word("hello", 0, 1), Word("there", 1, 2), Word("playing", 2, 3)]
    solo = encoder.encode([short])[0]
    paired = encoder.encode([short, long])
    assert paired[0].shape == (1, 8)
    assert paired[1].shape == (3, 8)
    np.testing.assert_allclose(solo, paired[0], atol=1e-6)
    assert np.isfinite(paired[1]).all()
