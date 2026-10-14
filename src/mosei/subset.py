"""Video-level training subsets that leave validation and test untouched."""

import numpy as np
from torch.utils.data import Subset


def training_subset(dataset, fraction: float, seed: int):
    """Keep whole training videos until the requested clip fraction is reached.

    Videos are shuffled separately inside each majority class, so a small sample
    still contains both sentiments. The prepared features stay normalized by the
    full training split.
    """
    fraction = float(fraction)
    if fraction >= 1:
        return dataset, None
    labels = np.asarray(dataset.labels)
    video_ids = [str(video_id) for video_id in dataset.video_ids]
    if len(video_ids) != len(labels):
        raise ValueError("Video IDs do not match training labels")
    clips, order = {}, []
    for index, video_id in enumerate(video_ids):
        if video_id not in clips:
            order.append(video_id)
            clips[video_id] = []
        clips[video_id].append(index)
    negative_videos, positive_videos = [], []
    for video_id in order:
        indices = clips[video_id]
        n_positive = int(np.sum(labels[indices] == 1))
        if n_positive * 2 >= len(indices):
            positive_videos.append(video_id)
        else:
            negative_videos.append(video_id)
    target = max(1, int(round(len(labels) * fraction)))
    n_positive = int(np.sum(labels == 1))
    n_negative = len(labels) - n_positive
    target_positive = min(n_positive, max(0, int(round(target * n_positive / len(labels)))))
    target_negative = min(n_negative, max(0, target - target_positive))
    rng = np.random.RandomState(seed)

    def pick(videos, target_clips):
        if target_clips <= 0:
            return []
        shuffled = list(videos)
        rng.shuffle(shuffled)
        chosen, total = [], 0
        for video_id in shuffled:
            if chosen and total >= target_clips:
                break
            chosen.append(video_id)
            total += len(clips[video_id])
        return chosen

    selected = set(pick(negative_videos, target_negative) + pick(positive_videos, target_positive))
    indices = [index for index, video_id in enumerate(video_ids) if video_id in selected]
    chosen_labels = labels[indices]
    counts = {
        "negative": int(np.sum(chosen_labels == 0)),
        "positive": int(np.sum(chosen_labels == 1)),
    }
    if counts["negative"] == 0 or counts["positive"] == 0:
        raise ValueError("Training subset must contain both sentiment classes")
    record = {
        "fraction": fraction,
        "seed": int(seed),
        "rule": (
            "Whole training videos, stratified by each video's majority label. "
            "Validation and test stay complete. Features keep the full-training normalization."
        ),
        "target_clips": target,
        "clip_count": len(indices),
        "video_count": len(selected),
        "full_clip_count": int(len(labels)),
        "full_video_count": len(order),
        "class_counts": counts,
        "video_ids": sorted(selected),
    }
    return Subset(dataset, indices), record
