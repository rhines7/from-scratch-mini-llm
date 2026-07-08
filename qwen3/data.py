"""
TinyStories data preparation.

Loads the TinyStories corpus from Hugging Face, splits it with a fixed seed,
trains the tokenizer on the training split only (so validation and test stay
out of the vocabulary), analyzes token-length distribution to pick a sequence
length, and builds causal-LM DataLoaders.

The tokenizer is fit on training text alone to avoid leaking held-out data into
the vocabulary. Sequence-length plotting lives in viz.py; this module only
computes the statistics.
"""

import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import Qwen3SmallTokenizer
from .config import (
    ARTIFACTS_DIR,
    DATASET_STATS_PATH,
    SEQ_LENGTH_FIGURE_PATH,
    SEQUENCE_LENGTHS_ARTIFACT,
    TOKENIZER_VOCAB_PATH,
    DEFAULT_SEED,
)


@dataclass
class DataConfig:
    """Configuration for dataset preparation.

    Defaults reproduce the committed run: 10,000 stories, an 80/10/10 split, and
    a 256-token cap chosen from the length analysis.
    """
    # Dataset size
    num_stories: int = 10000

    # Train/val/test split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Tokenizer settings (vocab_size is a target cap; realized size is smaller)
    vocab_size: int = 20000
    min_frequency: int = 2

    # Sequence settings
    max_seq_len: int = 256

    # DataLoader settings
    batch_size: int = 8
    num_workers: int = 0  # CPU training: avoid multiprocessing overhead

    random_seed: int = DEFAULT_SEED

    # Output paths (defaults resolve to the repo's canonical locations)
    tokenizer_path: str = field(default_factory=lambda: str(TOKENIZER_VOCAB_PATH))
    data_stats_path: str = field(default_factory=lambda: str(DATASET_STATS_PATH))
    figure_path: str = field(default_factory=lambda: str(SEQ_LENGTH_FIGURE_PATH))

    def __post_init__(self):
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        assert self.num_stories > 0, "num_stories must be positive"
        if self.num_stories < 1000:
            print(f"Warning: num_stories ({self.num_stories}) is very small; "
                  f"10k or more is recommended for a real run.")


class TinyStoriesDataset(Dataset):
    """Causal-LM dataset: tokenizes stories and shifts padding to -100 in labels."""

    def __init__(
        self,
        texts: List[str],
        tokenizer: Qwen3SmallTokenizer,
        max_seq_len: int = 256,
        add_special_tokens: bool = True
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.add_special_tokens = add_special_tokens

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        token_ids, attention_mask = self.tokenizer.encode(
            text,
            add_special_tokens=self.add_special_tokens,
            max_length=self.max_seq_len,
            padding=True,
            return_attention_mask=True
        )

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # Labels mirror inputs; the training loop shifts them by one position
        labels = input_ids.clone()

        # Ignore padding positions in the loss
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def load_tiny_stories(
    num_stories: int = 10000,
    split: str = 'train',
    verbose: bool = True
) -> List[str]:
    """Load and subsample TinyStories text from Hugging Face.

    Selection uses the first num_stories indices so the sample is reproducible.
    """
    from datasets import load_dataset

    if verbose:
        print(f"Loading TinyStories dataset (split='{split}')...")
        print("(First download is roughly 1-2 GB and may take a few minutes.)")

    try:
        dataset = load_dataset('roneneldan/TinyStories', split=split, streaming=False)
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("Check the network connection and that 'datasets' is installed.")
        raise

    if verbose:
        print(f"Total stories available: {len(dataset):,}")

    if num_stories < len(dataset):
        indices = list(range(num_stories))
        dataset = dataset.select(indices)

    texts = [example['text'] for example in dataset]

    if verbose:
        print(f"Loaded {len(texts):,} stories")

    return texts


def split_dataset(
    texts: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = DEFAULT_SEED,
    verbose: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    """Shuffle with a fixed seed and split into train/val/test."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    np.random.seed(random_seed)

    indices = np.random.permutation(len(texts))

    train_end = int(len(texts) * train_ratio)
    val_end = train_end + int(len(texts) * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_texts = [texts[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    test_texts = [texts[i] for i in test_indices]

    if verbose:
        print("\nDataset split:")
        print(f"  Training: {len(train_texts):,} stories ({train_ratio*100:.1f}%)")
        print(f"  Validation: {len(val_texts):,} stories ({val_ratio*100:.1f}%)")
        print(f"  Test: {len(test_texts):,} stories ({test_ratio*100:.1f}%)")

    return train_texts, val_texts, test_texts


def analyze_sequence_lengths(
    texts: List[str],
    tokenizer: Qwen3SmallTokenizer,
    verbose: bool = True
) -> Tuple[Dict, np.ndarray]:
    """Compute token-length statistics and return them with the raw length array.

    The 95th percentile is reported as a max_seq_len recommendation. The raw
    array is returned so viz.py can render the distribution without recomputing.
    """
    if verbose:
        print("\nAnalyzing sequence lengths...")

    lengths = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        lengths.append(len(token_ids))

    lengths_array = np.array(lengths)
    stats = {
        'min': int(np.min(lengths_array)),
        'max': int(np.max(lengths_array)),
        'mean': float(np.mean(lengths_array)),
        'median': float(np.median(lengths_array)),
        'std': float(np.std(lengths_array)),
        'percentile_50': float(np.percentile(lengths_array, 50)),
        'percentile_90': float(np.percentile(lengths_array, 90)),
        'percentile_95': float(np.percentile(lengths_array, 95)),
        'percentile_99': float(np.percentile(lengths_array, 99)),
    }

    recommended = int(stats['percentile_95'])
    coverage_95 = (lengths_array <= recommended).sum() / len(lengths_array) * 100
    stats['recommended_max_seq_len'] = recommended
    stats['coverage_95'] = coverage_95

    if verbose:
        print(f"  Min {stats['min']}, max {stats['max']}, "
              f"mean {stats['mean']:.1f}, median {stats['median']:.1f}")
        print(f"  95th percentile {recommended} covers {coverage_95:.1f}% of stories")

    return stats, lengths_array


def prepare_tiny_stories_dataset(
    config: Optional[DataConfig] = None,
    verbose: bool = True,
    make_figure: bool = True,
    write_artifacts: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Qwen3SmallTokenizer, Dict]:
    """Run the full data-prep stage and return loaders, tokenizer, and statistics.

    The tokenizer is fit on the training split only. Statistics are written to
    the artifacts directory; the length-distribution figure is saved via viz.py.

    write_artifacts is disabled for --quick smoke runs so the tiny-model
    statistics never overwrite the committed dataset summaries.
    """
    if config is None:
        config = DataConfig()

    print("=" * 70)
    print("Dataset preparation")
    print("=" * 70)

    all_texts = load_tiny_stories(
        num_stories=config.num_stories,
        split='train',
        verbose=verbose
    )

    train_texts, val_texts, test_texts = split_dataset(
        texts=all_texts,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed,
        verbose=verbose
    )

    # Fit tokenizer on training text only to avoid leakage
    tokenizer = Qwen3SmallTokenizer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        lowercase=False
    )

    vocab_stats = tokenizer.build_vocab(train_texts, verbose=verbose)

    tokenizer.save_vocab(config.tokenizer_path)

    length_stats, lengths_array = analyze_sequence_lengths(
        texts=train_texts,
        tokenizer=tokenizer,
        verbose=verbose
    )

    # Persist raw lengths so --figures-only can rebuild the distribution later
    if write_artifacts:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(SEQUENCE_LENGTHS_ARTIFACT, 'w') as f:
            json.dump(lengths_array.tolist(), f)

    if make_figure:
        # Local import keeps matplotlib out of the import path for training-only runs
        from .viz import plot_sequence_length_distribution
        plot_sequence_length_distribution(lengths_array, length_stats, config.figure_path)

    train_dataset = TinyStoriesDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        add_special_tokens=True
    )
    val_dataset = TinyStoriesDataset(
        texts=val_texts,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        add_special_tokens=True
    )
    test_dataset = TinyStoriesDataset(
        texts=test_texts,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        add_special_tokens=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False
    )

    print(f"  Train/val/test examples: "
          f"{len(train_dataset):,}/{len(val_dataset):,}/{len(test_dataset):,}")
    print(f"  Batch size {config.batch_size}; "
          f"{len(train_loader):,} training batches")

    stats = {
        'num_stories': config.num_stories,
        'train_size': len(train_texts),
        'val_size': len(val_texts),
        'test_size': len(test_texts),
        'vocab_size': len(tokenizer),
        'vocab_coverage': tokenizer.vocab_coverage,
        'max_seq_len': config.max_seq_len,
        'batch_size': config.batch_size,
        'vocab_stats': vocab_stats,
        'length_stats': length_stats,
        'config': {
            'num_stories': config.num_stories,
            'train_ratio': config.train_ratio,
            'val_ratio': config.val_ratio,
            'test_ratio': config.test_ratio,
            'vocab_size': config.vocab_size,
            'min_frequency': config.min_frequency,
            'max_seq_len': config.max_seq_len,
            'batch_size': config.batch_size,
            'random_seed': config.random_seed
        }
    }

    if write_artifacts:
        with open(config.data_stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Statistics saved to {config.data_stats_path}")

    return train_loader, val_loader, test_loader, tokenizer, stats


def test_dataloader(dataloader: DataLoader, tokenizer: Qwen3SmallTokenizer):
    """Fetch one batch and print shapes plus a decoded example for a sanity check."""
    print("\nInspecting one batch...")

    batch = next(iter(dataloader))

    print(f"  input_ids {tuple(batch['input_ids'].shape)}, "
          f"attention_mask {tuple(batch['attention_mask'].shape)}, "
          f"labels {tuple(batch['labels'].shape)}")

    input_ids = batch['input_ids'][0]
    decoded = tokenizer.decode(input_ids.tolist(), skip_special_tokens=True)
    print(f"  Decoded example: {decoded[:200]}")
