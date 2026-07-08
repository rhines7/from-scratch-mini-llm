"""
Downstream sentiment classification on top of the pre-trained model.

A lightweight head is trained over pooled hidden states from the frozen (or
optionally fine-tuned) Qwen3-small base. Freezing the base turns this into a
probe of the learned representations: only the small classifier head is trained,
which is fast and isolates how much sentiment signal the pre-training captured.

Defaults reproduce the committed run: frozen base, mean pooling, a 128-unit
hidden layer, 15 epochs at learning rate 1e-3.
"""

import os
import csv
import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .architecture import Qwen3Model, Qwen3Config
from .tokenizer import Qwen3SmallTokenizer
from . import artifacts
from .config import (
    EMOTIONS_CSV_PATH,
    SENTIMENT_CHECKPOINT_DIR,
    BEST_SENTIMENT_MODEL_PATH,
    BEST_MODEL_PATH,
    TOKENIZER_VOCAB_PATH,
    DEFAULT_SEED,
)

_CLASS_NAMES = {0: "negative", 1: "positive"}


def collect_misclassified(texts, labels, predictions, max_examples: int = 20):
    """Return up to max_examples test cases the classifier got wrong.

    The test loader is unshuffled, so predictions align positionally with the
    text/label lists. These examples anchor the error-analysis discussion in the
    report with concrete cases rather than aggregate numbers alone.
    """
    errors = []
    for text, true_label, pred in zip(texts, labels, predictions):
        if pred != true_label:
            errors.append({
                "text": text,
                "true": _CLASS_NAMES.get(true_label, str(true_label)),
                "predicted": _CLASS_NAMES.get(pred, str(pred)),
            })
            if len(errors) >= max_examples:
                break
    return errors


@dataclass
class SentimentConfig:
    """Configuration for sentiment classifier training."""
    # Data
    data_file: str = field(default_factory=lambda: str(EMOTIONS_CSV_PATH))
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1

    # Model / head
    freeze_base_model: bool = True   # Probe frozen representations by default
    pooling_strategy: str = "mean"   # "last", "mean", or "first"
    hidden_dim: int = 128
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    num_epochs: int = 15
    learning_rate: float = 1e-3      # Higher LR since only the head trains
    weight_decay: float = 0.01

    # Fine-tuning path (used when freeze_base_model is False)
    base_lr: float = 1e-5

    gradient_clip: float = 1.0
    class_weights: Optional[List[float]] = None

    log_every_n_steps: int = 50
    eval_every_n_steps: int = 200

    checkpoint_dir: str = field(default_factory=lambda: str(SENTIMENT_CHECKPOINT_DIR))
    best_model_path: str = field(default_factory=lambda: str(BEST_SENTIMENT_MODEL_PATH))

    device: str = "cpu"

    random_seed: int = DEFAULT_SEED

    def __post_init__(self):
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6
        assert self.pooling_strategy in ["last", "mean", "first"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)


class EmotionsDataset(Dataset):
    """Tokenized text/label pairs for binary sentiment classification."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Qwen3SmallTokenizer,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True
        )

        attention_mask = [1 if token_id != self.tokenizer.pad_id else 0
                          for token_id in encoding]

        return {
            'input_ids': torch.tensor(encoding, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_emotions_data(file_path: str) -> Tuple[List[str], List[str], List[int]]:
    """Read the emotions CSV, mapping sentiment strings to 0 (negative) / 1 (positive)."""
    texts = []
    sentiments = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
            sentiment = row['sentiment']
            sentiments.append(sentiment)
            labels.append(1 if sentiment == 'positive' else 0)

    return texts, sentiments, labels


def split_data(
    texts: List[str],
    labels: List[int],
    train_split: float,
    val_split: float,
    test_split: float,
    random_seed: int = DEFAULT_SEED
) -> Tuple:
    """Shuffle with a fixed seed and split into train/val/test lists."""
    combined = list(zip(texts, labels))
    random.seed(random_seed)
    random.shuffle(combined)
    texts, labels = zip(*combined)
    texts, labels = list(texts), list(labels)

    n = len(texts)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)

    train_texts = texts[:train_end]
    train_labels = labels[:train_end]

    val_texts = texts[train_end:val_end]
    val_labels = labels[train_end:val_end]

    test_texts = texts[val_end:]
    test_labels = labels[val_end:]

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


class SentimentClassifier(nn.Module):
    """Pooling + MLP head over Qwen3-small hidden states.

    The base model can be frozen (feature extraction) or fine-tuned. Pooling
    reduces the sequence of hidden states to one vector before the two-layer head.
    """

    def __init__(
        self,
        base_model: Qwen3Model,
        num_classes: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        pooling_strategy: str = "mean",
        freeze_base: bool = True
    ):
        super().__init__()

        self.base_model = base_model
        self.pooling_strategy = pooling_strategy
        self.freeze_base = freeze_base

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        d_model = base_model.config.d_model

        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def pool_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Reduce (batch, seq_len, d_model) to (batch, d_model) per the pooling strategy."""
        if self.pooling_strategy == "last":
            # Last non-padding position for each sequence
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            pooled = hidden_states[torch.arange(batch_size), seq_lengths]

        elif self.pooling_strategy == "mean":
            # Mean over non-padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask

        elif self.pooling_strategy == "first":
            # BOS token
            pooled = hidden_states[:, 0, :]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Skip base-model gradients entirely when frozen to save memory/compute
        if self.freeze_base:
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
        else:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

        hidden_states = outputs['last_hidden_state']

        pooled = self.pool_embeddings(hidden_states, attention_mask)

        logits = self.classifier(pooled)

        return logits


def compute_metrics(predictions: List[int], labels: List[int]) -> Dict:
    """Accuracy plus per-class precision/recall/F1 and macro-F1."""
    n = len(labels)
    assert len(predictions) == n

    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = correct / n

    metrics = {'accuracy': accuracy}

    for class_id, class_name in [(0, 'negative'), (1, 'positive')]:
        tp = sum((p == class_id and l == class_id) for p, l in zip(predictions, labels))
        fp = sum((p == class_id and l != class_id) for p, l in zip(predictions, labels))
        fn = sum((p != class_id and l == class_id) for p, l in zip(predictions, labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f'{class_name}_precision'] = precision
        metrics[f'{class_name}_recall'] = recall
        metrics[f'{class_name}_f1'] = f1

    metrics['macro_f1'] = (metrics['negative_f1'] + metrics['positive_f1']) / 2

    return metrics


def confusion_counts(predictions: List[int], labels: List[int], num_classes: int = 2) -> np.ndarray:
    """Return the confusion matrix as counts, indexed cm[true][pred]."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(labels, predictions):
        cm[true_label][pred_label] += 1
    return cm


# Metrics carried through the bootstrap; per-class precision/recall are included
# so the per-class figure can show real error bars, not just F1.
_CI_METRIC_KEYS = (
    "accuracy",
    "macro_f1",
    "negative_precision", "negative_recall", "negative_f1",
    "positive_precision", "positive_recall", "positive_f1",
)


def bootstrap_metric_cis(
    predictions: List[int],
    labels: List[int],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEED
) -> Dict:
    """Percentile bootstrap confidence intervals for the test metrics.

    Resampling the fixed test set quantifies sampling uncertainty on a single
    trained model. This is deliberately not a seed/training-variance estimate:
    it answers "how tight is this number given the test set?", not "how much
    would it move if I retrained?". Retraining variance would need multiple runs.
    """
    preds = np.asarray(predictions)
    labs = np.asarray(labels)
    n = len(labs)
    rng = np.random.default_rng(seed)

    samples = {key: [] for key in _CI_METRIC_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = compute_metrics(preds[idx].tolist(), labs[idx].tolist())
        for key in samples:
            samples[key].append(m[key])

    point = compute_metrics(predictions, labels)

    def _ci(key):
        values = samples[key]
        return {
            "point": point[key],
            "lower": float(np.percentile(values, 100 * alpha / 2)),
            "upper": float(np.percentile(values, 100 * (1 - alpha / 2))),
        }

    cis = {
        "method": "percentile_bootstrap",
        "n_boot": n_boot,
        "confidence": 1 - alpha,
    }
    cis.update({key: _ci(key) for key in _CI_METRIC_KEYS})
    return cis


# OOV-rate buckets for the error analysis; each entry is (label, predicate)
_OOV_BUCKETS = (
    ("0%", lambda r: r == 0.0),
    ("(0,10%]", lambda r: 0.0 < r <= 10.0),
    ("(10,25%]", lambda r: 10.0 < r <= 25.0),
    (">25%", lambda r: r > 25.0),
)


def oov_error_analysis(
    texts: List[str],
    labels: List[int],
    predictions: List[int],
    tokenizer: Qwen3SmallTokenizer
) -> Dict:
    """Relate per-sentence OOV rate to classification correctness.

    If the tokenizer domain gap is the ceiling on the probe, misclassified
    sentences should carry systematically higher OOV rates and accuracy should
    fall as OOV rises. This turns that hypothesis into direct evidence rather
    than an inference from the aggregate OOV number.
    """
    oov_rates = np.array([tokenizer.coverage([text])["oov_rate"] for text in texts])
    correct = np.array([int(p == l) for p, l in zip(predictions, labels)])

    buckets = []
    for label, predicate in _OOV_BUCKETS:
        mask = np.array([predicate(r) for r in oov_rates], dtype=bool)
        count = int(mask.sum())
        buckets.append({
            "range": label,
            "count": count,
            "accuracy": float(correct[mask].mean()) if count else None,
            "mean_oov": float(oov_rates[mask].mean()) if count else None,
        })

    incorrect = 1 - correct
    # Point-biserial correlation reduces to Pearson between OOV rate and the
    # binary error indicator; guard the degenerate zero-variance cases.
    if oov_rates.std() > 0 and incorrect.std() > 0:
        correlation = float(np.corrcoef(oov_rates, incorrect)[0, 1])
    else:
        correlation = None

    return {
        "n": len(texts),
        "correlation_oov_vs_error": correlation,
        "mean_oov_correct": float(oov_rates[correct == 1].mean()) if correct.any() else None,
        "mean_oov_incorrect": float(oov_rates[incorrect == 1].mean()) if incorrect.any() else None,
        "buckets": buckets,
    }


@torch.no_grad()
def evaluate(
    model: SentimentClassifier,
    dataloader: DataLoader,
    device: str = "cpu"
) -> Dict:
    """Run the classifier over a loader and return metrics plus raw predictions/labels."""
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_predictions.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_predictions, all_labels)
    metrics['loss'] = avg_loss

    # Retained for confusion-matrix rendering; trimmed before committing artifacts
    metrics['predictions'] = all_predictions
    metrics['labels'] = all_labels

    return metrics


def _load_base_model(base_model_path: str) -> Qwen3Model:
    """Rebuild the pre-trained base model from its checkpoint config."""
    checkpoint = torch.load(base_model_path, map_location='cpu', weights_only=False)
    model_config = Qwen3Config(**checkpoint['model_config'])
    base_model = Qwen3Model(model_config)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    return base_model


def train_sentiment_classifier(
    config: Optional[SentimentConfig] = None,
    base_model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    write_artifacts: bool = False,
    variant: str = "frozen"
) -> SentimentClassifier:
    """Train the sentiment head, select the best checkpoint by val macro-F1, and
    evaluate on the held-out test set.

    Only the classifier parameters are optimized when the base is frozen;
    otherwise the base is fine-tuned at a lower learning rate. Final test metrics
    and training history are written under the sentiment checkpoint directory.

    When write_artifacts is set, the committed summary artifact is (re)written:
    the "frozen" variant writes the headline sentiment metrics plus a sample of
    misclassified examples; the "finetuned" variant writes the comparison file.
    Callers gate this on full (non-quick) runs so smoke tests never overwrite the
    committed summaries.
    """
    if config is None:
        config = SentimentConfig()
    base_model_path = str(base_model_path or BEST_MODEL_PATH)
    tokenizer_path = str(tokenizer_path or TOKENIZER_VOCAB_PATH)

    print("=" * 70)
    print("Sentiment classification")
    print("=" * 70)

    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)

    print("\nLoading pre-trained base model and tokenizer...")
    base_model = _load_base_model(base_model_path)
    print(f"Base model: {base_model.count_parameters()['actual_millions']:.2f}M parameters")

    tokenizer = Qwen3SmallTokenizer(vocab_size=20000)
    tokenizer.load_vocab(tokenizer_path)

    texts, sentiments, labels = load_emotions_data(config.data_file)
    neg_count = labels.count(0)
    pos_count = labels.count(1)
    print(f"Loaded {len(texts)} samples "
          f"({neg_count} negative, {pos_count} positive)")

    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        split_data(texts, labels, config.train_split, config.val_split,
                   config.test_split, config.random_seed)

    train_dataset = EmotionsDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_dataset = EmotionsDataset(val_texts, val_labels, tokenizer, max_length=128)
    test_dataset = EmotionsDataset(test_texts, test_labels, tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = SentimentClassifier(
        base_model=base_model,
        num_classes=2,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        pooling_strategy=config.pooling_strategy,
        freeze_base=config.freeze_base_model
    )
    model = model.to(config.device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params/1e6:.3f}M "
          f"(base frozen: {config.freeze_base_model}, pooling: {config.pooling_strategy})")

    if config.freeze_base_model:
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW([
            {'params': model.base_model.parameters(), 'lr': config.base_lr},
            {'params': model.classifier.parameters(), 'lr': config.learning_rate}
        ], weight_decay=config.weight_decay)

    if config.class_weights:
        class_weights = torch.tensor(config.class_weights, dtype=torch.float32, device=config.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    best_val_f1 = 0.0
    global_step = 0

    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'steps': []
    }

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels_batch = batch['labels'].to(config.device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            epoch_correct += (preds == labels_batch).sum().item()
            epoch_total += labels_batch.size(0)

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{epoch_correct/epoch_total:.4f}"
            })

            global_step += 1

            if global_step % config.eval_every_n_steps == 0:
                val_metrics = evaluate(model, val_loader, config.device)

                print(f"\nStep {global_step} | val loss {val_metrics['loss']:.4f} | "
                      f"val acc {val_metrics['accuracy']:.4f} | "
                      f"val macro-F1 {val_metrics['macro_f1']:.4f}")

                # Keep the checkpoint with the best validation macro-F1
                if val_metrics['macro_f1'] > best_val_f1:
                    best_val_f1 = val_metrics['macro_f1']
                    print(f"New best (macro-F1 {best_val_f1:.4f})")

                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                        'global_step': global_step,
                        'epoch': epoch,
                        'metrics': {k: v for k, v in val_metrics.items()
                                    if k not in ('predictions', 'labels')}
                    }, config.best_model_path)

                training_history['val_loss'].append(val_metrics['loss'])
                training_history['val_acc'].append(val_metrics['accuracy'])
                training_history['val_f1'].append(val_metrics['macro_f1'])
                training_history['steps'].append(global_step)

                model.train()

        avg_loss = epoch_loss / len(train_loader)
        avg_acc = epoch_correct / epoch_total

        print(f"Epoch {epoch + 1} summary: loss {avg_loss:.4f}, acc {avg_acc:.4f}")

        training_history['train_loss'].append(avg_loss)
        training_history['train_acc'].append(avg_acc)

    print("\n" + "=" * 70)
    print("Test evaluation")
    print("=" * 70)

    best_checkpoint = torch.load(config.best_model_path, map_location=config.device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, config.device)

    print(f"Accuracy {test_metrics['accuracy']:.4f}, "
          f"macro-F1 {test_metrics['macro_f1']:.4f}")
    print(f"  Negative F1 {test_metrics['negative_f1']:.4f}, "
          f"positive F1 {test_metrics['positive_f1']:.4f}")

    cm = confusion_counts(test_metrics['predictions'], test_metrics['labels'])

    # Bootstrap CIs quantify test-set sampling uncertainty on this trained model
    test_metrics['confidence_intervals'] = bootstrap_metric_cis(
        test_metrics['predictions'], test_metrics['labels']
    )

    if write_artifacts:
        if variant == "finetuned":
            artifacts.write_finetuned_metrics(test_metrics, cm, best_val_f1)
        else:
            artifacts.write_sentiment_metrics(test_metrics, cm, best_val_f1)
            errors = collect_misclassified(
                test_texts, test_labels, test_metrics['predictions']
            )
            artifacts.write_sentiment_errors(errors)
            oov_analysis = oov_error_analysis(
                test_texts, test_labels, test_metrics['predictions'], tokenizer
            )
            artifacts.write_sentiment_oov_error(oov_analysis)
            print(f"OOV vs error correlation: "
                  f"{oov_analysis['correlation_oov_vs_error']}")

    history_path = os.path.join(config.checkpoint_dir, 'sentiment_training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    results = {
        'test_metrics': test_metrics,
        'best_val_f1': best_val_f1,
        'config': {
            'freeze_base_model': config.freeze_base_model,
            'pooling_strategy': config.pooling_strategy,
            'hidden_dim': config.hidden_dim,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs
        }
    }

    results_path = os.path.join(config.checkpoint_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_path}")
    print(f"Best validation macro-F1 {best_val_f1:.4f}; "
          f"test macro-F1 {test_metrics['macro_f1']:.4f}")

    return model


@torch.no_grad()
def evaluate_sentiment_test(
    config: Optional[SentimentConfig] = None,
    sentiment_model_path: Optional[str] = None,
    base_model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None
) -> Optional[Dict]:
    """Evaluate an already-trained sentiment classifier on the test split.

    Rebuilds the head using the pooling/hidden settings stored in the sentiment
    checkpoint so it matches the trained weights, then reports test metrics.
    Backs the pipeline's --eval-only path. Returns None if artifacts are missing.
    """
    if config is None:
        config = SentimentConfig()
    sentiment_model_path = str(sentiment_model_path or config.best_model_path)
    base_model_path = str(base_model_path or BEST_MODEL_PATH)
    tokenizer_path = str(tokenizer_path or TOKENIZER_VOCAB_PATH)

    for path, label in (
        (sentiment_model_path, "sentiment checkpoint"),
        (base_model_path, "base checkpoint"),
        (tokenizer_path, "tokenizer"),
    ):
        if not os.path.exists(path):
            print(f"Missing {label} at {path}. Train the classifier first.")
            return None

    checkpoint = torch.load(sentiment_model_path, map_location='cpu', weights_only=False)

    base_model = _load_base_model(base_model_path)

    tokenizer = Qwen3SmallTokenizer(vocab_size=20000)
    tokenizer.load_vocab(tokenizer_path)

    # Prefer the settings the checkpoint was trained with when available
    trained_config = checkpoint.get('config')
    pooling = getattr(trained_config, 'pooling_strategy', config.pooling_strategy)
    hidden_dim = getattr(trained_config, 'hidden_dim', config.hidden_dim)
    dropout = getattr(trained_config, 'dropout', config.dropout)

    model = SentimentClassifier(
        base_model=base_model,
        num_classes=2,
        hidden_dim=hidden_dim,
        dropout=dropout,
        pooling_strategy=pooling,
        freeze_base=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    texts, _, labels = load_emotions_data(config.data_file)
    _, _, _, _, test_texts, test_labels = split_data(
        texts, labels, config.train_split, config.val_split,
        config.test_split, config.random_seed
    )

    test_dataset = EmotionsDataset(test_texts, test_labels, tokenizer, max_length=128)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    test_metrics = evaluate(model, test_loader, config.device)

    print("=" * 70)
    print("Sentiment test evaluation")
    print("=" * 70)
    print(f"Accuracy {test_metrics['accuracy']:.4f}, "
          f"macro-F1 {test_metrics['macro_f1']:.4f}")
    print(f"  Negative F1 {test_metrics['negative_f1']:.4f}, "
          f"positive F1 {test_metrics['positive_f1']:.4f}")

    return test_metrics
