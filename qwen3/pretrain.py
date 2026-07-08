"""
Pre-training loop for Qwen3-small.

Trains the model on TinyStories with a causal-LM objective. Design choices that
matter for reproducibility:

- AdamW with beta2 = 0.95 (Qwen3 uses a lower second-moment decay than the 0.999
  default, which stabilizes training on short sequences)
- Cosine learning-rate schedule with linear warmup
- Gradient accumulation to emulate a larger effective batch on CPU
- Periodic validation with best-checkpoint saving by validation loss

Defaults reproduce the committed run: a 512-wide, 6-layer model (about 19.5M
parameters) trained for 5 epochs on 10,000 stories.
"""

import os
import json
import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .architecture import Qwen3Model, Qwen3Config
from .tokenizer import Qwen3SmallTokenizer
from .data import prepare_tiny_stories_dataset, DataConfig
from .config import CHECKPOINT_DIR, LOG_DIR, BEST_MODEL_PATH, DEFAULT_SEED


@dataclass
class TrainingConfig:
    """Hyperparameters and paths for pre-training.

    beta2 defaults to 0.95 per Qwen3. Path defaults resolve to the repo's
    checkpoints/ and logs/ directories.
    """
    # Optional explicit sub-configs; sensible defaults are built if left None
    model_config: Optional[Qwen3Config] = None
    data_config: Optional[DataConfig] = None

    # Optimizer
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95  # Qwen3-specific (vs. the usual 0.999)
    adam_epsilon: float = 1e-8

    # Schedule
    warmup_steps: int = 200
    max_steps: Optional[int] = None  # Auto-calculated if None
    min_lr_ratio: float = 0.1

    # Duration
    num_epochs: int = 5

    # Gradients
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Logging and validation cadence
    log_every_n_steps: int = 50
    validate_every_n_steps: int = 200
    save_every_n_steps: int = 500

    # Paths
    checkpoint_dir: str = field(default_factory=lambda: str(CHECKPOINT_DIR))
    log_dir: str = field(default_factory=lambda: str(LOG_DIR))
    best_model_path: str = field(default_factory=lambda: str(BEST_MODEL_PATH))

    device: str = "cpu"

    random_seed: int = DEFAULT_SEED

    def __post_init__(self):
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert 0 < self.min_lr_ratio < 1, "min_lr_ratio must be in (0, 1)"
        assert self.gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


def default_model_config(vocab_size: int, max_seq_len: int) -> Qwen3Config:
    """Canonical Qwen3-small config (about 19.5M parameters) used for the run."""
    return Qwen3Config(
        vocab_size=vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        num_kv_heads=8,
        intermediate_size=1024,
        max_seq_len=max_seq_len,
    )


def get_cosine_schedule_with_warmup(
    optimizer: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
    num_cycles: float = 0.5
) -> LambdaLR:
    """Linear warmup then cosine decay to min_lr_ratio of the peak learning rate."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def compute_loss(
    model: Qwen3Model,
    batch: Dict[str, torch.Tensor],
    device: str = "cpu"
) -> Tuple[torch.Tensor, Dict]:
    """Cross-entropy over next-token predictions; returns loss and metrics.

    Inputs and targets are shifted by one so position t predicts token t+1.
    Padding positions carry label -100 and are ignored by the loss.
    """
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    outputs = model(
        input_ids=input_ids[:, :-1],
        attention_mask=attention_mask[:, :-1],
        return_dict=True
    )

    logits = outputs['logits']  # (batch, seq_len-1, vocab_size)
    targets = labels[:, 1:]

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )

    perplexity = torch.exp(loss)
    num_tokens = (targets != -100).sum().item()

    metrics = {
        'loss': loss.item(),
        'perplexity': perplexity.item(),
        'num_tokens': num_tokens
    }

    return loss, metrics


def validate(
    model: Qwen3Model,
    val_loader: DataLoader,
    device: str = "cpu",
    max_batches: Optional[int] = None
) -> Dict:
    """Token-weighted average loss and perplexity over the validation set."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches and batch_idx >= max_batches:
                break

            loss, metrics = compute_loss(model, batch, device)

            total_loss += metrics['loss'] * metrics['num_tokens']
            total_tokens += metrics['num_tokens']
            num_batches += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    avg_perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

    return {
        'val_loss': avg_loss,
        'val_perplexity': avg_perplexity,
        'num_batches': num_batches,
        'num_tokens': total_tokens
    }


def save_checkpoint(
    model: Qwen3Model,
    optimizer: AdamW,
    scheduler: LambdaLR,
    training_config: TrainingConfig,
    global_step: int,
    epoch: int,
    metrics: Dict,
    is_best: bool = False
):
    """Persist model/optimizer/scheduler state; also write best_model_path when is_best."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'training_config': asdict(training_config),
        'model_config': asdict(model.config),
        'global_step': global_step,
        'epoch': epoch,
        'metrics': metrics
    }

    checkpoint_path = os.path.join(
        training_config.checkpoint_dir,
        f'checkpoint_step_{global_step}.pt'
    )
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

    if is_best:
        torch.save(checkpoint, training_config.best_model_path)
        print(f"Best model saved to {training_config.best_model_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: Qwen3Model,
    optimizer: Optional[AdamW] = None,
    scheduler: Optional[LambdaLR] = None
) -> Dict:
    """Load weights (and optionally optimizer/scheduler state) from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  Step: {checkpoint.get('global_step', 'N/A')}, "
          f"epoch: {checkpoint.get('epoch', 'N/A')}")

    return checkpoint


def train_qwen3_small(
    training_config: TrainingConfig,
    resume_from: Optional[str] = None,
    write_artifacts: bool = True
) -> Qwen3Model:
    """Run the full pre-training loop and return the trained model.

    Prepares data, builds the model (defaulting to the canonical config when one
    is not supplied), then trains with cosine-scheduled AdamW, periodic
    validation, and best-checkpoint saving. A final validation runs if fewer
    than two occurred during training.

    write_artifacts is forwarded to data preparation so --quick smoke runs do not
    overwrite the committed dataset statistics.
    """
    print("=" * 70)
    print("Pre-training")
    print("=" * 70)

    torch.manual_seed(training_config.random_seed)

    print("\nPreparing dataset...")
    train_loader, val_loader, test_loader, tokenizer, data_stats = \
        prepare_tiny_stories_dataset(
            config=training_config.data_config,
            verbose=True,
            write_artifacts=write_artifacts
        )

    print("\nInitializing model...")
    if training_config.model_config is None:
        max_seq_len = training_config.data_config.max_seq_len if training_config.data_config else 256
        training_config.model_config = default_model_config(len(tokenizer), max_seq_len)

    model = Qwen3Model(training_config.model_config)
    model = model.to(training_config.device)

    param_count = model.count_parameters()
    print(f"Model parameters: {param_count['actual_millions']:.2f}M")

    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        betas=(training_config.adam_beta1, training_config.adam_beta2),
        eps=training_config.adam_epsilon,
        weight_decay=training_config.weight_decay
    )

    steps_per_epoch = len(train_loader) // training_config.gradient_accumulation_steps
    total_steps = steps_per_epoch * training_config.num_epochs

    if training_config.max_steps:
        total_steps = min(total_steps, training_config.max_steps)

    print(f"  Epochs {training_config.num_epochs}, "
          f"steps/epoch {steps_per_epoch}, total steps {total_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config.warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=training_config.min_lr_ratio
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = load_checkpoint(resume_from, model, optimizer, scheduler)
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        if 'metrics' in checkpoint:
            best_val_loss = checkpoint['metrics'].get('val_loss', float('inf'))

    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    training_history = {
        'train_loss': [],
        'train_perplexity': [],
        'val_loss': [],
        'val_perplexity': [],
        'learning_rates': [],
        'steps': []
    }

    model.train()
    optimizer.zero_grad()

    num_validations = 0

    for epoch in range(start_epoch, training_config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{training_config.num_epochs}")

        epoch_loss = 0.0
        epoch_tokens = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            loss, metrics = compute_loss(model, batch, training_config.device)

            # Scale for gradient accumulation
            loss = loss / training_config.gradient_accumulation_steps
            loss.backward()

            epoch_loss += metrics['loss']
            epoch_tokens += metrics['num_tokens']

            if (batch_idx + 1) % training_config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    training_config.max_grad_norm
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'ppl': f"{metrics['perplexity']:.2f}",
                    'lr': f"{current_lr:.2e}"
                })

                if global_step % training_config.log_every_n_steps == 0:
                    training_history['train_loss'].append(metrics['loss'])
                    training_history['train_perplexity'].append(metrics['perplexity'])
                    training_history['learning_rates'].append(current_lr)
                    training_history['steps'].append(global_step)

                if global_step % training_config.validate_every_n_steps == 0:
                    val_metrics = validate(model, val_loader, training_config.device)

                    print(f"\nStep {global_step} | val loss {val_metrics['val_loss']:.4f} | "
                          f"val ppl {val_metrics['val_perplexity']:.2f}")

                    training_history['val_loss'].append(val_metrics['val_loss'])
                    training_history['val_perplexity'].append(val_metrics['val_perplexity'])

                    is_best = val_metrics['val_loss'] < best_val_loss
                    if is_best:
                        best_val_loss = val_metrics['val_loss']

                    save_checkpoint(
                        model, optimizer, scheduler, training_config,
                        global_step, epoch, val_metrics, is_best
                    )

                    num_validations += 1
                    model.train()

                elif global_step % training_config.save_every_n_steps == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, training_config,
                        global_step, epoch, {'train_loss': metrics['loss']}, False
                    )

                if training_config.max_steps and global_step >= training_config.max_steps:
                    print(f"\nReached maximum steps ({training_config.max_steps})")
                    break

        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_ppl = math.exp(avg_epoch_loss)

        print(f"Epoch {epoch + 1} summary: "
              f"avg loss {avg_epoch_loss:.4f}, avg ppl {avg_epoch_ppl:.2f}, "
              f"{epoch_tokens:,} tokens")

    # Guarantee at least two validation checkpoints
    if num_validations < 2:
        val_metrics = validate(model, val_loader, training_config.device)
        print(f"\nFinal validation: loss {val_metrics['val_loss']:.4f}, "
              f"ppl {val_metrics['val_perplexity']:.2f}")

        is_best = val_metrics['val_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val_loss']

        save_checkpoint(
            model, optimizer, scheduler, training_config,
            global_step, training_config.num_epochs - 1, val_metrics, is_best
        )

        num_validations += 1

    history_path = os.path.join(training_config.log_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    print("\n" + "=" * 70)
    print(f"Training complete: {global_step} steps, "
          f"{num_validations} validations, best val loss {best_val_loss:.4f}")
    print("=" * 70)

    return model
