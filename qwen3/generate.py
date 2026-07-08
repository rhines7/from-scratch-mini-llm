"""
Autoregressive text generation for the trained model.

Implements greedy decoding plus temperature, top-k, and nucleus (top-p)
sampling, with optional repetition and no-repeat-ngram penalties. A KV cache
makes single-prompt generation incremental. Also provides a checkpoint loader
and an interactive REPL used by the pipeline's --interactive flag.
"""

import os
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .architecture import Qwen3Model, Qwen3Config
from .tokenizer import Qwen3SmallTokenizer
from .config import BEST_MODEL_PATH, TOKENIZER_VOCAB_PATH


@dataclass
class GenerationConfig:
    """Sampling and decoding settings for generation."""
    # Length
    max_new_tokens: int = 100
    min_new_tokens: int = 0

    # Sampling strategy
    do_sample: bool = True          # False selects greedy decoding
    temperature: float = 1.0
    top_k: int = 50                 # 0 disables top-k
    top_p: float = 0.9              # 1.0 disables nucleus filtering

    # Repetition control
    repetition_penalty: float = 1.0  # 1.0 means no penalty
    no_repeat_ngram_size: int = 0    # 0 disables

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    use_cache: bool = True

    device: str = "cpu"

    def __post_init__(self):
        assert self.max_new_tokens > 0, "max_new_tokens must be positive"
        assert self.temperature > 0, "temperature must be positive"
        assert 0 <= self.top_p <= 1, "top_p must be in [0, 1]"
        assert self.top_k >= 0, "top_k must be non-negative"
        assert self.repetition_penalty > 0, "repetition_penalty must be positive"


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float
) -> torch.Tensor:
    """Divide (or multiply) logits of already-seen tokens to discourage repetition.

    Following the CTRL convention, positive logits are divided and negative
    logits multiplied so a penalty > 1 always lowers the token's probability.
    """
    if penalty == 1.0:
        return logits

    batch_size, vocab_size = logits.shape

    for i in range(batch_size):
        for token_id in set(generated_ids[i].tolist()):
            if logits[i, token_id] < 0:
                logits[i, token_id] *= penalty
            else:
                logits[i, token_id] /= penalty

    return logits


def apply_no_repeat_ngram(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    ngram_size: int
) -> torch.Tensor:
    """Ban tokens that would complete a previously seen n-gram."""
    if ngram_size <= 0 or generated_ids.shape[1] < ngram_size - 1:
        return logits

    batch_size = logits.shape[0]

    for i in range(batch_size):
        prefix = generated_ids[i, -(ngram_size - 1):].tolist()

        gen_tokens = generated_ids[i].tolist()
        for j in range(len(gen_tokens) - ngram_size + 1):
            if gen_tokens[j:j + ngram_size - 1] == prefix:
                banned_token = gen_tokens[j + ngram_size - 1]
                logits[i, banned_token] = float('-inf')

    return logits


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only the top_k highest logits; set the rest to -inf."""
    if top_k <= 0:
        return logits

    top_k = min(top_k, logits.size(-1))

    top_k_values, _ = torch.topk(logits, top_k, dim=-1)

    # Threshold at the k-th largest value per row
    min_top_k = top_k_values[:, -1:].expand_as(logits)
    logits = torch.where(logits < min_top_k, torch.full_like(logits, float('-inf')), logits)

    return logits


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens with cumulative prob >= top_p."""
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p

    # Shift right so the first token past the threshold is still kept
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )

    logits = logits.masked_fill(indices_to_remove, float('-inf'))

    return logits


def sample_next_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    generated_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Pick the next token given the configured decoding strategy."""
    if generated_ids is not None and config.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated_ids, config.repetition_penalty)

    if generated_ids is not None and config.no_repeat_ngram_size > 0:
        logits = apply_no_repeat_ngram(logits, generated_ids, config.no_repeat_ngram_size)

    if not config.do_sample:
        return torch.argmax(logits, dim=-1)

    logits = logits / config.temperature

    if config.top_k > 0:
        logits = top_k_filtering(logits, config.top_k)

    if config.top_p < 1.0:
        logits = top_p_filtering(logits, config.top_p)

    probs = F.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

    return next_tokens


@torch.no_grad()
def generate(
    model: Qwen3Model,
    tokenizer: Qwen3SmallTokenizer,
    prompt: str,
    config: GenerationConfig
) -> str:
    """Generate a continuation for a single prompt, returning prompt plus text."""
    model.eval()
    device = config.device
    model = model.to(device)

    # encode() prepends BOS, which is what the model saw during training
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    attention_mask = torch.ones_like(input_ids)

    generated_ids = input_ids.clone()
    past_key_values = None

    for step in range(config.max_new_tokens):
        # With a warm cache, feed only the last token
        if config.use_cache and past_key_values is not None:
            model_input_ids = generated_ids[:, -1:]
            model_attention_mask = attention_mask
        else:
            model_input_ids = generated_ids
            model_attention_mask = attention_mask

        outputs = model(
            input_ids=model_input_ids,
            attention_mask=model_attention_mask,
            past_key_values=past_key_values if config.use_cache else None,
            use_cache=config.use_cache,
            return_dict=True
        )

        next_token_logits = outputs['logits'][:, -1, :]

        if config.use_cache:
            past_key_values = outputs.get('past_key_values')

        next_token = sample_next_token(next_token_logits, config, generated_ids)

        if next_token.item() == config.eos_token_id and step >= config.min_new_tokens:
            break

        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)],
            dim=1
        )

    return tokenizer.decode(generated_ids[0].tolist())


@torch.no_grad()
def batch_generate(
    model: Qwen3Model,
    tokenizer: Qwen3SmallTokenizer,
    prompts: List[str],
    config: GenerationConfig
) -> List[str]:
    """Generate continuations for several prompts padded to a common length.

    This recomputes the full sequence each step (no cache) because prompts have
    different lengths; it favors simplicity over speed for short demos.
    """
    model.eval()
    device = config.device
    model = model.to(device)

    encoded_prompts = [
        tokenizer.encode(prompt, add_special_tokens=True)
        for prompt in prompts
    ]

    max_prompt_len = max(len(p) for p in encoded_prompts)

    batch_size = len(prompts)
    input_ids = torch.full(
        (batch_size, max_prompt_len),
        config.pad_token_id,
        dtype=torch.long,
        device=device
    )

    prompt_lengths = []
    for i, prompt_ids in enumerate(encoded_prompts):
        input_ids[i, :len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.long)
        prompt_lengths.append(len(prompt_ids))

    attention_mask = torch.zeros_like(input_ids)
    for i, length in enumerate(prompt_lengths):
        attention_mask[i, :length] = 1

    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

    generated_ids = input_ids.clone()

    for step in range(config.max_new_tokens):
        outputs = model(
            input_ids=generated_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        next_token_logits = outputs['logits'][:, -1, :]

        next_tokens = sample_next_token(next_token_logits, config, generated_ids)

        if step >= config.min_new_tokens:
            # Finished sequences emit padding from here on
            next_tokens = next_tokens * unfinished_sequences + \
                config.pad_token_id * (1 - unfinished_sequences)

            unfinished_sequences = unfinished_sequences.mul(
                (next_tokens != config.eos_token_id).long()
            )

        generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, unfinished_sequences.unsqueeze(1)],
            dim=1
        )

        if unfinished_sequences.sum() == 0:
            break

    generated_texts = []
    for i in range(batch_size):
        text = tokenizer.decode(generated_ids[i].tolist())
        generated_texts.append(text)

    return generated_texts


def test_generation_quality(
    model: Qwen3Model,
    tokenizer: Qwen3SmallTokenizer,
    config: GenerationConfig,
    test_prompts: Optional[List[str]] = None
) -> Dict:
    """Generate from a set of prompts and print each continuation."""
    if test_prompts is None:
        test_prompts = [
            "Once upon a time",
            "One day, a little",
            "There was a",
            "A boy and a girl",
            "In a big forest"
        ]

    print("=" * 70)
    print("Text generation")
    print("=" * 70)

    results = {
        'config': {
            'max_new_tokens': config.max_new_tokens,
            'temperature': config.temperature,
            'top_k': config.top_k,
            'top_p': config.top_p,
            'repetition_penalty': config.repetition_penalty,
            'do_sample': config.do_sample
        },
        'generations': []
    }

    for i, prompt in enumerate(test_prompts):
        generated_text = generate(model, tokenizer, prompt, config)

        print(f"\n[{i + 1}/{len(test_prompts)}] Prompt: \"{prompt}\"")
        print(generated_text)

        results['generations'].append({
            'prompt': prompt,
            'generated_text': generated_text,
            'length': len(generated_text.split())
        })

    return results


def compare_sampling_strategies(
    model: Qwen3Model,
    tokenizer: Qwen3SmallTokenizer,
    prompt: str = "Once upon a time",
    max_new_tokens: int = 50,
    device: str = "cpu"
) -> Dict:
    """Run the same prompt through several decoding strategies for comparison."""
    print("=" * 70)
    print(f"Sampling strategy comparison for: \"{prompt}\"")
    print("=" * 70)

    strategies = [
        ("Greedy", GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            device=device
        )),
        ("Temperature 0.7", GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_k=0,
            top_p=1.0,
            device=device
        )),
        ("Top-k (k=50)", GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
            device=device
        )),
        ("Top-p (p=0.9)", GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            device=device
        )),
        ("Combined (T=0.8, k=40, p=0.9, rep=1.2)", GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,
            device=device
        ))
    ]

    results = {'prompt': prompt, 'strategies': []}

    for strategy_name, config in strategies:
        generated_text = generate(model, tokenizer, prompt, config)

        print(f"\n{strategy_name}:")
        print(generated_text)

        results['strategies'].append({
            'name': strategy_name,
            'text': generated_text
        })

    return results


def load_model_and_tokenizer(
    checkpoint_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None
) -> Tuple[Optional[Qwen3Model], Optional[Qwen3SmallTokenizer]]:
    """Load a trained model and its tokenizer from disk.

    The model architecture is rebuilt from the config stored in the checkpoint,
    so this works regardless of the size used at training time. Returns
    (None, None) with a message when either file is missing.
    """
    checkpoint_path = str(checkpoint_path or BEST_MODEL_PATH)
    tokenizer_path = str(tokenizer_path or TOKENIZER_VOCAB_PATH)

    if not os.path.exists(checkpoint_path):
        print(f"Trained model not found at {checkpoint_path}. Run pre-training first.")
        return None, None

    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}. Run data preparation first.")
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    tokenizer = Qwen3SmallTokenizer(vocab_size=20000)
    tokenizer.load_vocab(tokenizer_path)

    model_config = Qwen3Config(**checkpoint['model_config'])
    model = Qwen3Model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model ({model.count_parameters()['actual_millions']:.2f}M parameters) "
          f"and tokenizer ({len(tokenizer)} tokens)")

    return model, tokenizer


def interactive_generate(
    model: Optional[Qwen3Model] = None,
    tokenizer: Optional[Qwen3SmallTokenizer] = None
):
    """Prompt-driven REPL for exploring the model.

    Commands: 'temp X', 'tokens X', 'greedy', 'sample', 'quit'. Loads the
    default checkpoint if a model and tokenizer are not passed in.
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        return

    print("\nInteractive generation. Commands: 'temp X', 'tokens X', "
          "'greedy', 'sample', 'quit'.")

    max_tokens = 50
    temperature = 0.8
    do_sample = True

    while True:
        try:
            user_input = input("Prompt: ").strip()

            if user_input.lower() == 'quit':
                break
            elif user_input.lower().startswith('temp '):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"Temperature set to {temperature}")
                except (IndexError, ValueError):
                    print("Usage: temp 0.5")
                continue
            elif user_input.lower().startswith('tokens '):
                try:
                    max_tokens = int(user_input.split()[1])
                    print(f"Max tokens set to {max_tokens}")
                except (IndexError, ValueError):
                    print("Usage: tokens 100")
                continue
            elif user_input.lower() == 'greedy':
                do_sample = False
                print("Greedy decoding")
                continue
            elif user_input.lower() == 'sample':
                do_sample = True
                print("Sampling")
                continue
            elif not user_input:
                continue

            config = GenerationConfig(
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.1,
                device="cpu"
            )

            print(generate(model, tokenizer, user_input, config))
            print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
