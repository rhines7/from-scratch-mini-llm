"""
Word-based tokenizer for Qwen3-small.

The tokenizer is trained from scratch on the TinyStories corpus rather than
reusing a pre-trained vocabulary. TinyStories has a small, regular vocabulary,
so word-level tokenization with a frequency threshold gives high coverage
without the complexity of subword merges.

Vocabulary is frequency-ranked; special tokens occupy the first four IDs and
follow Qwen3 naming. Vocab can be saved and reloaded so downstream stages reuse
the exact mapping produced during data preparation.
"""

import json
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union
import torch


class Qwen3SmallTokenizer:
    """Word-level tokenizer with frequency-ranked vocabulary and Qwen3 special tokens."""

    # Qwen3 special token conventions
    SPECIAL_TOKENS = {
        'pad': '<|pad|>',
        'bos': '<|begin_of_text|>',
        'eos': '<|end_of_text|>',
        'unk': '<|unknown|>'
    }

    def __init__(
        self,
        vocab_size: int = 20000,
        min_frequency: int = 2,
        lowercase: bool = False,
        special_tokens: Optional[Dict[str, str]] = None
    ):
        """Configure the tokenizer.

        vocab_size is the target cap including special tokens; the realized size
        depends on how many tokens clear min_frequency in the training corpus.
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.lowercase = lowercase

        self.special_tokens = special_tokens or self.SPECIAL_TOKENS.copy()

        # Vocabulary mappings (populated in build_vocab)
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}

        # Special tokens occupy IDs 0-3
        self.pad_token = self.special_tokens['pad']
        self.bos_token = self.special_tokens['bos']
        self.eos_token = self.special_tokens['eos']
        self.unk_token = self.special_tokens['unk']

        self.pad_id: int = 0
        self.bos_id: int = 1
        self.eos_id: int = 2
        self.unk_id: int = 3

        self._initialize_special_tokens()

        self.vocab_coverage: Optional[float] = None
        self.is_trained: bool = False

    def _initialize_special_tokens(self):
        """Reserve special token IDs at the start of the vocabulary."""
        self.vocab[self.pad_token] = self.pad_id
        self.vocab[self.bos_token] = self.bos_id
        self.vocab[self.eos_token] = self.eos_id
        self.vocab[self.unk_token] = self.unk_id

        self.inverse_vocab[self.pad_id] = self.pad_token
        self.inverse_vocab[self.bos_id] = self.bos_token
        self.inverse_vocab[self.eos_id] = self.eos_token
        self.inverse_vocab[self.unk_id] = self.unk_token

    def _tokenize_text(self, text: str) -> List[str]:
        """Split into words and standalone punctuation, normalizing whitespace."""
        text = ' '.join(text.split())

        if self.lowercase:
            text = text.lower()

        # Words/numbers, or single punctuation marks
        pattern = r"\w+|[^\w\s]"
        tokens = re.findall(pattern, text)

        return tokens

    def build_vocab(
        self,
        texts: List[str],
        verbose: bool = True
    ) -> Dict[str, int]:
        """Build the vocabulary from a training corpus.

        Tokens are counted, filtered by min_frequency, ranked by frequency, and
        assigned IDs after the reserved special tokens. Coverage is the share of
        corpus token occurrences represented by the final vocabulary.
        """
        if verbose:
            print(f"Building vocabulary from {len(texts):,} texts...")

        token_counter = Counter()
        total_tokens = 0

        for text in texts:
            tokens = self._tokenize_text(text)
            token_counter.update(tokens)
            total_tokens += len(tokens)

        if verbose:
            print(f"Found {len(token_counter):,} unique tokens")
            print(f"Total tokens: {total_tokens:,}")

        filtered_tokens = {
            token: count for token, count in token_counter.items()
            if count >= self.min_frequency
        }

        if verbose:
            print(f"After filtering (min_freq={self.min_frequency}): {len(filtered_tokens):,} tokens")

        sorted_tokens = sorted(
            filtered_tokens.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Reserve space for special tokens already assigned IDs 0-3
        num_special = len(self.special_tokens)
        max_vocab_tokens = self.vocab_size - num_special

        selected_tokens = sorted_tokens[:max_vocab_tokens]

        if verbose:
            print(f"Selected top {len(selected_tokens):,} tokens for vocabulary")

        current_id = num_special
        for token, count in selected_tokens:
            if token not in self.vocab:
                self.vocab[token] = current_id
                self.inverse_vocab[current_id] = token
                current_id += 1

        vocab_token_count = sum(
            count for token, count in token_counter.items()
            if token in self.vocab
        )
        self.vocab_coverage = (vocab_token_count / total_tokens) * 100

        self.is_trained = True

        if verbose:
            print("\nVocabulary statistics:")
            print(f"  Final vocabulary size: {len(self.vocab):,}")
            print(f"  Coverage: {self.vocab_coverage:.2f}%")
            print(f"  OOV rate: {100 - self.vocab_coverage:.2f}%")
            print(f"  Most common tokens: {[t for t, _ in selected_tokens[:10]]}")

        return {
            'vocab_size': len(self.vocab),
            'coverage': self.vocab_coverage,
            'total_tokens': total_tokens,
            'unique_tokens': len(token_counter),
            'min_frequency': self.min_frequency
        }

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        return_attention_mask: bool = False
    ) -> Union[List[int], Tuple[List[int], List[int]]]:
        """Convert text to token IDs, with optional special tokens, truncation, and padding."""
        if not self.is_trained:
            raise ValueError("Tokenizer not trained. Call build_vocab() first.")

        tokens = self._tokenize_text(text)

        # Out-of-vocabulary tokens map to unk_id
        token_ids = [
            self.vocab.get(token, self.unk_id) for token in tokens
        ]

        if add_special_tokens:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]

        if max_length is not None and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            # Preserve a trailing EOS when special tokens were requested
            if add_special_tokens:
                token_ids[-1] = self.eos_id

        attention_mask = [1] * len(token_ids)

        if padding and max_length is not None:
            padding_length = max_length - len(token_ids)
            if padding_length > 0:
                token_ids.extend([self.pad_id] * padding_length)
                attention_mask.extend([0] * padding_length)

        if return_attention_mask:
            return token_ids, attention_mask
        return token_ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization: bool = True
    ) -> str:
        """Convert token IDs back to text, optionally dropping special tokens and fixing spacing."""
        tokens = []
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        if skip_special_tokens:
            special_ids.add(self.unk_id)

        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            token = self.inverse_vocab.get(token_id, self.unk_token)
            tokens.append(token)

        text = ' '.join(tokens)

        if clean_up_tokenization:
            # Re-attach punctuation and normalize quote spacing
            text = re.sub(r'\s+([.,!?;:])', r'\1', text)
            text = re.sub(r'\s+"', r'"', text)
            text = re.sub(r'"\s+', r'" ', text)
            text = ' '.join(text.split())

        return text

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        return_tensors: bool = False
    ) -> Dict[str, Union[List[List[int]], torch.Tensor]]:
        """Encode multiple texts, padding to a common length for batching."""
        all_token_ids = []
        all_attention_masks = []

        for text in texts:
            token_ids, attention_mask = self.encode(
                text,
                add_special_tokens=add_special_tokens,
                return_attention_mask=True
            )
            all_token_ids.append(token_ids)
            all_attention_masks.append(attention_mask)

        if padding:
            if max_length is None:
                max_length = max(len(ids) for ids in all_token_ids)

            for i in range(len(all_token_ids)):
                padding_length = max_length - len(all_token_ids[i])
                if padding_length > 0:
                    all_token_ids[i].extend([self.pad_id] * padding_length)
                    all_attention_masks[i].extend([0] * padding_length)
                elif padding_length < 0:
                    all_token_ids[i] = all_token_ids[i][:max_length]
                    all_attention_masks[i] = all_attention_masks[i][:max_length]

        if return_tensors:
            all_token_ids = torch.tensor(all_token_ids, dtype=torch.long)
            all_attention_masks = torch.tensor(all_attention_masks, dtype=torch.long)

        return {
            'input_ids': all_token_ids,
            'attention_mask': all_attention_masks
        }

    def save_vocab(self, path: str):
        """Serialize vocabulary and settings to JSON so later stages reuse them."""
        vocab_data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency,
            'lowercase': self.lowercase,
            'special_tokens': self.special_tokens,
            'vocab_coverage': self.vocab_coverage,
            'is_trained': self.is_trained
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)

        print(f"Vocabulary saved to {path}")

    def load_vocab(self, path: str):
        """Restore vocabulary and settings from a saved JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        self.vocab = {k: int(v) for k, v in vocab_data['vocab'].items()}
        self.vocab_size = vocab_data['vocab_size']
        self.min_frequency = vocab_data['min_frequency']
        self.lowercase = vocab_data['lowercase']
        self.special_tokens = vocab_data['special_tokens']
        self.vocab_coverage = vocab_data.get('vocab_coverage')
        self.is_trained = vocab_data['is_trained']

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        self.pad_token = self.special_tokens['pad']
        self.bos_token = self.special_tokens['bos']
        self.eos_token = self.special_tokens['eos']
        self.unk_token = self.special_tokens['unk']

        print(f"Vocabulary loaded from {path}")
        print(f"  Vocabulary size: {len(self.vocab):,}")
        if self.vocab_coverage:
            print(f"  Coverage: {self.vocab_coverage:.2f}%")

    def get_vocab_size(self) -> int:
        """Return the current vocabulary size."""
        return len(self.vocab)

    def analyze_text(self, text: str) -> Dict:
        """Report token counts and OOV rate for a single text (first 20 tokens shown)."""
        tokens = self._tokenize_text(text)
        token_ids = self.encode(text, add_special_tokens=False)

        unk_count = sum(1 for tid in token_ids if tid == self.unk_id)

        return {
            'num_tokens': len(tokens),
            'num_unique_tokens': len(set(tokens)),
            'num_unk_tokens': unk_count,
            'unk_rate': (unk_count / len(tokens) * 100) if tokens else 0,
            'tokens': tokens[:20],
            'token_ids': token_ids[:20]
        }

    def coverage(self, texts: List[str]) -> Dict:
        """Aggregate token coverage / OOV rate over a corpus.

        Reports how much of an out-of-domain corpus the vocabulary covers. This
        is the metric that quantifies the domain gap when a tokenizer fit on one
        corpus (TinyStories) is applied to another (emotion sentences): a high
        OOV rate caps any downstream task built on these representations.
        """
        if not self.is_trained:
            raise ValueError("Tokenizer not trained. Call build_vocab() first.")

        total_tokens = 0
        oov_tokens = 0
        oov_types = set()
        seen_types = set()

        for text in texts:
            for token in self._tokenize_text(text):
                total_tokens += 1
                seen_types.add(token)
                if token not in self.vocab:
                    oov_tokens += 1
                    oov_types.add(token)

        oov_rate = (oov_tokens / total_tokens * 100) if total_tokens else 0.0

        return {
            "num_texts": len(texts),
            "total_tokens": total_tokens,
            "oov_tokens": oov_tokens,
            "oov_rate": oov_rate,
            "coverage": 100.0 - oov_rate,
            "unique_tokens": len(seen_types),
            "unique_oov_tokens": len(oov_types),
        }

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "not trained"
        coverage = f"{self.vocab_coverage:.2f}%" if self.vocab_coverage is not None else "N/A"
        return (
            f"Qwen3SmallTokenizer(vocab_size={len(self.vocab)}, "
            f"status={status}, coverage={coverage})"
        )


def create_tokenizer_from_texts(
    texts: List[str],
    vocab_size: int = 20000,
    min_frequency: int = 2,
    lowercase: bool = False,
    verbose: bool = True
) -> Qwen3SmallTokenizer:
    """Create and train a tokenizer in one call."""
    tokenizer = Qwen3SmallTokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        lowercase=lowercase
    )

    tokenizer.build_vocab(texts, verbose=verbose)

    return tokenizer
