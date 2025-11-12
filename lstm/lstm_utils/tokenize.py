from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from pathlib import Path
from typing import Iterable, Optional, List

from lstm_utils.load_dataset import load_data


def train_bpe_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 30000,
    min_frequency: int = 2,
    special_tokens: Optional[List[str]] = None,
) -> Tokenizer:
    """Обучаем BPE токенизатор на текстах."""
    if special_tokens is None:
        special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    texts_list = list(texts)
    tokenizer.train_from_iterator(texts_list, trainer=trainer)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, out_dir: str | Path) -> PreTrainedTokenizerFast:
    """Сохранение обученного токенизатора"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_json = out_dir / "tokenizer.json"
    tokenizer.save(str(tok_json))

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_json),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        mask_token="<mask>",
    )

    fast_tok.save_pretrained(str(out_dir))
    print(f"Tokenizer saved to folder: {str(out_dir)}")


if __name__ == "__main__":
    
    ds = load_data()
    
    a = [elem for elem in ds['train']['text']]
    b = [elem for elem in ds['test']['text']]
    c = a + b
    
    tokenizer = train_bpe_tokenizer(
        texts=c,
        vocab_size=5000,
        min_frequency=3
    )
    
    save_tokenizer(tokenizer, out_dir="../own_tokenizer")
