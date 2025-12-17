from __future__ import annotations

from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer, pre_tokenizers, decoders, normalizers, Regex, processors


def initialize_tokenizer(args):
    start_of_text_symbol = "<s>"
    end_of_text_symbol = "</s>"
    unk_symbol = "<unk>"
    mask_symbol = "<mask>"
    pad_symbol = "<pad>"

    special_tokens = [unk_symbol, start_of_text_symbol, end_of_text_symbol, pad_symbol, mask_symbol]
    special_tokens += [f"<special_{i}>" for i in range(11)]

    tokenizer = Tokenizer(BPE(
        unk_token=unk_symbol,
        byte_fallback=False,
        fuse_unk=False,
        ignore_merges=True
    ))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.Prepend(" "),
        normalizers.NFC(),
        normalizers.Replace(Regex("\n"), '\n '),
        normalizers.Replace(Regex(" *\n"), '\n'),
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(
            Regex("[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*| ?\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"),
            behavior="isolated",
            invert=False
        ),
        pre_tokenizers.ByteLevel(
            add_prefix_space=False, use_regex=False, trim_offsets=True
        )
    ])

    tokenizer.decoder = decoders.Sequence([
        decoders.ByteLevel(add_prefix_space=False, use_regex=False),
        decoders.Strip(' ', 1, 0),
        decoders.Replace("\n ", "\n")
    ])

    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{start_of_text_symbol} $A",
        pair=f"{start_of_text_symbol} $A {start_of_text_symbol} $B",
        special_tokens=[
            (start_of_text_symbol, 1),
        ]
    )

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    return tokenizer, trainer
