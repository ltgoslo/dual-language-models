# takes in the input directory, output directory, path to the tokenizer, and the max sequence length
# the input directory is the directory containing N sharded jsonl files
# the output directory is the directory where the each file is tokenized
from __future__ import annotations

from tokenizers import Tokenizer
import json
import torch
from tqdm import tqdm
from pathlib import Path


def tokenize(tokenizer: Tokenizer, text: str) -> torch.Tensor:
    """
    Tokenizes a text using a given tokenizer.
    """
    text = text.strip()
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    ids = torch.tensor(ids, dtype=torch.int32)

    return ids


def tokenize_shard(tokenizer: Tokenizer, input_path: Path, output_path: Path, output_valid_path: Path, max_size: int, verbose=False) -> None:
    """
    Takes an input path for a shard and saves its tokenized
    version into a given output path.
    """
    tokenized_documents = []
    tokenizer_documents_validation = []
    n_words, n_subwords = 0, 0
    for i, line in enumerate(tqdm(input_path.open("rt"), desc=f"Tokenizing {input_path}", disable=not verbose)):
        document = json.loads(line).strip()
        tokenized_document = tokenize(tokenizer, document)

        if n_subwords >= max_size:
            tokenizer_documents_validation.append(tokenized_document)
        else:
            tokenized_documents.append(tokenized_document)
            n_subwords += len(tokenized_document)
            n_words += len(document.split())

        if verbose and i == 0:
            print("Example tokenized document:")
            print(document)
            for token in tokenized_document:
                print(tokenizer.decode([token]))
            print(flush=True)

        if n_subwords >= max_size:
            tokenized_documents[-1] = tokenized_documents[-1][:tokenized_documents[-1].size(0) - (n_subwords - max_size)]

            if verbose:
                print(f"Reached max size of {max_size} subwords, truncating the last document.")
                print(f"Tokens / words ration: {n_subwords / n_words:.2f}")
            n_subwords = max_size

    torch.save(tokenized_documents, output_path)
    if output_valid_path is not None:
        torch.save(tokenizer_documents_validation, output_valid_path)

    if verbose:
        print(f"Tokenized {len(tokenized_documents)} documents with {n_subwords} subwords in total")
