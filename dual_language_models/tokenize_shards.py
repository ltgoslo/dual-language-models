import argparse
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm

from dual_language_models.tokenization.tokenize_shards import tokenize_shard


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shards", type=int, default=256)
    parser.add_argument("--input_path", type=str, default="data/hplt_2_32b_text_shards/{}.jsonl")
    parser.add_argument("--output_path", type=str, default="data/hplt_2_32b_token_shards/{}.bin")
    parser.add_argument("--output_valid_path", type=str, default="data/hplt_2_32b_valid_token_shards/{}.bin")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizers/tokenizer.json")
    parser.add_argument("--total_size", type=int, default=32*1024*1024*1024)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    for i in tqdm(range(args.n_shards)):
        input_file = args.input_path.format(i)
        output_file = args.output_path.format(i)
        output_valid_file = args.output_valid_path.format(i)

        tokenize_shard(
            tokenizer,
            Path(input_file),
            Path(output_file),
            Path(output_valid_file),
            max_size=args.total_size // args.n_shards,
            verbose=True
        )

    print("Tokenization complete.")


if __name__ == "__main__":
    main()
