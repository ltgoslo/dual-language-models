import argparse
import json
import requests
from tqdm import tqdm
import zstandard as zstd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shards", type=int, default=256)
    parser.add_argument("--n_input_shards", type=int, default=160)
    parser.add_argument("--url_path", type=str, default="https://data.hplt-project.org/two/cleaned/eng_Latn/{}.jsonl.zst")
    parser.add_argument("--intermediate_path", type=str, default="data/hplt_2_8b_raw/{}.jsonl.zst")
    parser.add_argument("--output_path", type=str, default="data/hplt_2_32b_text_shards/{}.jsonl")
    parser.add_argument("--total_size", type=int, default=2048*2048*2048*4)
    return parser.parse_args()


def iter_zst_lines(file_path, encoding='utf-8'):
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(f)

        buffer = b""
        while True:
            chunk = stream_reader.read(8192)
            if not chunk:
                break
            buffer += chunk
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                try:
                    yield line.decode(encoding)
                except UnicodeDecodeError:
                    return  # stop on malformed line at truncation
        # Optional: yield last line if it looks complete
        if buffer.strip():
            try:
                yield buffer.decode(encoding)
            except UnicodeDecodeError:
                pass  # ignore partial/broken line


if __name__ == "__main__":
    args = parse_args()

    for i in tqdm(range(args.n_input_shards), desc="Downloading input shards"):
        url_path = args.url_path.format(i + 1)
        output_path = args.intermediate_path.format(i)

        bytes_to_download = args.total_size // args.n_input_shards * 8  # assuming 8 bytes per word

        headers = {
            "Range": f"bytes=0-{bytes_to_download - 1}"
        }

        response = requests.get(url_path, headers=headers, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    # Open documents
    input_files = [iter_zst_lines(args.intermediate_path.format(i)) for i in range(args.n_input_shards)]
    output_files = [open(args.output_path.format(i), "w") for i in range(args.n_shards)]

    # Shard files
    shard_lengths = [0 for _ in range(args.n_shards)]
    max_shard_length = args.total_size // args.n_shards

    input_index, output_index = 0, 0
    while any(length < max_shard_length for length in shard_lengths):
        line = next(input_files[input_index])
        if not line:
            input_index = (input_index + 1) % len(input_files)
            continue

        sample = json.loads(line)
        if not sample.get("text"):
            input_index = (input_index + 1) % len(input_files)
            continue

        text = sample["text"].strip()
        if not text:
            input_index = (input_index + 1) % len(input_files)
            continue

        while shard_lengths[output_index] >= max_shard_length:
            output_index = (output_index + 1) % len(output_files)

        output_files[output_index].write(f"{json.dumps(text, ensure_ascii=False)}\n")
        shard_lengths[output_index] += len(text.split())

        input_index = (input_index + 1) % len(input_files)
        output_index = (output_index + 1) % len(output_files)

    # Close files
    for f in input_files:
        f.close()
    for f in output_files:
        f.close()
