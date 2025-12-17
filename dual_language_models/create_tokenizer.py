import argparse
import json
from tokenizers import Tokenizer
from dual_language_models.tokenization.init_tokenizer import initialize_tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BERT sharding')
    parser.add_argument('--num_shards', type=int, default=256, help='Number of shards to train on')
    parser.add_argument('--shard_path', type=str, default="data/hplt_2_32b_text_shards/{}.jsonl", help='Path to the shards')
    parser.add_argument('--vocab_path', type=str, default="tokenizers/tokenizer.json", help='Specify the output filename')
    parser.add_argument('--vocab_size', type=int, default=51_200, help='Number of subwords in the trained tokenizer')
    parser.add_argument('--min_frequency', type=int, default=1, help='Minimal number of occurences of every candidate subword')
    args = parser.parse_args()

    print("Initializing a BPE tokenizer", flush=True)
    tokenizer, trainer = initialize_tokenizer(args)

    print("Training the tokenizer", flush=True)

    def iterator():
        for shard in range(args.num_shards):
            for line in open(args.shard_path.format(shard), 'r'):
                text = json.loads(line).strip()
                if len(text) == 0:
                    continue
                yield text

    tokenizer.train_from_iterator(iterator(), trainer)

    print("Saving the tokenizer", flush=True)
    tokenizer.save(args.vocab_path)

    with open(args.vocab_path) as f:
        tokenizer_json = json.load(f)

    print("TEST")
    print("Trying to load the tokenizer...")
    tokenizer = Tokenizer.from_file(args.vocab_path)
    print("Success!")

    print("Samples from the tokenizer:")

    def test(tokenizer, text):
        subwords = tokenizer.encode(text).tokens
        return ' '.join(subwords)

    texts = [
        """One of the most impressive long term hobby projects is Robert's Rocket Project. He started building a 100 lbf liquid engine in 2001, fired a regeneratively cooled version in 2007, started building a regen 250 lbf in 2008.""",
        """what are examples of interfaces that allow you to manage sets of queries (SQL, splunk, lucene/elastic, xpath, whatever other language)?""",
        """### Increasingly seeing a big schism between what I think my research is & what others think it is. I don't do qualitative work and I'm not trained in anthro or theories of race or gender. I can't supervise students with these interests! I'm a sociophonetician who works on prosody!""",
        """The Northern Lights season is here... Taking these pictures is an art itself and requires preparation, so The Local spoke to an expert to find out how to take awe-inspiring snaps of the Northern Lights.""",
        """Some people have SOTA facial recognition abilities: "At the very upper end of the performance scale, a cohort of just 1-2% of the population are 'super-recognisers'-people who can memorise and recall unfamiliar faces, even after the briefest glimpse.\""""
    ]

    for text in texts:
        print(f"INPUT:  {text}\nTOKENS: {test(tokenizer, text)}\n", flush=True)
