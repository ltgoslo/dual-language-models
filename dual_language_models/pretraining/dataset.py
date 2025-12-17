from __future__ import annotations
import torch
from typing import TYPE_CHECKING
import random

if TYPE_CHECKING:
    from tokenizers import Tokenizer
    from argparse import Namespace


class Datasetv2:

    def __init__(self: Datasetv2, dataset: str, tokenizer: Tokenizer, args: Namespace, seq_length: int, rank: int, seed: int, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.seq_length = seq_length
        self.n_special_tokens = args.n_special_tokens
        self.args = args
        self.shuffle = shuffle
        self.current_idx = 0
        self.iterations = 0
        self.seed = seed

        self.mask_index = tokenizer.token_to_id("<mask>")
        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        all_documents = torch.load(f"{dataset}/{rank:d}.bin", weights_only=False)
        total_num_tokens = sum(len(doc) for doc in all_documents)
        remaining_num_tokens = total_num_tokens // args.n_repetitions
        self.documents = []
        while remaining_num_tokens > 0:
            document = all_documents[len(self.documents)][:remaining_num_tokens]
            self.documents.append(document)
            remaining_num_tokens -= len(document)

        self.inputs, self.outputs, self.doc_ids = self.chunk()

        print(f"Dataset {dataset}/{rank:d} loaded", flush=True)

    def chunk(self: Datasetv2) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if self.shuffle:
            random.seed(self.seed + self.iterations)
            random.shuffle(self.documents)

        input_chunks = []
        output_chunks = []
        document_id_chunks = []

        input_ids = torch.LongTensor([])
        output_ids = torch.LongTensor([])
        document_ids = torch.LongTensor([])
        current_document_id = 0

        for document in self.documents:
            document = torch.cat([torch.LongTensor([self.cls_index]), document])

            input_ids = torch.cat([input_ids, document[:-1]])
            output_ids = torch.cat([output_ids, document[1:]])
            document_ids = torch.cat([document_ids, torch.ones_like(document[:-1], dtype=torch.int) * current_document_id])

            while len(input_ids) >= self.seq_length:
                input_chunks.append(input_ids[:self.seq_length])
                output_chunks.append(output_ids[:self.seq_length])
                document_id_chunks.append(document_ids[:self.seq_length])
                input_ids = input_ids[self.seq_length:]
                output_ids = output_ids[self.seq_length:]
                document_ids = torch.zeros_like(input_ids, dtype=torch.int)
                current_document_id = 0

            current_document_id += 1

        if self.shuffle:
            indices = list(range(len(input_chunks)))
            random.shuffle(indices)
            input_chunks = [input_chunks[i] for i in indices]
            output_chunks = [output_chunks[i] for i in indices]
            document_id_chunks = [document_id_chunks[i] for i in indices]

        return input_chunks, output_chunks, document_id_chunks

    def load_state(self: Datasetv2, dataset_state: dict[str, int]) -> None:
        self.current_idx = dataset_state["current_idx"]
        self.iterations = dataset_state["iterations"]
        self.inputs, self.outputs, self.doc_ids = self.chunk()

    def get_state(self: Datasetv2) -> dict[str, int]:
        return {
            "current_idx": self.current_idx,
            "iterations": self.iterations,
        }
    
    def __len__(self: Datasetv2) -> int:
        return len(self.inputs)


class MaskedDatasetv2(Datasetv2):

    def __init__(self: MaskedDatasetv2, dataset: str, tokenizer: Tokenizer, args: Namespace, seq_length: int, rank: int, shuffle: bool = True):
        super().__init__(dataset, tokenizer, args, seq_length, rank, args.seed + 12345, shuffle)

        self.masking_strategy = SpanMaskingStrategy(args.n_special_tokens, args.mask_random_p, args.mask_keep_p, args.vocab_size, self.mask_index)

    def next(self, current_seq_len, batch_size):
        all_input_ids, all_target_ids, all_sequence_lengths, all_real_mask_p = [], [], [], []
        for _ in range(batch_size):

            input_ids, target_ids, sequence_lengths, real_mask_p = self.__getitem__(self.current_idx)
            self.current_idx += 1
            if self.current_idx >= len(self.inputs):
                if self.shuffle:
                    self.iterations += 1
                    self.inputs, self.outputs, self.doc_ids = self.chunk()
                    print("Dataset reloaded")
                self.current_idx = 0

            all_input_ids.append(input_ids)
            all_target_ids.append(target_ids)
            all_sequence_lengths.append(sequence_lengths)
            all_real_mask_p.append(real_mask_p)

        input_ids = torch.stack(all_input_ids)
        target_ids = torch.stack(all_target_ids)
        sequence_lengths = torch.stack(all_sequence_lengths)
        real_mask_p = torch.stack(all_real_mask_p).mean()

        return input_ids, target_ids, sequence_lengths, real_mask_p

    def apply_mask(self, input_ids, target_ids, mask_ratios, replacement_ids):
        mask_p = self.args.mask_p
        mask_p = torch.topk(mask_ratios, max(1, int(mask_ratios.size(0) * mask_p + torch.rand(1).item())), largest=False).values.max().item()

        mask = mask_ratios <= mask_p
        target_mask = torch.cat([mask[1:], torch.ones(1, dtype=torch.bool)])
        target_ids = torch.where(target_mask, target_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)

        real_mask_p = mask.sum() / mask_ratios.numel()

        return input_ids, target_ids, real_mask_p

    def __getitem__(self, index):
        tokens = self.inputs[index].long()
        targets = self.outputs[index].long()

        mask_ratios, replacement_tokens = self.masking_strategy(tokens)
        input_ids, target_ids, real_mask_p = self.apply_mask(tokens, targets, mask_ratios, replacement_tokens)
        sequence_lengths = self.doc_ids[index]

        return input_ids, target_ids, sequence_lengths, real_mask_p


class DiffusionDatasetv2(Datasetv2):

    def __init__(self: DiffusionDatasetv2, dataset: str, tokenizer: Tokenizer, args: Namespace, seq_length: int, rank: int, shuffle: bool = True):
        super().__init__(dataset, tokenizer, args, seq_length, rank, args.seed + 23456, shuffle)
        self.masking_strategy = DiffusionMaskingStrategy(args.n_special_tokens, args.vocab_size, self.mask_index)

    def next(self, current_seq_len, batch_size):
        all_input_ids, all_target_ids, all_sequence_lengths, all_real_mask_p = [], [], [], []
        for _ in range(batch_size):

            input_ids, target_ids, sequence_lengths, real_mask_p = self.__getitem__(self.current_idx)
            self.current_idx += 1
            if self.current_idx >= len(self.inputs):
                if self.shuffle:
                    self.iterations += 1
                    self.inputs, self.outputs, self.doc_ids = self.chunk()
                    print("Dataset reloaded")
                self.current_idx = 0

            all_input_ids.append(input_ids)
            all_target_ids.append(target_ids)
            all_sequence_lengths.append(sequence_lengths)
            all_real_mask_p.append(real_mask_p)

        input_ids = torch.stack(all_input_ids)
        target_ids = torch.stack(all_target_ids)
        sequence_lengths = torch.stack(all_sequence_lengths)
        real_mask_p = torch.stack(all_real_mask_p)

        return input_ids, target_ids, sequence_lengths, real_mask_p

    def apply_mask(self, input_ids, target_ids, mask_ratios, replacement_ids):
        mask_p = torch.rand(1).item()
        mask_p = torch.topk(mask_ratios, max(1, int(mask_ratios.size(0) * mask_p + torch.rand(1).item())), largest=False).values.max().item()

        mask = mask_ratios <= mask_p
        target_mask = torch.cat([mask[1:], torch.ones(1, dtype=torch.bool)])
        target_ids = torch.where(target_mask, target_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)

        real_mask_p = mask.sum() / mask_ratios.numel()
        real_mask_p = torch.full_like(input_ids, real_mask_p, dtype=torch.float)
        real_mask_p = (1 - 1e-4) * real_mask_p + 1e-4

        return input_ids, target_ids, real_mask_p

    def __getitem__(self, index):
        tokens = self.inputs[index].long()
        targets = self.outputs[index].long()

        mask_ratios, replacement_tokens = self.masking_strategy(tokens)
        input_ids, target_ids, real_mask_p = self.apply_mask(tokens, targets, mask_ratios, replacement_tokens)
        sequence_lengths = self.doc_ids[index]

        return input_ids, target_ids, sequence_lengths, real_mask_p


class CausalDatasetv2(Datasetv2):
    def __init__(self: CausalDatasetv2, dataset: str, tokenizer: Tokenizer, args: Namespace, seq_length: int, rank: int, shuffle: bool = True):
        super().__init__(dataset, tokenizer, args, seq_length, rank, args.seed + 34567, shuffle)

    def next(self, current_seq_len, batch_size):
        all_input_ids, all_target_ids, all_sequence_lengths = [], [], []
        for _ in range(batch_size):

            input_ids, target_ids, sequence_lengths, _ = self.__getitem__(self.current_idx)
            self.current_idx += 1
            if self.current_idx >= len(self.inputs):
                self.iterations += 1

                if self.shuffle:
                    self.iterations += 1
                    self.inputs, self.outputs, self.doc_ids = self.chunk()
                self.current_idx = 0

            all_input_ids.append(input_ids)
            all_target_ids.append(target_ids)
            all_sequence_lengths.append(sequence_lengths)

        input_ids = torch.stack(all_input_ids)
        target_ids = torch.stack(all_target_ids)
        sequence_lengths = torch.stack(all_sequence_lengths)

        return input_ids, target_ids, sequence_lengths, torch.zeros([])

    def __getitem__(self, index):
        input_ids = self.inputs[index].long()
        target_ids = self.outputs[index].long()
        sequence_lengths = self.doc_ids[index]

        return input_ids, target_ids, sequence_lengths, torch.zeros([])


class SpanMaskingStrategy:
    def __init__(self, n_special_tokens, random_p, keep_p, vocab_size, mask_token_id):
        self.n_special_tokens = n_special_tokens
        self.random_p = random_p
        self.keep_p = keep_p
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.max_span_length = 3

    def __call__(self, tokens):
        length = tokens.size(0)

        span_lengths = torch.randint(1, self.max_span_length + 1, size=(length,), dtype=torch.int)
        cumsum = torch.cumsum(span_lengths, dim=0)

        total_length = cumsum[-1].item()
        indices = torch.zeros(total_length, dtype=torch.int)
        indices[cumsum - span_lengths] = torch.arange(length, dtype=torch.int)
        indices = torch.cummax(indices, dim=0)[0]
        indices = indices[:length]

        max_index = indices[-1].item()
        span_random_numbers_1, span_random_numbers_2 = torch.rand([(max_index + 1) * 2]).chunk(2)

        mask_ratios = span_random_numbers_1[indices]

        mask_ratios[tokens < self.n_special_tokens] = float('inf')

        replacement_p = span_random_numbers_2[indices]
        random_mask = replacement_p < self.random_p

        replacement_tokens = tokens.clone()
        replacement_tokens[random_mask] = torch.randint(
            low=self.n_special_tokens,
            high=self.vocab_size,
            size=[random_mask.sum().item()],
            dtype=torch.long
        )
        replacement_tokens[replacement_p > (self.random_p + self.keep_p)] = self.mask_token_id

        return mask_ratios, replacement_tokens


class DiffusionMaskingStrategy:
    def __init__(self, n_special_tokens, vocab_size, mask_token_id):
        self.n_special_tokens = n_special_tokens
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    def __call__(self, tokens):
        length = tokens.size(0)

        mask_ratios = torch.rand(length)
        mask_ratios[tokens < self.n_special_tokens] = float('inf')

        replacement_tokens = tokens.clone()
        replacement_tokens.fill_(self.mask_token_id)

        return mask_ratios, replacement_tokens


class ValidationDataset:
    def __init__(self, dataset: str, tokenizer, args, seq_length, rank):
        self.dataset = dataset
        self.max_seq_length = seq_length + 1
        self.n_special_tokens = args.n_special_tokens
        self.args = args
        self.global_step = 0

        self.mask_index = tokenizer.token_to_id("<mask>")
        self.cls_index = tokenizer.token_to_id("<s>")
        self.pad_index = tokenizer.token_to_id("<pad>")

        self.doc_segments = []
        self.orders = []
        self.lens = []
        self.seed = args.seed
        documents = torch.load(f"{dataset}/{rank:d}.bin", weights_only=False)
        for i, document in enumerate(documents):
            if i % args.document_skip != 0:
                continue

            document = torch.cat([torch.LongTensor([self.cls_index]), document])
            self.doc_segments += [
                document[offset : offset + self.max_seq_length]
                for offset in range(0, len(document), self.max_seq_length)
                if len(document) > 0 and len(document) - offset > 1
            ]
        self.len = len(self.doc_segments)
        self.current_idx = 0


class ValidationCausalDataset(ValidationDataset):

    def iterate_over_all(self, seq_len, batch_size):
        while self.current_idx < self.len:
            yield self.next(seq_len, batch_size)
        self.current_idx = 0  # Reset for next iteration

    def next(self, current_seq_len, batch_size):
        all_input_ids, all_target_ids, all_sequence_lengths = [], [], []
        for _ in range(batch_size):
            input_ids, target_ids, sequence_lengths, _ = self._getitem()
            self.current_idx += 1

            all_input_ids.append(input_ids)
            all_target_ids.append(target_ids)
            all_sequence_lengths.append(sequence_lengths)

        input_ids = torch.stack(all_input_ids)
        target_ids = torch.stack(all_target_ids)
        sequence_lengths = torch.stack(all_sequence_lengths)

        return input_ids, target_ids, sequence_lengths, torch.zeros([])

    def _getitem(self):
        tokens = self.doc_segments[self.current_idx]
        seq_length = min(self.max_seq_length, tokens.size(0))

        input_ids = tokens[:seq_length].long()
        target_ids = tokens[:seq_length].long()

        document_index = 0
        sequence_lengths = torch.full((seq_length,), document_index, dtype=torch.int)

        while self.max_seq_length - input_ids.size(0) > 1:
            assert self.current_idx < self.len, "Looping around, make your validation dataset bigger."
            self.current_idx += 1
            tokens = self.doc_segments[self.current_idx].long()
            seq_length = min(self.max_seq_length - input_ids.size(0), tokens.size(0))

            # select random offset
            offset = 0
            if seq_length < tokens.size(0):
                offset = torch.randint(0, tokens.size(0) - seq_length, size=(1,)).item()

            tokens = tokens[offset:offset + seq_length]

            input_ids = torch.cat([
                input_ids,
                tokens
            ])
            target_ids = torch.cat([
                target_ids,
                tokens
            ])
            document_index += 1
            sequence_lengths = torch.cat([
                sequence_lengths,
                torch.full((seq_length,), document_index, dtype=torch.int)
            ])

        padding_length = self.max_seq_length - input_ids.size(0)
        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.pad_index] * padding_length)
            ])
            target_ids = torch.cat([
                target_ids,
                torch.LongTensor([-100] * padding_length)
            ])
            document_index += 1
            sequence_lengths = torch.cat([
                sequence_lengths,
                torch.full((padding_length,), document_index, dtype=torch.int)
            ])

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        sequence_lengths = sequence_lengths[:-1]

        return input_ids, target_ids, sequence_lengths, torch.zeros([])


class ValidationMaskedDataset(ValidationDataset):

    def __init__(self, dataset: str, tokenizer, args, seq_length, rank):
        super().__init__(dataset, tokenizer, args, seq_length, rank)

        self.masking_strategy = SpanMaskingStrategy(args.n_special_tokens, args.mask_random_p, args.mask_keep_p, args.vocab_size, self.mask_index)

    def iterate_over_all(self, seq_len, batch_size):
        while self.current_idx < self.len:
            yield self.next(seq_len, batch_size)
        self.current_idx = 0  # Reset for next iteration

    def next(self, current_seq_len, batch_size):
        all_input_ids, all_target_ids, all_sequence_lengths, all_real_mask_p = [], [], [], []
        for _ in range(batch_size):
            input_ids, target_ids, sequence_lengths, real_mask_p = self._getitem()
            self.current_idx += 1

            all_input_ids.append(input_ids)
            all_target_ids.append(target_ids)
            all_sequence_lengths.append(sequence_lengths)
            all_real_mask_p.append(real_mask_p)

        input_ids = torch.stack(all_input_ids)
        target_ids = torch.stack(all_target_ids)
        sequence_lengths = torch.stack(all_sequence_lengths)
        real_mask_p = torch.stack(all_real_mask_p).mean()

        return input_ids, target_ids, sequence_lengths, real_mask_p

    def apply_mask(self, input_ids, mask_ratios, replacement_ids):
        mask_p = self.args.mask_p_min
        mask_p = torch.topk(mask_ratios, max(1, int(mask_ratios.size(0) * mask_p + torch.rand(1).item())), largest=False).values.max().item()

        mask = mask_ratios <= mask_p
        target_ids = torch.where(mask, input_ids, -100)
        input_ids = torch.where(mask, replacement_ids, input_ids)

        real_mask_p = mask.sum() / mask_ratios.numel()

        return input_ids, target_ids, real_mask_p

    def _getitem(self):
        tokens = self.doc_segments[self.current_idx]
        seq_length = min(self.max_seq_length, tokens.size(0))
        tokens = tokens[:seq_length].long()

        mask_ratios, replacement_tokens = self.masking_strategy(tokens)
        input_ids, target_ids, real_mask_p = self.apply_mask(tokens, mask_ratios, replacement_tokens)

        document_index = 0
        sequence_lengths = torch.full((seq_length,), document_index, dtype=torch.int)

        while self.max_seq_length - input_ids.size(0) > 1:
            assert self.current_idx < self.len, "Looping around, make your validation dataset bigger."
            self.current_idx += 1
            tokens = self.doc_segments[self.current_idx].long()
            seq_length = min(self.max_seq_length - input_ids.size(0), tokens.size(0))

            # select random offset
            offset = 0
            if seq_length < tokens.size(0):
                offset = torch.randint(0, tokens.size(0) - seq_length, size=(1,)).item()

            tokens = tokens[offset:offset + seq_length]

            mask_ratios, replacement_tokens = self.masking_strategy(tokens)
            input_ids_, target_ids_, _ = self.apply_mask(tokens, mask_ratios, replacement_tokens)

            input_ids = torch.cat([
                input_ids,
                input_ids_,
            ])
            target_ids = torch.cat([
                target_ids,
                target_ids_
            ])

            document_index += 1
            sequence_lengths = torch.cat([
                sequence_lengths,
                torch.full((seq_length,), document_index, dtype=torch.int)
            ])

        padding_length = self.max_seq_length - input_ids.size(0)
        if padding_length > 0:
            input_ids = torch.cat([
                input_ids,
                torch.LongTensor([self.pad_index] * padding_length)
            ])
            target_ids = torch.cat([
                target_ids,
                torch.LongTensor([-100] * padding_length)
            ])
            document_index += 1
            sequence_lengths = torch.cat([
                sequence_lengths,
                torch.full((padding_length,), document_index, dtype=torch.int)
            ])

        input_ids = input_ids[:-1]
        target_ids = target_ids[1:]
        sequence_lengths = sequence_lengths[:-1]

        return input_ids, target_ids, sequence_lengths, real_mask_p
