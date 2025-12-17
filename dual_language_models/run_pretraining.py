import os
import sys
import os.path
import argparse
from tqdm import tqdm
from socket import gethostname
import json
import math
from pathlib import Path
from contextlib import nullcontext
import datetime

from tokenizers import Tokenizer
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch._dynamo

from dual_language_models.model.model import Model
from dual_language_models.optimizers.kimi_muon import Muon
from dual_language_models.utils import trapezoid_schedule, MaskScheduler, is_main_process, seed_everything
from dual_language_models.pretraining.dataset import ValidationCausalDataset, ValidationMaskedDataset, DiffusionDatasetv2, CausalDatasetv2
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.suppress_errors = True


if int(os.environ["SLURM_PROCID"]) == 0:
    import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", default="data/hplt_2_32b_token_shards", required=False, type=Path, help="Train dataset name.")
    parser.add_argument("--valid_path", default="data/hplt_2_32b_valid_token_shards", type=Path, help="Path to the validation dataset.")
    parser.add_argument("--name", default="Diffusion_Run_3:4_64x", type=str, help="Name of the run.")
    parser.add_argument("--wandb_project", default="hybrid_language_modelling", type=str, help="Name of the WandB project to log into.")
    parser.add_argument("--wandb_entity", default="ltg", type=str, help="The entity to log to on WandB (typically your wandb username).")
    parser.add_argument("--config_file", default="configs/large.json", type=Path, help="The BERT model config")
    parser.add_argument("--tokenizer_path", default="tokenizers/tokenizer.json", type=Path, help="Path to the tokenizer.")
    parser.add_argument("--output_dir", default="model_checkpoints", type=Path, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--checkpoint_foldername", default=None, type=Path, help="The checkpoint filename to resume training.")
    parser.add_argument("--hybrid_numerator", default=3, type=int, help="The numerator of the hybrid ratio.")
    parser.add_argument("--hybrid_denominator", default=4, type=int, help="The denominator of the hybrid ratio (the number of GPUs should be divisible by this number).")
    parser.add_argument("--max_seq_length", default=2048, type=int, help="Sequence length for training.")
    parser.add_argument("--local_batch_size", default=8, type=int, help="Batch size for training per GPU.")
    parser.add_argument("--global_batch_size", default=2048, type=int, help="Total batch size for training per GPUs and per grad accumulation step.")
    parser.add_argument("--learning_rate", default=7e-3, type=float, help="The initial learning rate for AdamW.")
    parser.add_argument("--number_of_tokens", default=6e11 * 4, type=int, help="Total number of tokens to train on.")
    parser.add_argument("--max_steps", default=2048 * 4, type=int)
    parser.add_argument("--document_skip", default=1, type=int)
    parser.add_argument("--validate_every", default=128, type=int, help="Run validation after every X training steps.")
    parser.add_argument("--validation_steps", default=1, type=int, help="Number of validation steps.")
    parser.add_argument("--scheduler", default="trapezoid", type=str, help="Which learning rate scheduler to use.", choices=["trapezoid", "cosine", "flat"])
    parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--cooldown_proportion", default=0.25, type=float, help="Proportion of training to perform linear learning rate cooldown for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--save_every', type=int, default=100, help="save every X steps")
    parser.add_argument("--checkpoint_style", default="exp", type=str, help="The style of checkpointing", choices=["linear", "exp"])
    parser.add_argument("--first_checkpoint", default=64.0, type=float, help="Represents the number of tokens/steps at which to save the first checkpoint.")
    parser.add_argument('--checkpoint_every', type=int, default=1e8, help="create a model chekpoint every X tokens/steps after the initial checkpoint.")
    parser.add_argument("--checkpoint_mult", default=math.sqrt(2), type=float, help="Checkpoint every power of X steps/tokens (times a initial checkpoint).")
    parser.add_argument("--checkpoint_on", default="steps", type=str, help="What to checkpoint on.")
    parser.add_argument("--checkpoint_init", default=True, action="store_true", help="Whether to save the initial untrained model.")
    parser.add_argument("--checkpoint_before_cooldown", default=True, action="store_true", help="Whether to checkpoint the model before starting cooldown.")
    parser.add_argument("--mask_p_max", default=0.3, type=float, help="Masking asking probability.")
    parser.add_argument("--mask_p_min", default=0.1, type=float, help="Minimum masking probability.")
    parser.add_argument("--mask_random_p", default=0.1, type=float, help="Probability of replacing the masked token with a random token.")
    parser.add_argument("--mask_keep_p", default=0.1, type=float, help="Probability of keeping the masked token.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--optimizer_eps", default=1e-8, type=float, help="Optimizer epsilon.")
    parser.add_argument("--optimizer_beta1", default=0.9, type=float, help="Optimizer beta1.")
    parser.add_argument("--optimizer_beta2", default=0.98, type=float, help="Optimizer beta2.")
    parser.add_argument("--max_gradient", default=1e9, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--n_special_tokens', default=16, type=int, help="Number of special tokens.")
    parser.add_argument('--z_loss_weight', default=0.0001, type=float, help="Weight for the z loss.")
    parser.add_argument("--experiment", default="dataset_size", type=str)
    parser.add_argument("--optimizer", default="muon", type=str, choices=["muon", "adamw"])
    parser.add_argument("--untie", default=True, action="store_true")
    parser.add_argument("--momentum", default=0.95, type=float)
    parser.add_argument("--n_repetitions", default=64, type=int, help="Number of times to repeat the dataset.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_path = (args.output_dir / args.name)
    args.output_path.mkdir(parents=True, exist_ok=True)
    args.steps_in_epoch = int(args.max_steps / args.n_repetitions)

    args.tokens_per_step = args.global_batch_size * args.max_seq_length

    return args


def setup_training(args, tokenizer):
    assert torch.cuda.is_available()
    args.n_gpu = torch.cuda.device_count()
    args.tokens_per_batch = args.global_batch_size * args.max_seq_length
    if args.max_steps is None:
        args.max_steps = (args.number_of_tokens // args.tokens_per_batch) + 1
    else:
        args.number_of_tokens = args.max_steps * args.tokens_per_batch
    if args.checkpoint_on == "steps":
        args.next_checkpoint = args.first_checkpoint
    elif args.checkpoint_on == "tokens":
        args.next_checkpoint = math.ceil(args.first_checkpoint / args.tokens_per_batch)
    if args.checkpoint_before_cooldown:
        args.cooldown_checkpoint = args.max_steps - int(args.cooldown_proportion * args.max_steps)
    else:
        args.cooldown_checkpoint = -1

    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["SLURM_PROCID"])
    args.gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert args.gpus_per_node == torch.cuda.device_count()  # Might create errors on ROCm
    print(f"Hello from rank {args.rank} of {args.world_size} on {gethostname()} where there are {args.gpus_per_node} allocated GPUs per node.", flush=True)

    args.accumulate_steps = max(1, (args.global_batch_size // args.world_size) // args.local_batch_size)

    assert args.world_size % args.hybrid_denominator == 0
    if (args.rank % args.hybrid_denominator) < args.hybrid_numerator:
        args.dataset_type = "masked"
    else:
        args.dataset_type = "causal"
    print(f"Dataset type: {args.dataset_type}", flush=True)

    args.local_rank = args.rank % args.gpus_per_node

    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        rank=args.rank,
        world_size=args.world_size,
        timeout=datetime.timedelta(minutes=10)
    )

    seed_everything(args.seed + args.rank)

    args.shard_rank = args.rank
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    print(f"RCCL started on device {args.device}", flush=True)
    print(f"host: {gethostname()}, rank: {args.rank}, local_rank: {args.local_rank}")

    args.vocab_size = tokenizer.get_vocab_size()

    if is_main_process():
        wandb.init(
            name=args.name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.experiment
        )
        wandb.config.update(args)
        wandb.save(sys.argv[0], policy="now")


def load_config(args):
    with args.config_file.open("r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args


def prepare_model_and_optimizer(args):
    print(f"Starting to load the config from {args.config_file}", flush=True)
    args = load_config(args)
    print("Starting to load the model", flush=True)
    model = Model(args)
    print("Model loaded", flush=True)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(args)
        wandb.config.update({"n_params": n_params})
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    for p in model.parameters():
        if not p.data.is_contiguous():
            print("Warning: non-contiguous parameter data", flush=True)
            p.data = p.data.contiguous()

    model.cuda(args.device)
    model.create_mask(args.device)

    # model = torch.compile(model)

    ddp_model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=False,
    )
    print(f"Model initialized on device {args.device}", flush=True)

    matrix_params = [(n, p) for n, p in model.encoder.named_parameters() if p.ndim == 2]
    if hasattr(model.classifier, "projection"):
        matrix_params.append(("classifier.projection.weight", model.classifier.projection.weight))
    other_params = [(n, p) for n, p in model.named_parameters() if p.ndim != 2]
    other_params.append(("embedding.word_embedding.weight", model.embedding.word_embedding.weight))
    if args.untie:
        other_params.append(("classifier.emb2vocab.weight", model.classifier.emb2vocab.weight))

    muon_parameters = [p for _, p in matrix_params]
    adamw_parameters = [p for _, p in other_params]

    if is_main_process():
        print(f"Parameters with {args.optimizer} Optimizer:")
        for n, _ in matrix_params:
            print(n)
        print("\nParameters with AdamW Optimizer:")
        for n, _ in other_params:
            print(n)
        print(flush=True)

    optimizer = Muon(
        muon_params=muon_parameters,
        lr=args.learning_rate,
        wd=args.weight_decay,
        momentum=args.momentum,
        nesterov=True,
        ns_steps=5,
        adamw_params=adamw_parameters
    )

    lr_scheduler = trapezoid_schedule(
        optimizer,
        int(args.max_steps * args.warmup_proportion),
        int(args.max_steps * args.cooldown_proportion),
        args.max_steps
    )
    mask_scheduler = MaskScheduler(
        args.mask_p_min,
        args.mask_p_max,
        int(args.max_steps * args.warmup_proportion),
        int(args.max_steps * args.cooldown_proportion),
        args.max_steps
    )
    args.mask_p = args.mask_p_max

    global_step = 0
    if args.checkpoint_foldername is not None:
        path_to_checkpoint = args.checkpoint_foldername / "state_dict.bin"
        state_dict = torch.load(path_to_checkpoint, map_location=args.device)
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        lr_scheduler.load_state_dict(state_dict["lr_schedulers"])
        mask_scheduler.load_state_dict(state_dict["mask_scheduler"])
        global_step = state_dict["global_step"]

    return model, ddp_model, optimizer, lr_scheduler, mask_scheduler, global_step


@torch.no_grad()
def get_batch(args, dataset, global_step):
    batch = dataset.next(args.max_seq_length, args.local_batch_size)
    input_ids, target_ids, doc_ids, mask_p = [t.cuda(non_blocking=True) for t in batch]
    input_ids, target_ids, mask_p = input_ids.t(), target_ids.t(), mask_p.t()

    return input_ids, target_ids, doc_ids, mask_p


def _do_checkpoint(global_step, args):
    if global_step >= args.next_checkpoint:
        if args.checkpoint_style == "exp":
            args.next_checkpoint *= args.checkpoint_mult
        elif args.checkpoint_style == "linear":
            args.next_checkpoint += args.checkpoint_every
        return True
    elif global_step == args.cooldown_checkpoint:
        return True
    return False


def training_loop(model, ddp_model, train_diffusion_dataset, train_causal_dataset, valid_diffusion_dataset, valid_causal_dataset, optimizer, lr_scheduler, mask_scheduler, global_step, args):
    if args.checkpoint_init and global_step == 0:
        save_checkpoint(model, optimizer, lr_scheduler, mask_scheduler, global_step, 0, train_diffusion_dataset, args)

    model = model.train()
    model.zero_grad(set_to_none=True)

    train_dataset = train_diffusion_dataset if args.dataset_type == "masked" else train_causal_dataset

    # initialize the dataloader and the metrics
    total_loss, total_accuracy, total_z_loss, total_mask_p, total_grad_norm = 0.0, 0.0, 0.0, 0.0, 0.0
    tokens_trained = global_step * args.tokens_per_step

    # calculate the number of forward passes to perform
    num_steps = int(args.max_steps * args.accumulate_steps)

    # Initialize the progress bar
    progress_bar = tqdm(total=args.max_steps, initial=global_step, disable=not is_main_process(), desc="Train iteration")

    # iterate over the steps
    for local_step in range(num_steps):
        epoch = global_step // args.steps_in_epoch
        if (epoch + args.rank) % args.hybrid_denominator < args.hybrid_numerator:
            dataset_type = "masked"
        else:
            dataset_type = "causal"
        if dataset_type != args.dataset_type:
            args.dataset_type = dataset_type
            train_dataset = train_diffusion_dataset if dataset_type == "masked" else train_causal_dataset
            model.change_model_type(dataset_type, args.device)

        next_batch = get_batch(args, train_dataset, global_step)

        input_ids, target_ids, doc_ids, mask_p = next_batch

        # forward pass, do a more detailed check of the model every 100 steps
        # with ModelLogger(enable=global_step % 100 == 0, module=model):
        with ddp_model.no_sync() if (local_step + 1) % args.accumulate_steps != 0 else nullcontext():

            output = ddp_model(input_ids, doc_ids, target_ids)

            loss, accuracy, z_loss, _ = output.loss, output.accuracy, output.z_loss, output.num_tokens

            if args.dataset_type == "masked":
                weight = 1.0 / mask_p[target_ids != -100]
            else:
                weight = 2.0
            
            loss = (loss / input_ids.numel() * weight).sum() / args.accumulate_steps
            z_loss = (z_loss / input_ids.numel() * weight).sum() / args.accumulate_steps

            # backward pass through both losses
            (loss + args.z_loss_weight * z_loss).backward()

        # add the tracked metrics (for gradient accumulation)
        with torch.no_grad():
            total_loss += loss.detach()
            total_accuracy += accuracy / args.accumulate_steps
            total_z_loss += z_loss.detach()
            total_mask_p += mask_p / args.accumulate_steps

        # gradient accumulation -- if we have accumulated enough gradients, we can perform the optimizer step; otherwise, we just continue and backpropagate through the next batch
        if (local_step + 1) % args.accumulate_steps != 0:
            continue

        tokens_trained += args.tokens_per_step

        # clip the gradients
        total_grad_norm += nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient) / args.accumulate_steps

        # optimizer step
        optimizer.step()
        lr_scheduler.step()
        args.mask_p = mask_scheduler.step()

        with torch.no_grad():
            # be careful here, not all GPUs work with the same training objective
            if args.dataset_type == "masked":
                total_mlm_loss = total_loss / (args.hybrid_numerator / args.hybrid_denominator)
                total_mlm_accuracy = total_accuracy / (args.hybrid_numerator / args.hybrid_denominator)
                total_clm_loss = torch.zeros_like(total_mlm_loss)
                total_clm_accuracy = torch.zeros_like(total_mlm_accuracy)
                total_mask_p = total_mask_p / (args.hybrid_numerator / args.hybrid_denominator)
            else:
                total_clm_loss = total_loss / (1 - args.hybrid_numerator / args.hybrid_denominator)
                total_clm_accuracy = total_accuracy / (1 - args.hybrid_numerator / args.hybrid_denominator)
                total_mlm_loss = torch.zeros_like(total_clm_loss)
                total_mlm_accuracy = torch.zeros_like(total_clm_accuracy)
                total_mask_p = torch.zeros_like(total_mask_p)

            # accumulate the metrics across GPUs
            metrics = torch.stack([total_loss, total_accuracy, total_z_loss, total_mask_p.mean(), total_mlm_loss, total_mlm_accuracy, total_clm_loss, total_clm_accuracy])
            dist.all_reduce(metrics, dist.ReduceOp.AVG)
            total_loss, total_accuracy, total_z_loss, total_mask_p, total_mlm_loss, total_mlm_accuracy, total_clm_loss, total_clm_accuracy = metrics.tolist()

            # log the metrics
            if is_main_process():
                wandb.log(
                    {
                        "train/loss": total_loss,
                        "train/z_loss": total_z_loss,
                        "train/perplexity": math.exp(total_loss),
                        "train/accuracy": total_accuracy * 100.0,
                        "train/mlm_loss": total_mlm_loss,
                        "train/mlm_accuracy": total_mlm_accuracy * 100.0,
                        "train/clm_loss": total_clm_loss,
                        "train/clm_accuracy": total_clm_accuracy * 100.0,
                        "stats/learning_rate_adamw": optimizer.param_groups[0]['lr'],
                        "stats/learning_rate_muon": optimizer.param_groups[0]['lr'],
                        "stats/grad_norm": total_grad_norm,
                        "stats/global_batch_size": args.global_batch_size * args.max_seq_length,
                        "stats/local_batch_size": args.local_batch_size * args.max_seq_length,  # Is this correct?
                        "stats/accumulate_steps": args.accumulate_steps,
                        "stats/mask_p": total_mask_p,
                        "global_step": global_step,
                        "tokens_trained": tokens_trained,
                    },
                    step=global_step
                )

        # zero the accumulated gradients and the metrics
        model.zero_grad(set_to_none=True)
        total_loss, total_accuracy, total_z_loss, total_mask_p, total_grad_norm = 0.0, 0.0, 0.0, 0.0, 0.0

        global_step += 1
        progress_bar.update()

        # Run validation
        if global_step % args.validate_every == 0:
            model = model.eval()
            with torch.no_grad():
                validate(model, ddp_model, valid_diffusion_dataset, valid_causal_dataset, global_step, args)
            model = model.train()

        # save a backup of the model and the full training state
        if global_step % args.save_every == 0:
            save(model, optimizer, lr_scheduler, mask_scheduler, global_step, train_dataset, args)

        # save a checkpoint of the model and full training state
        if _do_checkpoint(global_step, args):
            save_checkpoint(model, optimizer, lr_scheduler, mask_scheduler, global_step, tokens_trained, train_dataset, args)

        # Exiting the training due to hitting max steps
        if global_step >= args.max_steps:
            progress_bar.close()
            return

    progress_bar.close()


def validate(model, ddp_model, valid_diffusion_dataset, valid_causal_dataset, global_step, args):

    total_loss, total_accuracy, total_z_loss, total_mask_p = 0.0, 0.0, 0.0, 0.0
    local_step = 0
    valid_steps = 0

    valid_dataset = valid_diffusion_dataset if args.dataset_type == "masked" else valid_causal_dataset

    for batch in valid_dataset.iterate_over_all(args.max_seq_length, args.local_batch_size):
        input_ids, target_ids, doc_ids, mask_p = [t.cuda(non_blocking=True) for t in batch]
        input_ids, target_ids = input_ids.t(), target_ids.t()
        mask_p = mask_p.mean()

        with ddp_model.no_sync() if (local_step + 1) % args.accumulate_steps != 0 else nullcontext():

            output = ddp_model(input_ids, doc_ids, target_ids)
            local_step += 1

            loss, accuracy, z_loss, _ = output.loss, output.accuracy, output.z_loss, output.num_tokens
            loss, z_loss = loss.mean(), z_loss.mean()

            weight = 1.0 / args.accumulate_steps
            if mask_p != 0:
                weight = weight * (mask_p / args.mask_p_max)

        # add the tracked metrics (for gradient accumulation)
        total_loss += loss.detach() / args.accumulate_steps
        total_accuracy += accuracy / args.accumulate_steps
        total_z_loss += z_loss.detach() / args.accumulate_steps
        total_mask_p += mask_p / args.accumulate_steps

        # gradient accumulation -- if we have accumulated enough gradients, we can perform the optimizer step; otherwise, we just continue and backpropagate through the next batch
        if (local_step + 1) % args.accumulate_steps != 0:
            continue

        # be careful here, not all GPUs work with the same training objective
        if args.dataset_type == "masked":
            total_mlm_loss = total_loss / (args.hybrid_numerator / args.hybrid_denominator)
            total_mlm_accuracy = total_accuracy / (args.hybrid_numerator / args.hybrid_denominator)
            total_clm_loss = torch.zeros_like(total_mlm_loss)
            total_clm_accuracy = torch.zeros_like(total_mlm_accuracy)
            total_mask_p = total_mask_p / (args.hybrid_numerator / args.hybrid_denominator)
        else:
            total_clm_loss = total_loss / (1 - args.hybrid_numerator / args.hybrid_denominator)
            total_clm_accuracy = total_accuracy / (1 - args.hybrid_numerator / args.hybrid_denominator)
            total_mlm_loss = torch.zeros_like(total_clm_loss)
            total_mlm_accuracy = torch.zeros_like(total_clm_accuracy)
            total_mask_p = torch.zeros_like(total_mask_p)

        # accumulate the metrics across GPUs
        metrics = torch.stack([total_loss, total_accuracy, total_z_loss, total_mask_p, total_mlm_loss, total_mlm_accuracy, total_clm_loss, total_clm_accuracy])
        dist.all_reduce(metrics, dist.ReduceOp.AVG)
        total_loss, total_accuracy, total_z_loss, total_mask_p, total_mlm_loss, total_mlm_accuracy, total_clm_loss, total_clm_accuracy = metrics.tolist()

        # log the metrics
        if is_main_process():
            wandb.log(
                {
                    "valid/loss": total_loss,
                    "valid/z_loss": total_z_loss,
                    "valid/perplexity": math.exp(total_loss),
                    "valid/accuracy": total_accuracy * 100.0,
                    "valid/mlm_loss": total_mlm_loss,
                    "valid/mlm_accuracy": total_mlm_accuracy * 100.0,
                    "valid/clm_loss": total_clm_loss,
                    "valid/clm_accuracy": total_clm_accuracy * 100.0,
                },
                step=global_step
            )

        # zero the metrics
        total_loss, total_accuracy, total_z_loss, total_mask_p = 0.0, 0.0, 0.0, 0.0

        valid_steps += 1

        if valid_steps == args.validation_steps:
            break


def save(model, optimizer, lr_scheduler, mask_scheduler, global_step, train_dataset, args):
    path_to_save_folder = args.output_path / "final"
    path_to_save_folder.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        torch.save(
            model.state_dict(),
            path_to_save_folder / "state_dict.bin"
        )
        # torch.save(
        #     train_dataset.get_state(),
        #     path_to_save_folder / f"dataset_info_{args.dataset_type}_{args.shard_rank}.bin"
        # )


def save_checkpoint(model, optimizer, lr_scheduler, mask_scheduler, global_step, tokens_trained, train_dataset, args):
    if global_step != args.cooldown_checkpoint:
        path_to_save_folder = args.output_path / f"checkpoint_{tokens_trained / 1e9:.2f}B"
    else:
        path_to_save_folder = args.output_path / f"checkpoint_pre_cooldown_{tokens_trained / 1e9:.2f}B"
    path_to_save_folder.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        torch.save(
            model.state_dict(),
            path_to_save_folder / "state_dict.bin"
        )
        # torch.save(
        #     train_dataset.get_state(),
        #     path_to_save_folder / f"dataset_info_{args.dataset_type}_{args.shard_rank}.bin"
        # )


def load_train_dataset(args, tokenizer):
    valid_diffusion_dataset = ValidationMaskedDataset(args.valid_path, tokenizer, args, args.max_seq_length, args.shard_rank)
    valid_causal_dataset = ValidationCausalDataset(args.valid_path, tokenizer, args, args.max_seq_length, args.shard_rank)

    train_diffusion_dataset = DiffusionDatasetv2(args.train_path, tokenizer, args, args.max_seq_length, args.shard_rank, shuffle=True)
    train_causal_dataset = CausalDatasetv2(args.train_path, tokenizer, args, args.max_seq_length, args.shard_rank, shuffle=True)

    return train_diffusion_dataset, train_causal_dataset, valid_diffusion_dataset, valid_causal_dataset


if __name__ == "__main__":
    args = parse_arguments()

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    print(f"Tokenizer loaded from {args.tokenizer_path}", flush=True)

    setup_training(args, tokenizer)
    model, ddp_model, optimizer, lr_scheduler, mask_scheduler, global_step = prepare_model_and_optimizer(args)
    train_diffusion_dataset, train_causal_dataset, valid_diffusion_dataset, valid_causal_dataset = load_train_dataset(args, tokenizer)

    training_loop(model, ddp_model, train_diffusion_dataset, train_causal_dataset, valid_diffusion_dataset, valid_causal_dataset, optimizer, lr_scheduler, mask_scheduler, global_step, args)

    save(model, optimizer, lr_scheduler, mask_scheduler, args.max_steps, train_diffusion_dataset, args)
