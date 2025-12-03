import torch
from gptopt.train_distributed import train
from gptopt.optim.utils import get_scheduler, get_optimizer_factory
from gptopt.utils import hash_config, set_seed, get_worker_info
from gptopt.model import load_model
from gptopt.dataloader import DATA_DIR, ShardedDataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import copy 
import json
import os
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import uuid


@hydra.main(version_base=None, config_path="hydra_conf", config_name="config")
def main(config : DictConfig):
    config = OmegaConf.to_container(config, resolve=True)
    # Use Hydra-configured seed (overridable by sweeps)
    set_seed(config.get("seed", 42))

    # First set up DDP
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        dist.init_process_group(backend='nccl')
    world_size, rank, local_rank, device = get_worker_info()
    master_process = (rank == 0) # this process will do logging, checkpointing etc.
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    print(f"Using device: {device}")

    # Logging
    output_dir = HydraConfig.get().runtime.output_dir
    ckpt_dir = f"{output_dir}/checkpoints/" if config['logging_params'].get('save_checkpoint', False) else ""
    if master_process:
        print(f"Loading configuration from {HydraConfig.get().job.config_name}")
        print(f"Training on dataset {config['training_data']['dataset']['name']}")
        os.makedirs(output_dir, exist_ok=True)  
        if ckpt_dir != "": os.makedirs(ckpt_dir, exist_ok=True)

    # Load model
    model = load_model(config['gpt_model'], device)
    # Set the training parameters
    training_params = config['training_data']['training_params']
    opt_config = config["optimizer_params"]
    torch.set_float32_matmul_precision(training_params['tensorcore_precision'])

    # Load data
    dataset_path = os.path.join(DATA_DIR, f"{config['training_data']['dataset']['name']}-gpt2")
    if master_process: print(f"Load data from {dataset_path}")
    B, T = training_params['batch_size'], training_params['context_length']
    assert training_params['tokens_processed'] % (world_size * B * T) == 0 
    train_dataloader = ShardedDataLoader(dataset_path, B, T, "train", device)
    val_dataloader = ShardedDataLoader(dataset_path, B, T, "val", device)
    total_iterations = int(training_params['num_epochs'] * len(train_dataloader) / training_params['tokens_processed'])
    if master_process:
        print(f"Length of train dataset : {len(train_dataloader)/1e6:0.1f} million tokens")
        print(f"Length of validation dataset : {len(val_dataloader)/1e6:0.1f} million tokens")
        print(f"Total number of iterations : {total_iterations}")

    print()
    if master_process:
        print(f"Training with optimizer {opt_config['name']} and learning rate {opt_config['args']['lr']}")

    # Generate hash for the current optimizer configuration
    # config_hash = hash_config(opt_config, training_params, config['gpt_model'])
    random_job_id = str(uuid.uuid4()).split('-')[0]
    file_name = f"logs_jobid_{random_job_id}.json"
    output_path = os.path.join(output_dir, file_name)

    # copy model to ensure consistency
    model_copy = copy.deepcopy(model).to(device)
    if training_params['compile']:
        if master_process: print("Compiling model")
        model_copy = torch.compile(model_copy)

    if ddp:
        model_copy = DDP(model_copy, device_ids=[local_rank])

    # Setup optimizer
    opt_config_args = opt_config['args']

    # Optional helper: map named Muon variants (and related baselines) to concrete settings
    if opt_config['name'] == 'muon':
        muon_variant = opt_config_args.get('muon_variant', None)
        if muon_variant is not None:
            if muon_variant == 'pe_all':
                opt_config_args['polar_method'] = 'polarexpress'
                opt_config_args['muon_mode'] = 'stacked_qkv'
                if opt_config_args.get('polar_num_iters', None) is None:
                    opt_config_args['polar_num_iters'] = 5
                if opt_config_args.get('polar_cushion', None) is None:
                    opt_config_args['polar_cushion'] = 0.024
            elif muon_variant == 'ns_all':
                opt_config_args['polar_method'] = 'Keller'
                opt_config_args['muon_mode'] = 'stacked_qkv'
            elif muon_variant == 'pe_mod_all':
                opt_config_args['polar_method'] = 'polarexpress'
                opt_config_args['muon_mode'] = 'stacked_qkv'
                opt_config_args['polar_num_iters'] = 3
                opt_config_args['polar_cushion'] = 0.04
            elif muon_variant == 'pe_voffn':
                opt_config_args['polar_method'] = 'polarexpress'
                opt_config_args['muon_mode'] = 'voh_only'
                # Use a smaller LR for voffn variants
                opt_config_args['lr'] = 0.003
            elif muon_variant == 'ns_voffn':
                opt_config_args['polar_method'] = 'Keller'
                opt_config_args['muon_mode'] = 'voh_only'
                opt_config_args['lr'] = 0.003
            elif muon_variant == 'adamw_repro':
                # AdamW baseline for reproduce-paper sweeps
                opt_config['name'] = 'adamw'
                opt_config_args['lr'] = 0.0003
                opt_config_args['weight_decay'] = 0.0
                opt_config_args['betas'] = (0.9, 0.999)
                # Remove Muon-only arguments that AdamW does not accept
                for k in [
                    'adamw_betas',
                    'muon_mode',
                    'split_heads',
                    'polar_method',
                    'polar_num_iters',
                    'polar_safety',
                    'polar_cushion',
                    'ns_steps',
                ]:
                    opt_config_args.pop(k, None)
            elif muon_variant == 'adamw_timing':
                # AdamW baseline for timing sweeps
                opt_config['name'] = 'adamw'
                opt_config_args['lr'] = 0.0003
                opt_config_args['weight_decay'] = 0.01
                opt_config_args['betas'] = (0.9, 0.999)
                for k in [
                    'adamw_betas',
                    'muon_mode',
                    'split_heads',
                    'polar_method',
                    'polar_num_iters',
                    'polar_safety',
                    'polar_cushion',
                    'ns_steps',
                ]:
                    opt_config_args.pop(k, None)
            else:
                raise ValueError(f"Unknown muon_variant: {muon_variant}")
            # Do not pass helper field into the optimizer constructor
            opt_config_args['muon_variant'] = None

    # Filter out None/null values to avoid passing unused parameters to optimizer
    opt_config_args = {k: v for k, v in opt_config_args.items() if v is not None}

    # Instantiate optimizer with possibly-updated name (e.g., adamw baselines)
    optimizer_obj = get_optimizer_factory(opt_config['name'])
    if opt_config['name'] in ['muon']:
        opt_config_args['nheads'] = config['gpt_model'].get('n_head', None)
    optimizer = optimizer_obj(model_copy.named_parameters(), **opt_config_args)
    scheduler = get_scheduler(config['lr_schedule'], optimizer, total_iterations=total_iterations)

    # Initialize wandb
    if master_process and config['logging_params'].get('wandb', None) is not None:
        config_for_wandb_logging = dict(**config, world_size=world_size)
        wandb_config = config['logging_params']['wandb'].copy()  # Make a copy to avoid modifying original
        if "dir" not in wandb_config:
            wandb_config['dir'] = "./outputs/wandb"
        # Merge tags before passing to wandb.init to avoid duplicate keyword argument
        merged_tags = wandb_config.pop('tags', []) + [str(HydraConfig.get().job.name)]
        wandb_run = wandb.init(
            **wandb_config,
            id=random_job_id,
            tags=merged_tags,
            config=config_for_wandb_logging,
            reinit='create_new',
        )
    else:
        wandb_run = None

    # Train
    try:
        logger = train(train_dataloader, val_dataloader, model_copy, optimizer, training_params,
                    scheduler=scheduler, ckpt_dir=ckpt_dir,
                    logging_params=config['logging_params'], wandb_run=wandb_run)
    finally:
        if master_process and wandb_run is not None:
            wandb_run.finish()

    # Save
    if master_process:
        logger.name = opt_config['name'] + '-lr-' + str(opt_config['args']['lr'])

        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Overwriting")
        with open(output_path, 'w') as file:
            json.dump(logger.__dict__, file)
        print(f"Saved output to {output_path}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
