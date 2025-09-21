import os

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset.deformation_dataset import DeformationDataset
from model.traversability_model import TerrainTraversabilityEncoder
from model.traversability_model import FourierTerrainTraversabilityEncoder
from model.traversability_model import TerrainTraversabilityEncoderAngleFree


def save_model(model, optimizer, scheduler, config, best=True, epoch=None):

    ckpt_path = config['train']['ckpt_dir'] + config['train']['experiment_name'] + "/"

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    if best:
        ckpt_path += "best.pth"
    elif epoch is not None:
        ckpt_path += f"epoch_{epoch}.pth"
    else:
        ckpt_path += "latest.pth"

    state_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }

    torch.save(state_dict, ckpt_path)


def load_model(
        config,
        training_set_length = None,
        load_model_state_dict = True,
        load_optimizer_state_dict = True,
        load_scheduler_state_dict = True,
        device = "cuda"
):

    ckpt_path = config['train']['ckpt_dir'] + config['train']['experiment_name'] + "/best.pth"

    if load_model_state_dict or load_optimizer_state_dict or load_scheduler_state_dict:
        assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} does not exist!"
        state_dict = torch.load(ckpt_path)

    # Initialize Model
    if config['network']['model_type'] == "fourier":
        model = FourierTerrainTraversabilityEncoder(config['network'])
    elif config['network']['model_type'] == "angle_free":
        model = TerrainTraversabilityEncoderAngleFree(config['network'])
    elif config['network']['model_type'] == "mlp":
        model = TerrainTraversabilityEncoder(config['network'])
    else:
        print(f"Unsupported model type {config['network']['model_type']}")
        exit(1)

    # Load the Model State Dict
    if load_model_state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    model.to(device)

    # Initialize Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr = config['train']['start_lr'],
        weight_decay = config['train']['weight_decay']
    )
    optimizer.zero_grad()

    # Load the Optimizer State Dict
    if load_optimizer_state_dict:
        assert 'optimizer_state_dict' in state_dict, "No optimizer state dict found in checkpoint!"
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    # Initialize the LR Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = config['train']['max_lr'],
        epochs = config['train']['epochs'],
        steps_per_epoch = int(training_set_length / config['train']['batch_size']),
        pct_start = config['train']['pct_start'],
        div_factor = config['train']['max_lr'] / config['train']['start_lr'],
        final_div_factor = config['train']['start_lr'] / config['train']['end_lr'],
        anneal_strategy = config['train']['anneal_strategy']
    )

    # Load the Scheduler State Dict
    if load_scheduler_state_dict:
        assert 'scheduler_state_dict' in state_dict, "No scheduler state dict found in checkpoint!"
        scheduler.load_state_dict(state_dict['scheduler_state_dict'])

    return model, optimizer, scheduler


def get_tensorboard_writer(config):

    logdir = config['train']['logdir'] + config['train']['experiment_name'] + "/"

    if(os.path.isdir(logdir)):
        filelist = [f for f in os.listdir(logdir)]
        for f in filelist:
            os.remove(os.path.join(logdir, f))
    writer = SummaryWriter(logdir)

    return writer


def get_dataset(config, split='train', plotting=False):

    assert split in ['train', 'test'], "Invalid split, should be 'train' or 'test'"
    assert config['network']['model_type'] == config['dataset']['mode'], \
        f"Network model type and dataset mode must be the same, got {config['network']['model_type']} and {config['dataset']['mode']}"

    # get base directory based on split
    base_dir = os.path.join(config['dataset']['base_dir'], split)

    # Retrieve DeformationDataset object
    dataset = DeformationDataset(
        base_dir = base_dir,
        angles_per_object = config['dataset']['angles_per_object'],
        max_points_per_object = config['dataset']['max_points_per_object'],
        add_noise = config['dataset']['add_noise'],
        noise_config = config['dataset']['noise_config'] if 'noise_config' in config['dataset'] else None,
        num_bins = config['dataset']['num_bins'],
        min_deformation = config['dataset']['min_deformation'],
        max_deformation = config['dataset']['max_deformation'],
        mode = config['dataset']['mode']
    )

    # If plotting is enabled, override batch size to 1
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = config['train']['batch_size'] if split == 'train' else config['dataset']['angles_per_object'],
        shuffle = split == 'train' if not plotting else True
    )

    # return dataset length as well for scheduler step calculation
    return dataloader, len(dataset)