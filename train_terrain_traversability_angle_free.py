import argparse
import yaml
from tqdm import tqdm

import torch

from model.loss import *
from my_utils.training import *


def train_terrain_traversability(config):

    # Initialize Dataset
    training_loader, dataset_length = get_dataset(
        config = config, split='train'
    )

    # Initialize Model, Optimier, and Learning Rate Scheduler
    model, optimizer, scheduler = load_model(
        config,
        training_set_length = dataset_length,
        load_model_state_dict = False,
        load_optimizer_state_dict = False,
        load_scheduler_state_dict = False,
        device = "cuda"
    )

    # Initialize tensorboard writer
    writer = get_tensorboard_writer(config)

    # Start training process
    batch_size = config['train']['batch_size']
    total_batches = int(dataset_length / batch_size)
    print(f"[INFO] Total Batches per Epoch: {total_batches}")

    num_batches = 0
    best_loss = 1e10

    for epoch in range(config['train']['epochs']):

        model.train()

        for i, data in tqdm(enumerate(training_loader)):

            pcds, distributions = data

            # Convert to CUDA Tensors
            pcds = pcds.float().cuda()
            distributions = distributions.float().cuda()

            input_dict = {
                'batched_pts': pcds
            }

            predicted_distributions = model(input_dict)
            predicted_distributions = predicted_distributions / predicted_distributions.sum(dim=-1, keepdim=True)
            distributions = distributions.view(-1, model.num_bins)

            loss = EMD2_loss(distributions, predicted_distributions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Logging
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('Train/Loss', loss.item(), num_batches)
            writer.add_scalar('Train/Learning_Rate', current_lr, num_batches)
            print('[%d, %5d] loss: %.5f' % (epoch+1, i+1, loss.item()))

            # Save best checkpoint
            if loss.item() < best_loss:
                best_loss = loss.item()
                save_model(
                    model = model,
                    optimizer = optimizer,
                    scheduler = scheduler,
                    config = config,
                    best=True
                )

            num_batches += 1

        # Evaluate every 5 epochs
        if epoch % 5 == 0:
            eval_loss = evaluate_terrain_traversability(config, model=model)
            writer.add_scalar('Eval/Loss', eval_loss, epoch)

        if epoch == config['train']['epochs'] - 1:
            save_model(
                model = model,
                optimizer = optimizer,
                scheduler = scheduler,
                config = config,
                epoch=epoch
            )


def evaluate_terrain_traversability(config, model=None):

    # Initialize Dataset
    eval_loader, dataset_length = get_dataset(
        config = config, split='test'
    )

    if model is None:
        model, _, _ = load_model(
            config,
            training_set_length = dataset_length,
            load_model_state_dict = True,
            load_optimizer_state_dict = False,
            load_scheduler_state_dict = False,
            device = "cuda"
        )
    
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for i, data in tqdm(enumerate(eval_loader)):

            pcds, distributions = data

            # Convert to CUDA Tensors
            pcds = pcds.float().cuda()
            distributions = distributions.float().cuda()

            input_dict = {
                'batched_pts': pcds
            }

            predicted_distributions = model(input_dict)
            predicted_distributions = predicted_distributions / predicted_distributions.sum(dim=-1, keepdim=True)
            distributions = distributions.view(-1, model.num_bins)

            loss = EMD2_loss(distributions, predicted_distributions)

            total_loss += loss.item()
            total_samples += pcds.size(0)

    avg_loss = total_loss / total_samples
    print(f"[EVAL] Average Loss: {avg_loss}")

    return avg_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train Terrain Traversability Model'
    )
    parser.add_argument(
        '--config', type=str, default='config/terrain_traversability.yaml',
        help='Path to the config file.'
    )
    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    train_terrain_traversability(config)