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

    print(f"total trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

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

            pcds, angles, distributions = data

            # Convert to CUDA Tensors
            pcds = pcds.float().cuda()
            angles = torch.hstack(angles).float().cuda()
            distributions = torch.hstack(distributions).float().cuda()
            distributions = distributions.view(-1, angles.shape[1], model.num_bins)

            input_dict = {
                'batched_pts': pcds,
                'heading_angles': angles
            }

            if config['network']['model_type'] == "fourier":

                # Training Fourier Model. Predict Fourier Coefficients and query with angles
                fourier_coefficients = model(input_dict)
                fourier_bases = model.fourier_basis(angles)
                predicted_distributions = torch.einsum('bik,bjk->bij', fourier_bases, fourier_coefficients)
                predicted_distributions = predicted_distributions.view(-1, angles.shape[1], model.num_bins)
                predicted_distributions = torch.nn.functional.sigmoid(predicted_distributions)
                predicted_distributions = predicted_distributions / predicted_distributions.sum(dim=-1, keepdim=True)
                loss = EMD2_loss(distributions, predicted_distributions)

                # If Compressed Sensing is enabled, add regularization loss (i.e. L1 norm of the coefficients)
                # The idea behind this is to encourage sparsity in the Fourier coefficients, while provide
                # a sufficient number of bases to reconstruct finer details of the signal.
                if config['train'].get('compressed_sensing', False):
                    reg_weight = config['train'].get('regularizer_weight', 0.1)
                    l1_norm = torch.norm(fourier_coefficients, p=1, dim=-1).mean()
                    writer.add_scalar('Train/L1_Regularization', l1_norm.item(), num_batches)

                    if epoch < 0.05 * config['train']['epochs']:
                        # Warm up period for the first 5% of training epochs
                        loss += reg_weight * l1_norm
                    else:
                        # After warm up, apply a smaller weight on the regularization term
                        loss += reg_weight*0.1 * l1_norm
            
            else:

                # Training MLP Model. Predict Distributions Directly
                predicted_distributions = model(input_dict)
                predicted_distributions = predicted_distributions / predicted_distributions.sum(dim=-1, keepdim=True)
                predicted_distributions = predicted_distributions.view(-1, angles.shape[1], model.num_bins)
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
    
    # model.eval()

    eval_losses = []

    with torch.no_grad():

        for i, data in tqdm(enumerate(eval_loader)):

            pcds, angles, distributions = data

            # Convert to CUDA Tensors
            pcds = pcds.float().cuda()
            angles = torch.hstack(angles).float().cuda()
            distributions = torch.hstack(distributions).float().cuda()
            distributions = distributions.view(-1, angles.shape[1], model.num_bins)

            input_dict = {
                'batched_pts': pcds,
                'heading_angles': angles
            }

            if config['network']['model_type'] == "fourier":

                # Training Fourier Model. Predict Fourier Coefficients and query with angles
                fourier_coefficients = model(input_dict)
                fourier_bases = model.fourier_basis(angles)
                predicted_distributions = torch.einsum('bik,bjk->bij', fourier_bases, fourier_coefficients)
                predicted_distributions = predicted_distributions.view(-1, angles.shape[1], model.num_bins)
                predicted_distributions = torch.nn.functional.sigmoid(predicted_distributions)
                predicted_distributions = predicted_distributions / predicted_distributions.sum(dim=-1, keepdim=True)
                loss = EMD2_loss(distributions, predicted_distributions)
            
            else:

                # Training MLP Model. Predict Distributions Directly
                predicted_distributions = model(input_dict)
                predicted_distributions = predicted_distributions / predicted_distributions.sum(dim=-1, keepdim=True)
                predicted_distributions = predicted_distributions.view(-1, angles.shape[1], model.num_bins)
                loss = EMD2_loss(distributions, predicted_distributions)

            eval_losses.append(loss.item())

    avg_loss = sum(eval_losses) / len(eval_losses) if eval_losses else 0
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