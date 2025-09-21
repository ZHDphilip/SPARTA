import torch
import torch.nn as nn


def build_conv(config, add_norm = True):

    assert isinstance(config['in_channels'], list)

    blocks = []

    for i in range(len(config['in_channels'])):

        blocks.append(
            nn.Conv2d(
                in_channels = config['in_channels'][i],
                out_channels = config['out_channels'][i],
                stride = config['stride'][i],
                kernel_size = config['kernel_size'][i],
                padding = config['padding'][i]
            )
        )

        if add_norm:
            blocks.append(
                nn.BatchNorm2d(num_features = config['out_channels'][i])
            )
        
        blocks.append(nn.ReLU())

    return nn.Sequential(*blocks)


def build_head(config, add_norm = True, norm_func = nn.LayerNorm):

    if isinstance(config['hidden_dims'], list):

        head_blocks = []

        if len(config['hidden_dims']) > 0:

            for i in range(len(config['hidden_dims'])):

                if i == 0:
                    head_blocks.append(
                        nn.Linear(
                            in_features = config['input_dim'],
                            out_features = config['hidden_dims'][i]
                        )
                    )
                else:
                    head_blocks.append(
                        nn.Linear(
                            in_features = config['hidden_dims'][i-1],
                            out_features = config['hidden_dims'][i]
                        )
                    )

                if add_norm:
                    head_blocks.append(
                        norm_func(config['hidden_dims'][i])
                    )

                head_blocks.append(
                    nn.ReLU()
                )
                
            head_blocks.append(
                nn.Linear(
                    in_features = config['hidden_dims'][-1],
                    out_features = config['output_dim']
                )
            )

    else:

        head_blocks = [
            nn.Linear(
                in_features = config['input_dim'],
                out_features = config['hidden_dims']
            )
        ]

        if add_norm:
            head_blocks.append(norm_func(config['hidden_dims']))
        
        head_blocks.append(nn.ReLU())

        head_blocks.append(
            nn.Linear(
                in_features = config['hidden_dims'],
                out_features = config['output_dim']
            )
        )

    return nn.Sequential(*head_blocks)