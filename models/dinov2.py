import torch
import loguru


def dinov2(model_size='s', with_register=False):

    if model_size == 's':
        latent_dim = 384
        loguru.logger.info("Latent dim of dinov2 is 384")
    elif model_size == 'b':
        latent_dim = 768
        loguru.logger.info("Latent dim of dinov2 is 768")
    elif model_size == 'l':
        latent_dim = 1024
        loguru.logger.info("Latent dim of dinov2 is 1024")
    else:
        raise ValueError("model size of dinov2 is not correct!")

    if with_register:
        model_type_reg = f'dinov2_vit{model_size}14_reg'
        dinov2_model = torch.hub.load('facebookresearch/dinov2', model_type_reg)
    else:
        model_type = f'dinov2_vit{model_size}14'
        dinov2_model = torch.hub.load('facebookresearch/dinov2', model_type)

    return dinov2_model

