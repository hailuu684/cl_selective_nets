import torch
import timm
import torch.nn as nn


class FeatureReducer(nn.Module):
    def __init__(self, desire_latent_dim=512, initial_dim=5120):
        super(FeatureReducer, self).__init__()
        valid_latent_dim = [4096, 2048, 1024, 512, 384, 256, 128, 64, 32]
        assert desire_latent_dim in valid_latent_dim, 'Invalid latent dim'

        self.linear_layers = nn.ModuleList()
        current_dim = initial_dim
        for i, latent_dim in enumerate(valid_latent_dim):

            if latent_dim < current_dim:
                self.linear_layers.append(nn.Linear(current_dim, latent_dim))
                current_dim = latent_dim
                if latent_dim == desire_latent_dim:
                    break

    def forward(self, x):
        for layer in self.linear_layers:
            x = layer(x)
        return x


class CustomModel(nn.Module):
    def __init__(self, base_model, feature_reducer):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.feature_reducer = feature_reducer

    def forward(self, x, return_spatial_features=False):
        summary, spatial_features = self.base_model(x)
        reduced_summary = self.feature_reducer(summary)

        if return_spatial_features:
            return reduced_summary, spatial_features
        else:
            return reduced_summary


def radio_v2(args, desire_latent_dim=512, model_version='radio_v2'):

    valid_radio_model = ['radio_v2', 'e-radio_v2']

    assert model_version in valid_radio_model, 'Invalid radio model'

    base_model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version,
                                progress=True, skip_validation=True)

    base_model.eval()

    if "e-radio" in model_version:
        base_model.model.set_optimal_window_size((args.size, args.size))  # where it expects a tuple of (height, width) of the input image.

    # Create the FeatureReducer
    # Adjust initial_dim to match the output dimension of the summary from the RADIO model

    if model_version == 'radio_v2':
        initial_dim = 5120
    else:
        initial_dim = 1536

    feature_reducer = FeatureReducer(desire_latent_dim=desire_latent_dim, initial_dim=initial_dim)  # Adjust `initial_dim` as needed

    # Create the custom wrapper model
    custom_model = CustomModel(base_model, feature_reducer)

    if torch.cuda.is_available():
        custom_model.to("cuda")

    return custom_model

