# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from scipy.spatial.distance import cdist
import numpy as np
from scipy.special import softmax

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., return_features=False):
        super().__init__()

        self.return_features = return_features

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        if self.return_features:
            return x

        return self.mlp_head(x)


class ViTwithGates(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, num_child_models=5, num_tasks=5, taskcla,
                 channels=3, dim_head=64, dropout=0., emb_dropout=0., s_gate=100):
        super().__init__()

        self.taskcla = taskcla
        self.s_gate = s_gate
        self.num_child_models = num_child_models
        self.child_models = []
        # self.nets = [ViT(image_size=image_size, patch_size=patch_size,
        #                  num_classes=num_classes, dim=dim, depth=depth,
        #                  dim_head=dim_head, dropout=dropout, heads=heads, mlp_dim=mlp_dim,
        #                  emb_dropout=emb_dropout, return_features=True) for _ in range(num_child_models)]

        self.nets = nn.ModuleList([ViT(image_size=image_size, patch_size=patch_size,
                                   num_classes=num_classes, dim=dim, depth=depth,
                                   dim_head=dim_head, dropout=dropout, heads=heads, mlp_dim=mlp_dim,
                                   emb_dropout=emb_dropout, return_features=True) for _ in range(num_child_models)])
        # Convert to cuda
        self.nets = [net.to('cuda') for net in self.nets]

        self.efc = nn.ModuleList([nn.Embedding(len(taskcla), 256) for _ in range(num_child_models)])

        self.last = nn.ModuleList()

        for t, n in self.taskcla:
            self.last.append(torch.nn.Linear(256, num_classes))
            self.ncls = n

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256)
        )

        # Task-specific gating mechanism
        self.num_tasks = num_tasks
        self.sigmoid_gate = nn.Sigmoid()

        # Task Prediction Branch
        self.task_scores_head = nn.Linear(256, num_tasks)

    def forward(self, img, task_id, return_experts=False, return_task_pred=False):

        masks = self.mask(task_id, s=self.s_gate)

        nets_features = [net(img) for net in self.nets]

        nets_outputs = [self.mlp_head(net_features) for net_features in nets_features]
        nets_outputs = [net_output * mask.expand_as(net_output) for (net_output, mask) in zip(nets_outputs, masks)]
        nets_outputs = [net_output.unsqueeze(0) for net_output in nets_outputs]
        self.child_models = nets_outputs

        total_outputs = torch.sum(torch.cat(nets_outputs, 0), dim=0).squeeze(0)

        logits = self.last[task_id](total_outputs)

        if return_experts:
            outputs_expert = nets_features
            logit_expert = nets_outputs
            return logits, outputs_expert, logit_expert

        if return_task_pred:
            torch.save(nets_outputs, '/home/luu/projects/cl_selective_nets/results/nets_outputs.pt')
            aggregated_features = torch.stack(nets_outputs).sum(dim=0)
            task_scores = self.task_scores_head(aggregated_features)
            task_predictions = nn.functional.softmax(task_scores, dim=1)
            predicted_task_id = torch.argmax(task_predictions, dim=1)

            return predicted_task_id, logits

        return logits

    def mask(self, task, s=100):
        total_gfc = [self.sigmoid_gate(s * efc(task)) for efc in self.efc]

        return total_gfc


class ViTwithSelectiveSubnets(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, num_tasks=5, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        self.child_models = []
        self.nets = nn.ModuleList([ViT(image_size=image_size, patch_size=patch_size,
                                       num_classes=num_classes, dim=dim, depth=depth,
                                       dim_head=dim_head, dropout=dropout, heads=heads, mlp_dim=mlp_dim,
                                       emb_dropout=emb_dropout, return_features=False) for _ in range(num_tasks)])

        self.subnet_performance = torch.zeros(num_tasks)

    def forward(self, img, task_id=None, mode='train'):

        if task_id is not None and mode == 'train':
            # selective nets
            selected_net = self.nets[task_id]

            logits = selected_net(img)

        # eval_upto_current_task: Means evaluate in a way that all classes in the past up to the current task
        # Example: current task is 2, classes = (4, 5). Task 0: (0, 1), Task 1: (2, 3). Then number of classes
        #          available is (0, 1, 2, 3, 4, 5). Not the classes in the future which are 6 -> 9
        elif mode == 'eval_upto_current_task':

            # Note that evaluate each single image, not batch
            outputs = []
            confidences = []

            # The idea is that which net produce extreme response to the current images
            for net in self.nets:

                output = net(img)  # Get logits for each subnet, (32, 10), 32 is batch_size
                # outputs.append(output)

                # Use numpy
                output_numpy = output.detach().clone().cpu().numpy()

                # Transforming to probabilities for comparison
                probs = softmax(output_numpy, axis=1)  # (32, 10)

                # Sum of probabilities
                """
                net_0 prediction
                [5.7552214e-04 1.3564850e+01 6.8130658e-04 6.1788253e-04 1.8430006e+01
                 2.8869390e-04 6.9698814e-04 5.0782855e-04 5.5186177e-04 1.2267524e-03]
                 
                 net_1 prediction
                [8.6937792e-04 1.6247439e-03 1.0769253e-03 2.6116767e-03 1.7811676e-03
                 1.3644723e-03 9.6003103e+00 2.6361849e-03 1.4609677e-03 2.2386261e+01]
                 .
                 .
                 .
                """
                sum_probs = np.sum(probs, axis=0)  # (1, 10)

                # Find the maximum response
                confidences.append(np.max(sum_probs))

            # Find in the index of maximum response out of 5 sub-nets
            best_model_index_confidence = np.argmax(confidences)

            # selective nets
            selected_net = self.nets[int(best_model_index_confidence)]

            logits = selected_net(img)

        # In this task, all classes are available in the beginning
        elif mode == 'eval_task_agnostic':
            confidences = []

            for net in self.nets:
                output = net(img)  # Assuming img is a single image tensor
                output_numpy = output.detach().cpu().numpy()  # Adjusted for clarity

                # Calculate softmax probabilities for each class
                probs = softmax(output_numpy, axis=1)  # Ensure correct axis for softmax

                # Take the max probability as the confidence score
                confidence_score = np.max(probs)
                confidences.append(confidence_score)

            # Select the model index with the highest confidence score
            best_model_index = np.argmax(confidences)
            selected_net = self.nets[int(best_model_index)]
            logits = selected_net(img)
        else:
            raise Exception("Choose correct mode: Train or eval")

        return logits

    def index_of_closest_to_zero(self, lst):
        # Initialize variables to store the minimum distance and the index of the closest element
        min_distance = float('inf')
        index_of_min = -1

        # Iterate through the list with index
        for index, value in enumerate(lst):
            # Calculate the absolute distance from 0
            distance = abs(value)

            # Update min_distance and index_of_min if the current distance is smaller
            if distance < min_distance:
                min_distance = distance
                index_of_min = index

        return index_of_min
