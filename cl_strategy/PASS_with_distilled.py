# https://github.com/Impression2805/CVPR21_PASS/blob/main/PASS.py

from typing import Iterable, Sequence, Optional, TypeVar, Union, List

import loguru
import numpy as np
import torch
import wandb
from avalanche.benchmarks.utils import concat_datasets
from avalanche.evaluation import metric_utils
from avalanche.evaluation.metrics import ConfusionMatrix
from avalanche.training.templates import SupervisedTemplate
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from avalanche.benchmarks import CLExperience
from avalanche.core import BasePlugin
from collections import defaultdict
from avalanche.evaluation.metrics import Accuracy, LossMetric, BWT, ExperienceForgetting, ForwardTransfer
from itertools import chain
TExperienceType = TypeVar("TExperienceType", bound=CLExperience)
TPluginType = TypeVar("TPluginType", bound=BasePlugin, contravariant=True)


def _group_experiences_by_stream(
        eval_streams: Iterable[Union[Iterable[CLExperience], CLExperience]]
) -> List[List[CLExperience]]:
    exps: List[CLExperience] = []
    # First, we unpack the list of experiences.
    for exp in eval_streams:
        if isinstance(exp, Iterable):
            exps.extend(exp)
        else:
            exps.append(exp)
    # Then, we group them by stream.
    exps_by_stream = defaultdict(list)
    for exp in exps:
        sname = exp.origin_stream.name
        exps_by_stream[sname].append(exp)
    # Finally, we return a list of lists.
    return list(list(exps_by_stream.values()))


def _experiences_parameter_as_iterable(
        experiences: Union[Iterable[TExperienceType], TExperienceType]
) -> Iterable[TExperienceType]:
    if isinstance(experiences, Iterable):
        return experiences
    else:
        return [experiences]