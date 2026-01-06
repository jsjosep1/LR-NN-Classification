"""
TODO: Finish and submit your code for logistic regression, neural network, and hyperparameter search.

"""

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def logistic_regression(device):
    # TODO: implement logistic regression here

    results = dict(
        model=None,
    )

    return results


class FNN(nn.Module):
    def __init__(self, loss_type, num_classes):
        super(FNN, self).__init__()

        self.loss_type = loss_type
        self.num_classes = num_classes

        """add your code here"""

    def forward(self, x):
        output = None

        """add your code here"""
        return output

    def get_loss(self, output, target):
        loss = None

        """add your code here"""
        return loss


def tune_hyper_parameter(target_metric, device):
    # TODO: implement logistic regression and FNN hyper-parameter tuning here
    best_params = best_metric = None

    return best_params, best_metric