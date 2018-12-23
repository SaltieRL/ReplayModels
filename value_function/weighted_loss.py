import logging

import torch
from torch._jit_internal import weak_module, weak_script_method
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

logger = logging.getLogger(__name__)


@weak_module
class WeightedMSELoss(_Loss):
    """
    Weights only the 0.5 values. (All samples are weighted 1 except for the values where target is 0.5)
    If dynamic_weights,
        Samples with target 0.5 are weighted 1 / (count of samples with target 0.5).
    otherwise
        Samples with target 0.5 are weighted 1/5.

    Can be extended to do a value counts with np.unique(target, return_counts=True).
    But that will be slower and will likely not provide any real gain now.
    """

    def __init__(self, size_average=None, reduce=None, reduction='mean', dynamic_weights: bool = True):
        super().__init__(size_average, reduce, reduction)
        self.dynamic_weights = dynamic_weights

    @weak_script_method
    def forward(self, input_, target):
        mse_loss = (input_ - target) ** 2

        value_mask = target == 0.5
        weights = torch.ones_like(target)

        if self.dynamic_weights:
            value_count = torch.sum(value_mask, dtype=torch.float)
            weights[value_mask] = 1 / value_count
        else:
            weights[value_mask] = 1 / 5

        weighted_loss = mse_loss * Variable(weights).expand_as(target)

        # target_numpy = target.data.cpu().numpy()
        # Not weighted, as weights only affect where target is 0.5, where loss = 0 for baseline guess
        baseline_loss = (torch.ones_like(target) * 0.5 - target) ** 2
        logger.info(f"baseline mean loss: {baseline_loss.mean()}")
        logger.info(f"mean loss: {torch.mean(weighted_loss)}")

        if self.reduction != 'none':
            return torch.mean(weighted_loss) if self.reduction == 'mean' else torch.sum(weighted_loss)

        return weighted_loss
