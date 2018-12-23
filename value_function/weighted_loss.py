import torch
from torch._jit_internal import weak_module, weak_script_method
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss


@weak_module
class WeightedMSELoss(_Loss):
    """
    Weights only the 0.5 values. (All samples are weighted 1 except for the values where target is 0.5)
    Samples with target 0.5 are weighted 1 / (count of samples with target 0.5).

    Can be extended to do a value counts with np.unique(target, return_counts=True).
    But that will be slower and will likely not provide any real gain now.
    """
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)

    @weak_script_method
    def forward(self, input_, target):
        mse_loss = (input_ - target) ** 2

        value_mask = target == 0.5
        value_count = torch.sum(value_mask, dtype=torch.float)
        weights = torch.ones_like(target)
        weights[value_mask] = 1 / value_count

        weighted_loss = mse_loss * Variable(weights).expand_as(target)

        # target_numpy = target.data.cpu().numpy()
        # Not weighted, as weights only affect where target is 0.5, where loss = 0 for baseline guess
        baseline_loss = (torch.ones_like(target) * 0.5 - target) ** 2
        print(f"baseline_loss: {baseline_loss.mean()}")
        if self.reduction != 'none':
            return torch.mean(weighted_loss) if self.reduction == 'mean' else torch.sum(weighted_loss)

        return weighted_loss


