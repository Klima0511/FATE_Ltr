import torch as t
from torch.nn.functional import mse_loss

class MSELoss(t.nn.Module):
    """
    A Mean Squared Error Loss that computes the mean of the squared differences
    between predictions and actual values.
    """

    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, label):
        # 使用 PyTorch 的内置 mse_loss 函数
        loss = mse_loss(pred, label, reduction=self.reduction)
        return loss
