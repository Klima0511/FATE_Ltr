import torch as t
import torch.nn as nn

# 定义激活函数选择函数
def get_AF(activation_function):
    if activation_function == "GE":
        return nn.GELU()
    else:
        return nn.ReLU()


class LTRBatchNorm(nn.Module):
    """
    Note: given multiple query instances, the mean and variance are computed across queries.
    """
    def __init__(self, num_features, momentum=0.1, affine=True, track_running_stats=False):
        '''
        @param num_features: C from an expected input of size (N, C, L) or from input of size (N, L)
        @param momentum: the value used for the running_mean and running_var computation. Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1
        @param affine: a boolean value that when set to True, this module has learnable affine parameters. Default: True
        @param track_running_stats:
        '''
        super(LTRBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, X):
        '''
        @param X: Expected 2D or 3D input
        @return:
        '''
        if 3 == X.dim():
            return self.bn(X.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            return self.bn(X)

# 定义自定义模型类
class MyCustomModel(nn.Module):
    def __init__(self):
        super(MyCustomModel, self).__init__()
        # 模型参数
        N = 136  # 输入特征数量
        H = 100  # 隐藏层维度
        O = 1    # 输出层维度
        ff_dims = [N, H, H, H, O]  # 网络维度

        self.layers = nn.Sequential()
        for i in range(1, len(ff_dims)):
            self.layers.add_module(f'ff_{i}', nn.Linear(ff_dims[i - 1], ff_dims[i]))  # 线性层
            if i < len(ff_dims) - 1:  # 除了输出层外的层
                self.layers.add_module(f'dr_{i}', nn.Dropout(0.1))  # Dropout层
                self.layers.add_module(f'bn_{i}', LTRBatchNorm(ff_dims[i], affine=True))  # 批量归一化层
                self.layers.add_module(f'act_{i}', get_AF("GE"))  # GELU激活函数层

        # 添加尾部激活函数
        self.layers.add_module(f'act_tl', get_AF("GE"))

    def forward(self, x):
        return self.layers(x)