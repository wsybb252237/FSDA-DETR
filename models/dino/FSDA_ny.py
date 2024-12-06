import torch
from torch.autograd import Function
from torch import nn
import torch.nn.functional as F
from util.misc import NestedTensor

def samples_and_targets_decomposition(samples, targets):
    B = samples.tensors.shape[0]

    src_samples = NestedTensor(samples.tensors[:B // 2], samples.mask[:B // 2])

    tgt_samples = NestedTensor(samples.tensors[B // 2:], samples.mask[B // 2:])

    src_targets = targets[:B//2]

    tgt_targets = targets[B//2:]

    return src_samples, tgt_samples, src_targets, tgt_targets


def feature_decomposition(srcs, masks, poss):


    B, _, _, _ = srcs[0].shape
    # source
    srcs_source = []
    masks_source = []
    poss_source = []

    # target
    srcs_target = []
    masks_target = []
    poss_target = []


    for i in range(len(srcs)):
        # source
        srcs_source.append(srcs[i][:B//2,:,:,:])
        masks_source.append(masks[i][:B//2,:,:])
        poss_source.append(poss[i][:B//2,:,:,:])

        # target
        srcs_target.append(srcs[i][B // 2:, :, :, :])
        masks_target.append(masks[i][B // 2:, :, :])
        poss_target.append(poss[i][B // 2:, :, :, :])
        '''
        Debug usage - Determine whether the feature maps, masks, 
        and positional encodings of the source and target domains are correctly divided.
        '''

    return srcs_source, masks_source, poss_source, srcs_target, masks_target, poss_target, srcs, masks, poss



def guassian_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Compute Gaussian kernel matrix
    :param x: Source domain features [batch_size, channels, height, width]
    :param y: Target domain features [batch_size, channels, height, width]
    :param kernel_mul: Kernel multiplication factor, used to calculate bandwidth
    :param kernel_num: Number of kernels
    :param fix_sigma: Fixed kernel bandwidth
    :return: Gaussian kernel matrix
    """
    n_samples = int(x.size()[0]) + int(y.size()[0])

    # Flatten 4D tensors to 2D [batch_size, features]
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)

    total = torch.cat([x, y], dim=0)  # Concatenate source and target domain features
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)  # Calculate L2 distance

    # Use fixed kernel bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # Calculate kernel values for each bandwidth
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val) / len(kernel_val)  # Return mean kernel matrix



def mmd_loss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    MMD loss calculation
    :param source: Source domain features [batch_size, features]
    :param target: Target domain features [batch_size, features]
    :param kernel_mul: Kernel multiplication factor
    :param kernel_num: Number of kernels
    :param fix_sigma: Fixed kernel bandwidth
    :return: MMD loss
    """
    batch_size = source.size(0)
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]  # Kernel matrix of source domain features
    YY = kernels[batch_size:, batch_size:]  # Kernel matrix of target domain features
    XY = kernels[:batch_size, batch_size:]  # Kernel matrix between source and target domain features
    YX = kernels[batch_size:, :batch_size]  # Kernel matrix between target and source domain features


    # Compute the MMD loss
    loss = torch.mean(XX + YY - XY - YX)
    return loss


class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha = 0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha
        return output


# GRLayer is applied to the input 𝑥
def grad_reverse(x):
    return GRLayer.apply(x)


# D is a convolutional neural network with two convolutional layers (Conv1 and Conv2) and a ReLU activation function
class FSDA_Discriminator(nn.Module):
    def __init__(self, dim):
        super(FSDA_Discriminator, self).__init__()
        self.dim = dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1, bias=False)
        self.Conv2 = nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=False)
        self.reLu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.reLu(self.Conv1(x))
        x = self.Conv2(x)
        return x

def get_valid_feature(mask):
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_h = valid_H
    valid_w = valid_W
    valid_ratio = torch.stack([valid_h, valid_w], -1)
    return valid_ratio