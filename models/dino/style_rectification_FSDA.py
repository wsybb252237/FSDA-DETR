import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from timm.models.layers import trunc_normal_

def style_extraction(x, eps=1e-6):
    mu = x.mean(dim=[2, 3])
    var = x.var(dim=[2, 3])
    sig = (var + eps).sqrt()
    mu = mu.detach()
    sig = sig.detach()
    return mu, sig

def momentum_update(old_value, new_value, momentum):
    update = momentum * old_value + (1 - momentum) * new_value
    return update



class StyleRectification(nn.Module):
    def __init__(self, num_prototype=2,
                 channel_size=64,
                 batch_size=4,
                 gamma=0.9,
                 dis_mode='abs',
                 channel_wise=False):
        super(StyleRectification, self).__init__()
        self.num_prototype = num_prototype
        self.channel_size = channel_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.dis_mode = dis_mode
        self.channel_wise = channel_wise
        self.style_mu = nn.Parameter(torch.zeros(self.num_prototype, self.channel_size),
                                     requires_grad=True)
        self.style_sig = nn.Parameter(torch.ones(self.num_prototype, self.channel_size),
                                      requires_grad=True)
        trunc_normal_(self.style_mu, std=0.02)

    def abs_distance(self, cur_mu, cur_sig, proto_mu, proto_sig, batch):
        cur_mu = cur_mu.view(batch, 1, self.channel_size)
        cur_sig = cur_sig.view(batch, 1, self.channel_size)
        proto_mu = proto_mu.view(1, self.num_prototype, self.channel_size)
        proto_sig = proto_sig.view(1, self.num_prototype, self.channel_size)

        cur_mu_sig = cur_mu / cur_sig
        proto_mu_sig = proto_mu / proto_sig
        distance = torch.abs(cur_mu_sig - proto_mu_sig)
        return distance

    def was_distance(self, cur_mu, cur_sig, proto_mu, proto_sig, batch):
        cur_mu = cur_mu.view(batch, 1, self.channel_size)
        cur_sig = cur_sig.view(batch, 1, self.channel_size)
        proto_mu = proto_mu.view(1, self.num_prototype, self.channel_size)
        proto_sig = proto_sig.view(1, self.num_prototype, self.channel_size)

        distance = (cur_mu - proto_mu).pow(2) + (cur_sig.pow(2) + proto_sig.pow(2) - 2 * cur_sig * proto_sig)
        return distance

    def kl_distance(self, cur_mu, cur_sig, proto_mu, proto_sig, batch):
        cur_mu = cur_mu.view(batch, 1, self.channel_size)
        cur_sig = cur_sig.view(batch, 1, self.channel_size)
        proto_mu = proto_mu.view(1, self.num_prototype, self.channel_size)
        proto_sig = proto_sig.view(1, self.num_prototype, self.channel_size)

        cur_mu = cur_mu.expand(-1, self.num_prototype, -1).reshape(batch * self.num_prototype, -1)
        cur_sig = cur_sig.expand(-1, self.num_prototype, -1).reshape(batch * self.num_prototype, -1)
        proto_mu = proto_mu.expand(batch, -1, -1).reshape(batch * self.num_prototype, -1)
        proto_sig = proto_sig.expand(batch, -1, -1).reshape(batch * self.num_prototype, -1)

        cur_distribution = torch.distributions.Normal(cur_mu, cur_sig)
        proto_distribution = torch.distributions.Normal(proto_mu, proto_sig)

        distance = torch.distributions.kl_divergence(cur_distribution, proto_distribution)
        distance = distance.reshape(batch, self.num_prototype, -1)
        return distance

    def forward(self, fea):
        batch = fea.shape[0]
        # [self.num_prototype:,c]
        proto_mu = self.style_mu.data.clone()
        proto_sig = self.style_sig.data.clone()

        # [b,channel]
        cur_mu, cur_sig = style_extraction(fea)


        if self.dis_mode == 'abs':
            distance = self.abs_distance(cur_mu, cur_sig, proto_mu, proto_sig, batch)
        elif self.dis_mode == 'was':
            distance = self.was_distance(cur_mu, cur_sig, proto_mu, proto_sig, batch)  #[batch,self.num_prototype, self.channel_size]
        elif self.dis_mode == 'kl':
            distance = self.kl_distance(cur_mu, cur_sig, proto_mu, proto_sig, batch)
        else:
            raise NotImplementedError('No this distance mode!')

        if not self.channel_wise:
            # [batch,self.num_prototype]
            distance = distance.mean(dim=2)


        # Normalize the values

        alpha = 1.0 / (1.0 + distance)
        # alpha = torch.exp(alpha) / torch.sum(torch.exp(alpha), dim=1, keepdim=True)
        # Map to the range 0-1 [batch,self.num_prototype]

        alpha = F.softmax(alpha, dim=1)

        # Compute the feature channel statistics

        if not self.channel_wise:
            # [batch,self.num_prototype] *  [self.num_prototype:c] = [batch,channel]
            # Apply weighting to the style features of the source domain

            mixed_mu = torch.mm(alpha, proto_mu)
            mixed_sig = torch.mm(alpha, proto_sig)

        else:
            proto_mu = proto_mu[None, ...]
            proto_sig = proto_sig[None, ...]
            mixed_mu = torch.sum(alpha * proto_mu, dim=1)
            mixed_sig = torch.sum(alpha * proto_sig, dim=1)


        # Scale the extracted features using the updated mean and variance
        fea = ((fea - cur_mu[:, :, None, None]) / cur_sig[:, :, None, None]) * mixed_sig[:, :, None, None] + mixed_mu[:,
                                                                                                             :, None,
                                                                                                             None]
        # Update the learned mu and sigma
        if self.training:
            # Learned mu and sigma
            proto_mu_update = self.style_mu.data.clone()
            proto_sig_update = self.style_sig.data.clone()

            for dataset_id in range(self.num_prototype):
                # Individually update the mu and sigma for the source domain and the target domain
                # Process according to the domain-wise partitioning of the cross-domain data in the mixed training

                mu = cur_mu[dataset_id * self.batch_size:(dataset_id + 1) * self.batch_size, ...].mean(dim=0)
                sig = cur_sig[dataset_id * self.batch_size:(dataset_id + 1) * self.batch_size, ...].mean(dim=0)

                proto_mu_update[dataset_id] = momentum_update(old_value=proto_mu_update[dataset_id], new_value=mu,
                                                              momentum=self.gamma)
                proto_sig_update[dataset_id] = momentum_update(old_value=proto_sig_update[dataset_id], new_value=sig,
                                                               momentum=self.gamma)

            self.style_mu = nn.Parameter(proto_mu_update, requires_grad=False)
            self.style_sig = nn.Parameter(proto_sig_update, requires_grad=False)

            if dist.is_available() and dist.is_initialized():
                proto_mu = self.style_mu.data.clone()
                dist.all_reduce(proto_mu.div_(dist.get_world_size()))
                self.style_mu = nn.Parameter(proto_mu, requires_grad=False)

                proto_sig = self.style_sig.data.clone()
                dist.all_reduce(proto_sig.div_(dist.get_world_size()))
                self.style_sig = nn.Parameter(proto_sig, requires_grad=False)
        #
        return fea