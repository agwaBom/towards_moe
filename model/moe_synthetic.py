import torch
import torch.nn as nn
import torch.nn.functional as F

class Custom_Conv1d(nn.Module):
    def __init__(self, filter_size, patch_dim, linear=True, small_init=True):
        super(Custom_Conv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=filter_size*2, kernel_size=int(patch_dim), stride=int(patch_dim))
        self.filter_size = filter_size
        self.linear = linear
        if small_init:
            self.conv.weight = torch.nn.Parameter(self.conv.weight*0.0001) 
            self.conv.bias = torch.nn.Parameter(self.conv.bias*0.0001) 

    def cubic_act(self, x):
        return x**3

    def forward(self, x):
        # x: [num_data, 1, num_patches * patch_dim]
        x = self.conv(x)
        # x: [num_data, filter_size, num_patches]
        if self.linear == False:
            x = self.cubic_act(x)
        x = torch.sum(x, dim=-1)
        output = torch.stack([torch.sum(x[:,:self.filter_size],1), torch.sum(x[:,self.filter_size:],1)]).transpose(1,0)

        return output

class MoE(nn.Module):
    def __init__(self, num_patches, num_expert, patch_dim, filter_size, linear=True, strategy='top-1'):
        super(MoE, self).__init__()
        self.gating_param = nn.Conv1d(in_channels=1, out_channels=num_expert, kernel_size=int(patch_dim*num_patches/num_patches), stride=int(patch_dim*num_patches/num_patches))
        self.softmax = nn.Softmax(dim=-1)
        self.num_expert = num_expert

        self.strategy = strategy # Gating strategy

        self.expert = nn.ModuleList()
        for i in range(num_expert):
            self.expert.append(Custom_Conv1d(filter_size, patch_dim, linear))

        # Zero init
        self.gating_param.weight = torch.nn.Parameter(self.gating_param.weight * 0)

    def forward(self, x):
        # x [num_data, 1, num_patches * patch_dim]

        # h(x; Θ): [num_data, num_expert, patch_dim]
        h = self.gating_param(x)

        # h: [num_data, num_expert]
        h = torch.sum(h, dim=-1)

        # Calculate π : [num_data, num_expert]
        if self.strategy == 'top-1':
            h = h + torch.rand(x.shape[0], x.shape[1])

            # this will be argmax top 1 routing
            pi_val, pi_idx = h.topk(k=1, dim=-1) 

            mask = F.one_hot(pi_idx, self.num_expert).type(torch.float)
            density = mask.mean(dim=0)
            density_h = h.mean(dim=0)
            loss = (density_h * density).mean() * float(self.num_expert ** 2)

            # We did not implement mask_flat that controls exceeded expert_capacity
            # Original paper not used it
            combine_tensor = (pi_val[..., None] * mask)

            dispatch_tensor = torch.einsum('bne->ben', mask)
            
            # x: [num_data, 1, num_patches * patch_dim]
            # dispatch_tensor: [num_data, num_expert, 1]
            # expert_inputs: [num_expert, num_data, 1, num_patches * patch_dim]
            expert_inputs = torch.einsum('bnd,ben->ebd', x, dispatch_tensor).unsqueeze(dim=2)
            output = []
            for i in range(self.num_expert):
                output.append(self.expert[i](expert_inputs[i]))
            output = torch.stack(output, dim=0)
            # output: [num_expert, num_data, 2]

            # dispatch_tensor [num_data, num_expert, 1]
            combine_tensor = torch.transpose(combine_tensor, dim0=-2, dim1=-1)
            output = torch.einsum("dei,edj->dj", combine_tensor, output)

        else:
            # [to-do] soft-routing model
            pi_val = self.softmax(h)

            # pi_val, pi_idx: [num_data, num_expert]
            # pi_idx returns order of idx from lowest to highest value
            # [lowest_value_idx, ..., highest_value_idx]
            pi_idx = torch.argsort(pi_val, dim=-1)

        return output, dispatch_tensor.squeeze(dim=-1), loss
