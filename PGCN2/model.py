import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A, dims):
        if dims == 2:
            # print(torch.mm(A.transpose(0, 1), x[0, 0, :, :])[:10, :5])
            x = torch.einsum('ncvl,vw->ncwl', (x, A))
            # print(x[0, 0, :10, :5])
            # sys.exit()
        elif dims == 3:
            # print(torch.mm(A[0, :, :].transpose(0, 1), x[0, 0, :, :])[:10, :5])
            x = torch.einsum('ncvl,nvw->ncwl', (x, A))
            # print(x[0, 0, :10, :5])
            # sys.exit()
        else:
            raise NotImplementedError('PGCN not implemented for A of dimension ' + str(dims))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        # import matplotlib.pyplot as plt
        out = [x]
        # print(x.size())
        # print(len(support))
        i = 0
        for a in support:
            # if i == 2:
            #     x1 = self.nconv(x, preva)
            # else:
            # print(x.size())
            # print(a.size())
            x1 = self.nconv(x, a, a.dim())
            # if i == 2:
            #     x1 = x1 * 0
            out.append(x1)

            for k in range(2, self.order + 1):
                # if i == 2:
                #     x2 = self.nconv(x1, preva)
                # else:
                x2 = self.nconv(x1, a, a.dim())
                # if i == 2:
                #     x2 = x2 * 0
                out.append(x2)
                x1 = x2
            i += 1
            # preva = a
            # if i == 2:
            #     break
                # for j in range(10):
                #     plt.imshow(x2.detach().cpu()[j, 0, :50, :])
                #     plt.colorbar()
                #     plt.show()
            # plt.imshow(a.cpu())
            # plt.colorbar()
            # plt.show()

        h = torch.cat(out, dim=1)
        # print(h.size())
        # print(self.mlp.weights.size())
        # sys.exit()
        # self.mlp.
        # h = torch.einsum('ncvl,vw->ncwl', (h, ))
        # h = self.mlp.weight
        h = self.mlp(h)
        # print(h.size())
        # sys.exit()
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None,
                 in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                 kernel_size=2, blocks=4, layers=2, adp=None):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.adp = adp

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                # self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                # self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.adpvec = nn.Parameter(torch.randn(12, 12).to(device), requires_grad=True).to(device)
                # self.supports_len += 2
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                # m, p, n = torch.svd(aptinit)
                # initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                # initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                # self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                # self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.adpvec = nn.Parameter(torch.randn(12, 12).to(device), requires_grad=True).to(device)
                self.supports_len += 1
                # self.supports_len += 2

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1, 1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        in_len = input.size(3)
        # print(input[:10, 0, 0, 0])
        # sys.exit()
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            # # average over batch
            # # print(torch.cuda.memory_allocated(device='cuda:0'))
            # xn = torch.mean(input[:, 0, :, :], dim=0)
            # xn = ((xn.transpose(0, 1) - torch.min(xn, dim=1)[0]) /
            #       (torch.max(xn, dim=1)[0] - torch.min(xn, dim=1)[0])).transpose(0, 1)
            # xn = (xn.transpose(0, 1) / torch.sqrt(torch.sum(xn ** 2, dim=1))).transpose(0, 1)
            # xn = torch.nan_to_num(xn, nan=1 / np.sqrt(12))
            # # xn = torch.nan_to_num(xn, nan=0)
            # adp = torch.mm(xn, self.adpvec[-in_len:, -in_len:])
            # adp = torch.mm(adp, xn.transpose(0, 1))
            # adp = F.softmax(F.relu(adp), dim=1)
            # print(torch.cuda.memory_allocated(device='cuda:0'))
            # sys.exit()
            # if len(torch.where(torch.isnan(adp))[0]) > 0:
            #     print(len(torch.where(torch.isnan(adp))[0]))
            #     import numpy as np
            #     np.save('/media/hdd1/YS/KDD22/wtf/input.npy', input.detach().cpu().numpy())
            #     np.save('/media/hdd1/YS/KDD22/wtf/xn.npy', xn.detach().cpu().numpy())
            #     np.save('/media/hdd1/YS/KDD22/wtf/adp.npy', adp.detach().cpu().numpy())
            #     np.save('/media/hdd1/YS/KDD22/wtf/adpvec.npy', self.adpvec.detach().cpu().numpy())
            #     sys.exit()
                # input.detach().cpu().numpy()

            # # not averaging
            xn = input[:, 0, :, -12:]
            xn = (xn - xn.min(dim=-1)[0].unsqueeze(-1)) / \
                 (xn.max(dim=-1)[0] - xn.min(dim=-1)[0]).unsqueeze(-1)
            # xn = torch.nan_to_num(xn, nan=1 / np.sqrt(12))
            xn = torch.nan_to_num(xn, nan=0.5)
            xn = xn / torch.sqrt((xn ** 2).sum(dim=-1)).unsqueeze(-1)
            adp = torch.einsum('nvt, tc->nvc', (xn, self.adpvec))
            adp = torch.bmm(adp, xn.permute(0, 2, 1))
            adp = F.softmax(F.relu(adp), dim=1)

            new_supports = self.supports + [adp]

            # adp2 = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            # new_supports = self.supports + [adp2]
            # new_supports = self.supports + [adp] + [adp2]
            #
            # del xn, adp, adp2

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            # print(torch.cuda.memory_allocated(device='cuda:0'))

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            # if i == 0:
            #     print(x[:5, 0, 0, :5])
            x = self.bn[i](x)
            # if i == 0:
            #     print(x[:5, 0, 0, :5])

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x





