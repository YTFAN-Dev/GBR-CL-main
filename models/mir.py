# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Maximally Interfered Retrieval.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--mir_cands', type=int, default=50,
                        help='Number of candidate samples to consider for interference.')
    return parser


class Mir(ContinualModel):
    NAME = 'mir'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Mir, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.mir_replay_batch_size = self.args.minibatch_size
        self.mir_cands = self.args.mir_cands

    def get_stored_al(self, inputs, labels, buf_inputs, buf_labels):
        """标准 MIR 干扰检索（Aljundi et al., 2019）。

        1) 用当前新批次的梯度构造一步虚拟参数更新 theta_v = theta - lr * grad；
        2) 在 theta_v 下重新评估候选样本损失，干扰度 score = loss(theta_v) - loss(theta)；
        3) 返回干扰度最高的候选样本下标。

        干扰度评估放在 eval 模式且不更新 BatchNorm 统计量，评估完成后恢复训练模式。
        """
        bn_mods = [m for m in self.net.modules()
                   if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        bn_state = [(m,
                     None if m.running_mean is None else m.running_mean.clone(),
                     None if m.running_var is None else m.running_var.clone(),
                     None if m.num_batches_tracked is None else m.num_batches_tracked.clone())
                    for m in bn_mods]
        was_training = self.net.training
        try:
            outputs = self.net(inputs)
            loss_cur = self.loss(outputs, labels.long())
            grads = torch.autograd.grad(loss_cur, self.net.parameters())

            lr = self.opt.param_groups[0]['lr']
            self.net.eval()
            with torch.no_grad():
                pre_loss = self.loss(self.net(buf_inputs), buf_labels.long(), reduction='none')
                for p, g in zip(self.net.parameters(), grads):
                    p.data = p.data - lr * g
                try:
                    post_loss = self.loss(self.net(buf_inputs), buf_labels.long(), reduction='none')
                finally:
                    for p, g in zip(self.net.parameters(), grads):
                        p.data = p.data + lr * g
            scores = post_loss - pre_loss
            n_smp = min(self.mir_replay_batch_size, buf_inputs.shape[0])
            return torch.topk(scores, n_smp).indices
        finally:
            for m, rm, rv, nbt in bn_state:
                if rm is not None:
                    m.running_mean.copy_(rm)
                if rv is not None:
                    m.running_var.copy_(rv)
                if nbt is not None:
                    m.num_batches_tracked.copy_(nbt)
            self.net.train(was_training)

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.mir_cands, transform=self.transform)
            sel = self.get_stored_al(inputs, labels, buf_inputs, buf_labels)
            tinputs = torch.cat((inputs, buf_inputs[sel]))
            tlabels = torch.cat((labels, buf_labels[sel]))
        else:
            tinputs = inputs
            tlabels = labels
        outputs = self.net(tinputs)
        loss = self.loss(outputs, tlabels.long())
        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])
        return loss.item()
