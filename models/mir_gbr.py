# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Maximally Interfered Retrieval with Granular Ball.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--mir_cands', type=int, default=50,
                        help='Number of candidate samples to consider for interference.')
    parser.add_argument('--center_weight', type=float, default=1.81,
                        help='Weight for center samples.')
    parser.add_argument('--edge_weight', type=float, default=0.33,
                        help='Weight for edge samples.')
    parser.add_argument('--no_gb_weight', action='store_true', default=False,
                        help='Disable granular ball weighted loss.')
    parser.add_argument('--no_gaussian_sampling', action='store_true', default=False,
                        help='Disable Gaussian sampling for buffer storage.')
    parser.add_argument('--gaussian_sigma', type=float, default=0.91,
                        help='高斯分布的标准差，用于新样本粒球的取样概率计算。')
    parser.add_argument('--buffer_k', type=int, default=3,
                        help='缓冲区采样分母，采样数 = 新样本数 / k。')
    return parser


class MirGBR(ContinualModel):
    NAME = 'mir_gbr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(MirGBR, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.mir_replay_batch_size = self.args.minibatch_size
        self.mir_cands = self.args.mir_cands

        self.center_weight = args.center_weight
        self.edge_weight = args.edge_weight
        self.use_gb_weight = not args.no_gb_weight
        self.use_gaussian_sampling = not args.no_gaussian_sampling
        self.gaussian_sigma = args.gaussian_sigma
        self.buffer_k = args.buffer_k

    def _extract_features(self, x):
        with torch.no_grad():
            features = None
            try:
                features = self.net(x, returnt='feature')
            except Exception:
                pass
            if features is None:
                try:
                    _, features = self.net(x, returnt='all')
                except Exception:
                    pass
            features = F.normalize(features, p=2, dim=1)
            return features

    def _compute_gb_weights(self, features, labels):
        n = features.shape[0]
        sample_weights = torch.zeros(n, device=self.device)
        unique_labels = torch.unique(labels)
        for lbl in unique_labels:
            mask = (labels == lbl)
            gb_features = features[mask]
            gb_indices = torch.where(mask)[0]
            gb_center = gb_features.mean(dim=0)
            gb_distances = torch.norm(gb_features - gb_center.unsqueeze(0), p=2, dim=1)
            gb_radius = gb_distances.mean()
            center_mask = gb_distances <= gb_radius
            edge_mask = (gb_distances > gb_radius) & (gb_distances <= 2 * gb_radius)
            sample_weights[gb_indices[center_mask]] = self.center_weight
            sample_weights[gb_indices[edge_mask]] = self.edge_weight
        return sample_weights

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
        if self.use_gb_weight:
            with torch.no_grad():
                z = self._extract_features(tinputs)
                weights = self._compute_gb_weights(z, tlabels)
            ce = self.loss(outputs, tlabels.long(), reduction='none')
            valid = weights > 0
            if valid.sum() == 0:
                loss = self.loss(outputs, tlabels.long())
            else:
                loss = (ce[valid] * weights[valid]).sum() / weights[valid].sum()
        else:
            loss = self.loss(outputs, tlabels.long())
        loss.backward()
        self.opt.step()

        if self.use_gaussian_sampling:
            with torch.no_grad():
                new_samples_features = self._extract_features(inputs)
                unique_labels = torch.unique(labels)
                total_N = real_batch_size
                K = max(1, total_N // self.buffer_k)

                all_selected_not_aug = []
                all_selected_labels = []

                for cls in unique_labels:
                    cls_mask = (labels == cls)
                    cls_indices = torch.where(cls_mask)[0]
                    n_c = cls_indices.size(0)

                    cls_features = new_samples_features[cls_indices]
                    cls_center = cls_features.mean(dim=0)
                    cls_radius = torch.norm(cls_features - cls_center.unsqueeze(0), p=2, dim=1).mean()

                    cls_distances = torch.norm(cls_features - cls_center.unsqueeze(0), p=2, dim=1)
                    cls_prob = torch.exp(- (cls_distances ** 2) / (2 * self.gaussian_sigma ** 2))
                    cls_prob = torch.nan_to_num(cls_prob, nan=0.0)
                    cls_sum = cls_prob.sum()
                    if not torch.isfinite(cls_sum) or cls_sum <= 0:
                        cls_prob = torch.ones_like(cls_prob) / cls_prob.numel()
                    else:
                        cls_prob = cls_prob / cls_sum

                    cls_sample_num = max(1, round(K * n_c / total_N))
                    cls_sample_num = min(cls_sample_num, n_c)

                    cls_selected_local = torch.multinomial(
                        cls_prob, num_samples=cls_sample_num, replacement=False
                    )
                    cls_selected_global = cls_indices[cls_selected_local]

                    all_selected_not_aug.append(not_aug_inputs[cls_selected_global])
                    all_selected_labels.append(labels[cls_selected_global])

                if len(all_selected_not_aug) > 0:
                    selected_not_aug = torch.cat(all_selected_not_aug, dim=0)
                    selected_labels = torch.cat(all_selected_labels, dim=0)
                    if selected_not_aug.size(0) > K:
                        trim_indices = torch.randperm(selected_not_aug.size(0), device=self.device)[:K]
                        selected_not_aug = selected_not_aug[trim_indices]
                        selected_labels = selected_labels[trim_indices]
                else:
                    selected_not_aug = not_aug_inputs[:K]
                    selected_labels = labels[:K]
            self.buffer.add_data(examples=selected_not_aug, labels=selected_labels)
        else:
            self.buffer.add_data(examples=not_aug_inputs,
                                 labels=labels[:real_batch_size])
        return loss.item()
