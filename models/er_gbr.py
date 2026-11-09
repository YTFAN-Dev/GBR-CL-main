import torch
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via Experience Replay with Granular Ball.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
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


class ErGBR(ContinualModel):
    NAME = 'er_gbr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErGBR, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.epoch = 0
        self.task = 0

        self.center_weight = args.center_weight
        self.edge_weight = args.edge_weight
        self.use_gb_weight = not args.no_gb_weight
        self.use_gaussian_sampling = not args.no_gaussian_sampling
        self.gaussian_sigma = args.gaussian_sigma
        self.buffer_k = args.buffer_k

    def _extract_features(self, x):
        with torch.no_grad():
            features = None
            
            # 方法1: 直接用 returnt='features' (ResNet18.py 标准用法)
            try:
                features = self.net(x, returnt='features')
            except:
                pass
            
            # 方法2: 如果失败，用 returnt='all' 取第二个返回值
            if features is None:
                try:
                    _, features = self.net(x, returnt='all')
                except:
                    pass
            
            # 方法3: 如果还失败，手动走 ResNet 前向提取特征
            if features is None:
                # ResNet18.py / ResNet_meta.py / ResNet.py 通用结构
                if hasattr(self.net, 'conv1'):
                    out = self.net.conv1(x)
                    if hasattr(self.net, 'bn1'):
                        out = self.net.bn1(out)
                    out = F.relu(out)
                    
                    if hasattr(self.net, 'layer1'):
                        out = self.net.layer1(out)
                        out = self.net.layer2(out)
                        out = self.net.layer3(out)
                        out = self.net.layer4(out)
                    elif hasattr(self.net, 'conv2_x'):
                        out = self.net.conv2_x(out)
                        out = self.net.conv3_x(out)
                        out = self.net.conv4_x(out)
                        out = self.net.conv5_x(out)
                    
                    out = F.avg_pool2d(out, out.shape[2])
                    features = out.view(out.size(0), -1)
            
            # L2 归一化
            features = F.normalize(features, p=2, dim=1)
            return features

    def _compute_center(self, features):
        return features.mean(dim=0)

    def _compute_radius(self, features, center):
        distances = torch.norm(features - center.unsqueeze(0), p=2, dim=1)
        return distances.mean()

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        center_w = self.center_weight
        edge_w = self.edge_weight

        buf_inputs = None
        buf_labels = None
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)

        x_all = inputs
        y_all = labels
        if buf_inputs is not None:
            x_all = torch.cat([inputs, buf_inputs], dim=0)
            y_all = torch.cat([labels, buf_labels], dim=0)
        n = x_all.shape[0]

        if self.use_gb_weight:
            with torch.no_grad():
                z_all = self._extract_features(x_all)

                # 按标签类别直接分裂为粒球（每个类别一个粒球）
                unique_labels = torch.unique(y_all)
                granular_balls = []
                for lbl in unique_labels:
                    mask = (y_all == lbl)
                    gb_features = z_all[mask]
                    gb_indices = torch.where(mask)[0]
                    gb_center = gb_features.mean(dim=0)
                    gb_distances = torch.norm(gb_features - gb_center.unsqueeze(0), p=2, dim=1)
                    gb_radius = gb_distances.mean()
                    granular_balls.append({
                        'features': gb_features,
                        'labels': y_all[mask],
                        'indices': gb_indices,
                        'center': gb_center,
                        'radius': gb_radius
                    })

                # 收集中心样本和边缘样本的索引，离群样本（>2*radius）直接剔除
                center_indices_list = []
                edge_indices_list = []
                for gb in granular_balls:
                    gb_features = gb['features']
                    gb_indices = gb['indices']
                    gb_center = gb['center']
                    gb_radius = gb['radius']

                    distances = torch.norm(gb_features - gb_center.unsqueeze(0), p=2, dim=1)
                    center_mask = distances <= gb_radius
                    edge_mask = (distances > gb_radius) & (distances <= 2 * gb_radius)

                    center_idx = gb_indices[center_mask]
                    edge_idx = gb_indices[edge_mask]

                    if center_idx.size(0) > 0:
                        center_indices_list.append(center_idx)
                    if edge_idx.size(0) > 0:
                        edge_indices_list.append(edge_idx)

        if self.use_gb_weight:
            outputs_all = self.net(x_all)
            loss_all = self.loss(outputs_all, y_all.long(), reduction='none')

            # 构建权重向量，默认权重为0（离群样本不参与损失）
            sample_weights = torch.zeros(n, device=self.device)

            if len(center_indices_list) > 0:
                all_center_idx = torch.cat(center_indices_list, dim=0)
                sample_weights[all_center_idx] = center_w

            if len(edge_indices_list) > 0:
                all_edge_idx = torch.cat(edge_indices_list, dim=0)
                sample_weights[all_edge_idx] = edge_w

            # 除以权重之和
            valid_mask = sample_weights > 0
            if valid_mask.sum() == 0:
                total_loss = loss_all.mean()
            else:
                total_loss = (loss_all[valid_mask] * sample_weights[valid_mask]).sum() / sample_weights[valid_mask].sum()
        else:
            # 关闭粒球加权：使用与ER基线完全一致的损失计算方式
            total_loss = self.loss(self.net(inputs), labels.long())
            if buf_inputs is not None:
                total_loss += self.loss(self.net(buf_inputs), buf_labels.long())

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        # ========== 按类别分别构建粒球 + 按类别高斯采样（仅用于选存储样本） ==========
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

                    cls_center = self._compute_center(cls_features)
                    cls_radius = self._compute_radius(cls_features, cls_center)

                    cls_distances = torch.norm(cls_features - cls_center.unsqueeze(0), p=2, dim=1)

                    cls_prob = torch.exp(- (cls_distances ** 2) / (2 * self.gaussian_sigma ** 2))
                    cls_prob = cls_prob / cls_prob.sum()

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
            self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        return total_loss.item()
