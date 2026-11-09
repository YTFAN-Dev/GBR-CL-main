import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from utils.simclrloss import SupConLoss


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via Supervised Contrastive Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--lambda_sc', type=float, default=0.1,
                        help='Weight of supervised contrastive loss.')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for SupCon loss.')
    parser.add_argument('--proj_dim', type=int, default=128,
                        help='Projection head output dim.')
    return parser


class Scr(ContinualModel):
    NAME = 'scr'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Scr, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.epoch = 0
        self.task = 0

        feat_dim = self.net.fc.in_features
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, self.args.proj_dim)
        ).to(self.device)
        self.supcon = SupConLoss(temperature=self.args.temperature,
                                 base_temperature=self.args.temperature)

        if 'meta' in self.args.backbone:
            self.opt = torch.optim.SGD(list(self.net.params()) + list(self.projection.parameters()), lr=self.args.lr)
        else:
            self.opt = torch.optim.SGD(list(self.net.parameters()) + list(self.projection.parameters()), lr=self.args.lr)

        self.class_samples = {}
        self.class_means = {}

    def _extract_features(self, x):
        with torch.no_grad():
            features = self.net(x, returnt='feature')
            features = F.normalize(features, p=2, dim=1)
        return features

    def _store_samples(self, not_aug_inputs, labels):
        quota = self.args.buffer_size // self.num_cls
        for i in range(labels.size(0)):
            lbl = labels[i].item()
            if lbl not in self.class_samples:
                self.class_samples[lbl] = []
            if len(self.class_samples[lbl]) < quota:
                self.class_samples[lbl].append(not_aug_inputs[i])
            else:
                r = np.random.randint(0, quota)
                self.class_samples[lbl][r] = not_aug_inputs[i]

    def _rebuild_buffer(self):
        self.buffer.empty()
        for lbl, samples in self.class_samples.items():
            samples = torch.stack(samples)
            self.buffer.add_data(
                examples=samples,
                labels=torch.full((samples.size(0),), lbl, dtype=torch.int64, device=self.device)
            )

    def _update_class_means_from_buffer(self):
        if self.buffer.is_empty():
            return
        buf_x, buf_lab = self.buffer.get_all_data()
        with torch.no_grad():
            features = self.net(buf_x, returnt='feature')
        features = F.normalize(features, p=2, dim=1)
        self.class_means = {}
        for lbl in torch.unique(buf_lab):
            lbl = lbl.item()
            mask = (buf_lab == lbl)
            self.class_means[lbl] = F.normalize(features[mask].mean(dim=0), p=2, dim=0)

    def forward(self, x):
        features = self._extract_features(x)
        logits = torch.full((x.size(0), self.num_cls), -1e9, device=x.device)
        if len(self.class_means) == 0:
            return logits
        means = torch.stack([self.class_means[c] for c in sorted(self.class_means.keys())]).to(x.device)
        cls_logits = features @ means.T
        for i, c in enumerate(sorted(self.class_means.keys())):
            logits[:, c] = cls_logits[:, i]
        return logits

    def observe(self, inputs, labels, not_aug_inputs):
        self._store_samples(not_aug_inputs, labels)
        self._rebuild_buffer()

        buf_inputs, buf_labels = None, None
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)

        x_all = inputs
        y_all = labels
        if buf_inputs is not None:
            x_all = torch.cat([inputs, buf_inputs], dim=0)
            y_all = torch.cat([labels, buf_labels], dim=0)

        outputs, features = self.net(x_all, returnt='all')
        proj = self.projection(features)

        ce_loss = self.loss(outputs, y_all.long())
        supcon_loss = self.supcon(proj.unsqueeze(1), y_all)
        loss = ce_loss + self.args.lambda_sc * supcon_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()

    def end_task(self, dataset):
        self._rebuild_buffer()
        self._update_class_means_from_buffer()
        self.task += 1
