import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via Repeated Augmented Rehearsal (RAR).')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--mem_iters', type=int, default=5,
                        help='Number of rehearsal iterations per batch.')
    parser.add_argument('--mem_ratio', type=float, default=1.0,
                        help='Weight for the memory (replay) loss.')
    parser.add_argument('--incoming_ratio', type=float, default=1.0,
                        help='Weight for the incoming (current) loss.')
    return parser


class Rar(ContinualModel):
    NAME = 'rar'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Rar, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.mem_iters = args.mem_iters
        self.mem_ratio = args.mem_ratio
        self.incoming_ratio = args.incoming_ratio
        self.epoch = 0
        self.task = 0

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]
        loss = 0.0

        for _ in range(self.mem_iters):
            if not self.buffer.is_empty():
                buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size,
                                                              transform=self.transform)
                x_all = torch.cat([inputs, buf_inputs], dim=0)
                y_all = torch.cat([labels, buf_labels], dim=0)
                mem_num = buf_inputs.shape[0]
            else:
                x_all = inputs
                y_all = labels
                mem_num = 0

            outputs = self.net(x_all)
            losses = self.loss(outputs, y_all.long(), reduction='none')

            if mem_num > 0:
                mem_loss = losses[:mem_num].mean()
                incoming_loss = losses[mem_num:].mean()
                loss = self.mem_ratio * mem_loss + self.incoming_ratio * incoming_loss
            else:
                loss = losses.mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels)

        return loss.item()
