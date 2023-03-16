import math

import torch
import torch.nn as nn
from torch import Tensor


class NormalizedEntropyLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ['none', 'mean', 'sum'], 'invalid reduction'
        self.reduction = reduction

    def forward(self, logits: Tensor):
        assert logits.dim() in [3, 4]
        dim = logits.shape[1]
        p_log_p = nn.functional.softmax(
            logits, dim=1) * nn.functional.log_softmax(logits, dim=1)
        ent = -1.0 * p_log_p.sum(dim=1)  # b x h x w OR b x n
        loss = ent / math.log(dim)
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


class CDCLoss(nn.Module):
    def __init__(self,
                 feat_dim: int = 128,
                 temperature: float = 0.3,
                 num_grid: int = 7,
                 queue_len: int = 65536,
                 warm_up_steps: int = 2500,
                 confidence_threshold: float = 0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.temperature = temperature
        self.num_grid = num_grid
        self.queue_len = queue_len
        self.warm_up_steps = warm_up_steps
        self.confidence_threshold = confidence_threshold

        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, p=2, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, emb_anc, emb_pos, confidence):
        # since the features are normalized afterwards, we don't need to worry about the
        # normalization factor during pooling (for a weighted average)
        emb_anc = emb_anc * confidence
        emb_anc = nn.functional.adaptive_avg_pool2d(emb_anc, self.num_grid).permute(0, 2, 3, 1).contiguous().view(-1, self.feat_dim)
        emb_anc_zero_mask = torch.linalg.norm(emb_anc, dim=1) != 0

        emb_pos = emb_pos * confidence
        emb_pos = nn.functional.adaptive_avg_pool2d(emb_pos, self.num_grid).permute(0, 2, 3, 1).contiguous().view(-1, self.feat_dim)
        emb_pos_zero_mask = torch.linalg.norm(emb_pos, dim=1) != 0

        avg_confidence = nn.functional.adaptive_avg_pool2d(confidence, self.num_grid).view(-1)
        avg_confidence_mask = avg_confidence >= self.confidence_threshold

        zero_mask = emb_anc_zero_mask & emb_pos_zero_mask & avg_confidence_mask

        emb_anc = emb_anc[zero_mask]
        emb_pos = emb_pos[zero_mask]
        emb_anc = nn.functional.normalize(emb_anc, p=2, dim=1)
        emb_pos = nn.functional.normalize(emb_pos, p=2, dim=1)

        l_pos_dense = torch.einsum('nc,nc->n', [emb_anc, emb_pos]).unsqueeze(-1)
        l_neg_dense = torch.einsum('nc,ck->nk', [emb_anc, self.queue.clone().detach()])

        logits = torch.cat((l_pos_dense, l_neg_dense), dim=1) / self.temperature
        labels = torch.zeros((logits.size(0), ), dtype=torch.long, device=logits.device)
        loss = nn.functional.cross_entropy(logits, labels, reduction='mean')

        return loss

    @torch.no_grad()
    def update_queue(self, emb_neg):
        emb_neg = nn.functional.adaptive_avg_pool2d(emb_neg, self.num_grid).permute(0, 2, 3, 1).contiguous().view(-1, self.feat_dim)
        emb_neg = nn.functional.normalize(emb_neg, p=2, dim=1)

        batch_size = emb_neg.shape[0]

        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.queue_len:
            sec1, sec2 = self.queue_len - \
                ptr, ptr + batch_size - self.queue_len
            emb_neg1, emb_neg2 = torch.split(emb_neg, [sec1, sec2], dim=0)
            self.queue[:, -sec1:] = emb_neg1.transpose(0, 1)
            self.queue[:, :sec2] = emb_neg2.transpose(0, 1)
        else:
            self.queue[:, ptr:ptr + batch_size] = emb_neg.transpose(0, 1)

        ptr = (ptr + batch_size) % self.queue_len  # move pointer
        self.queue_ptr[0] = ptr
