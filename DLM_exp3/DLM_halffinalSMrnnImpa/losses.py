# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

# list all possible utterances
language_dict = {'initial_none': 0, 'initial_uniform': 1, 'initial_skewed': 2, 'initial_medium': 3, 'final_skewed': 4, 'final_medium': 5, 'final_uniform': 6, 'initial_test': 7, 'initial_uniformlong': 8, 'initial_uniformlocal': 9, 'initial_skewedlong': 10, 'initial_skewedlocal': 11, 'initial_mediumlong': 12, 'initial_mediumlocal': 13, 'final_uniformlong': 14, 'final_uniformlocal': 15, 'final_skewedlong': 16, 'final_skewedlocal': 17, 'final_mediumlong': 18, 'final_mediumlocal': 19, 'initial': 20}
inv_language_dict = {v: k for k, v in language_dict.items()}


class MyLoss:
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(
        self,
        sender_input: torch.Tensor,
        labels: torch.Tensor,
        speaker_output: torch.Tensor,
        speaker_message: torch.Tensor,
        listener_output: torch.Tensor,
        listener_prediction: torch.Tensor,
        aux_input: Dict[str, torch.Tensor],
        is_training: True,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        batch_size = sender_input.size(0)
        total_steps = labels.shape[1]


        n_attributes = labels.size(1)
        total_loss = 0
        total_acc = 0
        correct_samples = (
            (listener_prediction == labels).view(batch_size, -1).detach()
        )
        acc = (torch.sum(correct_samples, dim=-1) == n_attributes).float()

        crible_acc=torch.zeros(size=speaker_message.size())
        crible_loss=torch.zeros(size=speaker_message.size())

        # for index, listener_out in enumerate(listener_output):
        #     if index == len(listener_output) - 1:
        #         weight = 1
        #         n_values = listener_out.size(-1)
        #         listener_out = listener_out.view(batch_size * total_steps, n_values) # original: [32, 6, 30] after: [32*6, 30]
        #         labels = labels.view(batch_size * n_attributes)
        #         loss = F.cross_entropy(listener_out, labels, reduction="none")
        #         loss = loss.view(batch_size, -1).mean(dim=1)
        #         loss = loss * weight
        #         total_loss += loss
        #     else:
        #         # continue
        #         labels_copy = labels.clone()
        #         weight = 0.6 ** (len(listener_output) - int(index) - 1)
        #         n_values = listener_out.size(-1)
        #         listener_out = listener_out[:, [0, 1, 5], :]
        #         listener_out = listener_out.view(batch_size * 3, n_values)
        #         labels_copy = labels_copy[:, [0, 1, 5]]
        #         # print(f"labels_copy: {labels_copy}")
        #         labels_copy = labels_copy.view(batch_size * 3)
        #         loss = F.cross_entropy(listener_out, labels_copy, reduction="none")
        #         loss = loss.view(batch_size, -1).mean(dim=1)
        #         loss = loss * weight
        #         total_loss += loss

        # for index, listener_out in enumerate(listener_output):
        #     if index == len(listener_output) - 1:
        #         weight = 1
        #         n_values = listener_out.size(-1)
        #         listener_out = listener_out.view(batch_size * total_steps, n_values) # original: [32, 6, 30] after: [32*6, 30]
        #         labels = labels.view(batch_size * n_attributes)
        #         loss = F.cross_entropy(listener_out, labels, reduction="none")
        #         loss = loss.view(batch_size, -1).mean(dim=1)
        #         loss = loss * weight
        #         total_loss += loss
        #     else:
        #         # continue
        #         labels_copy = labels.clone()
        #         # weight = 0.6 ** (len(listener_output) - int(index) - 1)
        #         weight = 1
        #         n_values = listener_out.size(-1)
        #         listener_out = listener_out[:, [0, 1, 5], :]
        #         listener_out = listener_out.view(batch_size * 3, n_values)
        #         labels_copy = labels_copy[:, [0, 1, 5]]
        #         # print(f"labels_copy: {labels_copy}")
        #         labels_copy = labels_copy.view(batch_size * 3)
        #         loss = F.cross_entropy(listener_out, labels_copy, reduction="none")
        #         loss = loss.view(batch_size, -1).mean(dim=1)
        #         loss = loss * weight
        #         total_loss += loss

        for index, listener_out in enumerate(listener_output):
            labels_copy = labels.clone()
            weight = 1
            n_values = listener_out.size(-1)
            listener_out = listener_out.view(batch_size * total_steps, n_values) # original: [32, 6, 30] after: [32*6, 30]
            labels_copy = labels_copy.view(batch_size * n_attributes)
            loss = F.cross_entropy(listener_out, labels_copy, reduction="none")
            loss = loss.view(batch_size, -1).mean(dim=1)
            loss = loss * weight
            total_loss += loss

        return total_loss, {"acc": acc}
        # return loss, {"acc": acc}

