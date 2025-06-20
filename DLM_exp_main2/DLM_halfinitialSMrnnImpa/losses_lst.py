# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn

# uttr to meaning
class MyLoss_lst:
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(
        self,
        sender_input: torch.Tensor,
        message,                            # listener prediction
        receiver_input: torch.Tensor,
        receiver_output: torch.Tensor,
        labels: torch.Tensor,               # meaning
        aux_input: Dict[str, torch.Tensor],
        eval=False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        accumulate_loss = 0

        batch_size = sender_input.size(0)
        # acc test
        eq = torch.eq(message, labels).sum(dim=1)

        eq_verb = torch.eq(message[:, 0], labels[:, 0])


        eq_subject = torch.eq(message[:, 1], labels[:, 1])
        eq_subjectm1 = torch.eq(message[:, 2], labels[:, 2])
        eq_subjectm2 = torch.eq(message[:, 3], labels[:, 3])
        eq_subjectm3 = torch.eq(message[:, 4], labels[:, 4])

        eq_object = torch.eq(message[:, 5], labels[:, 5])

        total_steps = labels.shape[1]
        acc = torch.eq(eq, total_steps) * 1.0
        
        acc_verb = torch.eq(eq_verb, int(1)) * 1.0

        acc_subject = torch.eq(eq_subject, int(1)) * 1.0
        acc_subjectm1 = torch.eq(eq_subjectm1, int(1)) * 1.0
        acc_subjectm2 = torch.eq(eq_subjectm2, int(1)) * 1.0
        acc_subjectm3 = torch.eq(eq_subjectm3, int(1)) * 1.0

        acc_object = torch.eq(eq_object, int(1)) * 1.0


        # not right
        # for r in receiver_output:
        #     for i in range(receiver_output[r].shape[1]):
        #         step_output = receiver_output[r][:, i, :].contiguous().view(batch_size, -1)
        #         l = self.criterion(step_output, labels[:, i])
        #         accumulate_loss += l

        for i in range(receiver_output[-1].shape[1]):
            step_output = receiver_output[-1][:, i, :].contiguous().view(batch_size, -1)
            l = self.criterion(step_output, labels[:, i])
            accumulate_loss += l

        # for i in range(receiver_output[-1].shape[1]):
        #     step_output = receiver_output[-1][:, i, :].contiguous().view(batch_size, -1)
        #     l = self.criterion(step_output, labels[:, i])
        #     accumulate_loss += l
            
        # return accumulate_loss, {"acc": acc, "acc_verb": acc_verb, "acc_subject": acc_subject, "acc_subjectm1": acc_subjectm1, "acc_subjectm2": acc_subjectm2, "acc_subjectm3": acc_subjectm3, "acc_object": acc_object, "acc_objectm1": acc_objectm1, "acc_objectm2": acc_objectm2, "acc_objectm3": acc_objectm3}
        # return accumulate_loss, {"acc": acc, "acc_verb": acc_verb, "acc_subject": acc_subject, "acc_subjectm1": acc_subjectm1, "acc_subjectm2": acc_subjectm2, "acc_subjectm3": acc_subjectm3, "acc_object": acc_object}
        return accumulate_loss, {"acc": acc, "acc_verb": acc_verb, "acc_subject": acc_subject, "acc_subjectm1": acc_subjectm1, "acc_subjectm2": acc_subjectm2, "acc_subjectm3": acc_subjectm3, "acc_object": acc_object}