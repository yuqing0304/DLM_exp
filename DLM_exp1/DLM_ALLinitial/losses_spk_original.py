# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import numpy as np
import torch
from utils import cutting_length

# list all possible utterances
language_dict = {'initial': 0, 'initial_long': 1, 'initial_local': 2, 'initial_none': 3, 'final_long': 4, 'final_local': 5, 'final_none': 6, 'initial_uniform': 7, 'initial_skewed': 8, 'initial_medium': 9, 'initial_test': 10, 'initial_uniformlong': 11, 'initial_uniformlocal': 12, 'initial_skewedlong': 13, 'initial_skewedlocal': 14, 'initial_mediumlong': 15, 'initial_mediumlocal': 16}
inv_language_dict = {v: k for k, v in language_dict.items()}


def vari_len_acc_compute(labels, message, valid_len):

    batch_size = labels.size(0)

    corr_ = []
    for i in range(batch_size):
        valid_l = valid_len[i]

        eq = torch.eq(message[i, :valid_l], labels[i, 1:valid_l+1]).sum()
        corr_.append(torch.eq(eq, valid_l))

    corr = torch.stack(corr_)

    return corr

def acc_eval_compute(labels, aux_inpute, message, valid_len, pad_id):

    language = inv_language_dict[aux_inpute['language'][0].item()]

    corr1 = vari_len_acc_compute(labels, message, valid_len)

    mk_idx = aux_inpute['mk_idx'][0].item()
    inanimate_idx1 = aux_inpute['inanimate_idx1'][0].item()
    inanimate_idx2 = aux_inpute['inanimate_idx2'][0].item()
    inanimate_idx3 = aux_inpute['inanimate_idx3'][0].item()


    mk_locate = torch.nonzero(torch.eq(labels, mk_idx))
    inanimate_locate1 = torch.nonzero(torch.eq(labels, inanimate_idx1))
    inanimate_locate2 = torch.nonzero(torch.eq(labels, inanimate_idx2))
    inanimate_locate3 = torch.nonzero(torch.eq(labels, inanimate_idx3))

    inanimate_locate = torch.cat((inanimate_locate1, inanimate_locate2, inanimate_locate3), dim=0)
    # print("inanimate_locate", inanimate_locate)

    sorted_inanimate_locate = torch.tensor(sorted(inanimate_locate.tolist(), key=lambda x: x[0]))



    if language == 'initial_local' or language == 'initial_long' or language == 'final_local' or language == 'final_long' or language == 'initial_test' or language == 'initial_uniformlong' or language == 'initial_skewedlong' or language == 'initial_mediumlong' or language == 'initial_uniformlocal' or language == 'initial_skewedlocal' or language == 'initial_mediumlocal':
        acc = corr1 * 1.0    
    elif language == 'initial_none' or language == 'initial_uniform' or language == 'initial_skewed' or language == 'initial_medium':
        to7_3 = [0, 1, 6, 7, 2, 3, 4, 5, 8]
        to4_3 = [0, 1, 3, 4, 5, 6, 7, 2, 8]

        to3_7 = [0, 1, 4, 5, 6, 7, 2, 3, 8]
        to3_4 = [0, 1, 7, 2, 3, 4, 5, 6, 8]

        new_labels = []
        new_l = None

        if mk_locate.size(0) != labels.size(0):
            acc = corr1 * 1.0
            return acc

        for i, j in mk_locate:
            if j == 7:
                new_l = labels[i][to7_3]
            elif j == 4:
                new_l = labels[i][to4_3]
            elif j == 3:
                if sorted_inanimate_locate[i][1] == 7:
                    new_l = labels[i][to3_7]
                elif sorted_inanimate_locate[i][1] == 6:
                    new_l = labels[i][to3_4]  
            else:
                new_l = None
            new_labels.append(new_l)
        new_labels = torch.stack(new_labels)
        # print("labels", labels)
        # print("new_labels", new_labels)
        # exit(0)

        corr2 = vari_len_acc_compute(new_labels, message, valid_len)

        acc = torch.logical_or(corr1, corr2) * 1.0

    elif language == 'final_none' or language == 'final_uniform' or language == 'final_skewed' or language == 'final_medium':
        to6_2 = [0, 5, 6, 1, 2, 3, 4, 7, 8]
        to2_6 = [0, 3, 4, 5, 6, 1, 2, 7, 8]

        to6_5 = [0, 2, 3, 4, 5, 6, 1, 7, 8]
        to5_6 = [0, 6, 1, 2, 3, 4, 5, 7, 8]

        new_labels = []
        new_l = None

        if mk_locate.size(0) != labels.size(0):
            acc = corr1 * 1.0
            return acc

        for i, j in mk_locate:
            if j == 2:
                new_l = labels[i][to2_6]
            elif j == 5:
                new_l = labels[i][to5_6]
            elif j == 6:
                if sorted_inanimate_locate[i][1] == 2:
                    new_l = labels[i][to6_2]
                elif sorted_inanimate_locate[i][1] == 3:
                    new_l = labels[i][to6_5]  
            else:
                new_l = None
            new_labels.append(new_l)
        new_labels = torch.stack(new_labels)


        corr2 = vari_len_acc_compute(new_labels, message, valid_len)

        acc = torch.logical_or(corr1, corr2) * 1.0

    return acc

# def acc_eval_compute(labels, aux_inpute, message, valid_len, pad_id):
#     # first extracts the language from aux_inpute
#     language = inv_language_dict[aux_inpute['language'][0].item()]
#     # then calls vari_len_acc_compute to compute the accuracy of the predicted message tensor
#     corr1 = vari_len_acc_compute(labels, message, valid_len)
#     # print(f'corr1:{(corr1*1.0).sum()}')

#     # print("aux_inpute", aux_inpute)
#     # aux_inpute {'language': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     #     0, 0, 0, 0, 0, 0, 0, 0]), 'mk_idx': tensor([20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
#     #     20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]), 'order': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#     #     0, 0, 0, 0, 0, 0, 0, 0])}

#     # The rest of the function is specific to each language and implements different post-processing steps to compute the final accuracy.
#     mk_idx = aux_inpute['mk_idx'][0].item()
#     # print("mk_idx", mk_idx) # 20

#     if language == 'initial_local' or language == 'initial_long':
#         acc = corr1 * 1.0        
#     elif language == 'initial':

#         to7_3 = [0, 1, 6, 7, 2, 3, 4, 5, 8]
#         to4_3 = [0, 1, 3, 4, 5, 6, 7, 2, 8]

#         to3_7 = [0, 1, 4, 5, 6, 7, 2, 3, 8]
#         to3_4 = [0, 1, 7, 2, 3, 4, 5, 6, 8]

#         # "free_mk"
#         # f2r = [0, 2, 3, 1, 4, 5]
#         # r2f = [0, 3, 1, 2, 4, 5]
#         # Name_2 mk Name_1 Verb_1 
#         # Name_1 Name_2 mk Verb_1  
 
#         # VERB_4 ANIMATE_3 ADPOSITION_1 ADJECTIVE_2 INANIMATE_1 ANIMATE_6 NONE NONE NONE	
#         # Verb_4 Animate_3 Adposition_1 Adjective_2 Inanimate_1 Animate_6 mk - VSO (7)
#         # Verb_4 Animate_6 mk Animate_3 Adposition_1 Adjective_2 Inanimate_1 - VOS (3)
#         # 7->3 [0, 1, 6, 7, 2, 3, 4, 5, 8]
#         # 3->7 [0, 1, 4, 5, 6, 7, 2, 3, 8]

#         # VERB_1 ANIMATE_2 NONE NONE NONE ANIMATE_1 ADPOSITION_3 ADJECTIVE_1 INANIMATE_1 	
#         # Verb_1 Animate_2 Animate_1 mk Adposition_3 Adjective_1 Inanimate_1 - VSO (4)
#         # Verb_1 Animate_1 mk Adposition_3 Adjective_1 Inanimate_1 Animate_2 - VOS (3)
#         # 4->3 [0, 1, 3, 4, 5, 6, 7, 2, 8]
#         # 3->4 [0, 1, 7, 2, 3, 4, 5, 6, 8]


#         # # The resulting tensor, mk_locate, will have shape (n, 1), where n is the number of occurrences of mk_idx in labels. Each row in mk_locate will be a one-dimensional tensor containing the indices of one occurrence of mk_idx in labels
#         mk_locate = torch.nonzero(torch.eq(labels, mk_idx))
        
#         # result = torch.eq(a, b) | torch.eq(a, c)
#         inanimate_locate = torch.nonzero(torch.eq(labels, 14) | torch.eq(labels, 15))
#         # print("mk_locate", mk_locate)
#         # print("inanimate_locate", inanimate_locate)

#         new_labels = []
#         # # first checks if all examples have the marker index (mk_idx) in the expected position (index 2 or 3) in the labels tensor
#         if mk_locate.size(0) != labels.size(0):
#             acc = corr1 * 1.0
#             return acc

#         for i, j in mk_locate:
#             if j == 7:
#                 new_l = labels[i][to7_3]
#             elif j == 4:
#                 new_l = labels[i][to4_3]
#             elif j == 3:
#                 # inanimate_1,2: 14,15
#                 if inanimate_locate[i][1] == 7:
#                     new_l = labels[i][to3_7]           
#                 elif inanimate_locate[i][1] == 6:
#                     new_l = labels[i][to3_4]           
#             else:
#                 new_l = None
#             new_labels.append(new_l)
#         new_labels = torch.stack(new_labels)
#         # print("labels", labels)
#         # print("new_labels", new_labels)

#         corr2 = vari_len_acc_compute(new_labels, message, valid_len)
#         # print(corr2.sum())
#         # acc is a tensor where each element represents whether the model predicted either of the two classes correctly at that position
#         acc = torch.logical_or(corr1, corr2) * 1.0

#     return acc


class MyLoss_spk:
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(
        self,
        sender_input: torch.Tensor,
        message,
        receiver_input: torch.Tensor,
        receiver_output: torch.Tensor,
        labels: torch.Tensor,
        aux_inpute: Dict[str, torch.Tensor],
        is_training: True,
        eos_id,
        pad_id,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # cut length for computing loss: cutting_length(message, receiver_output, labels)
        message_, receiver_output_ = cutting_length(message, receiver_output, labels)

        accumulate_loss = 0

        batch_size = sender_input.size(0)
        # acc test
        # count eos, not count sos
        valid_len = (labels == eos_id).nonzero()[:, -1]
        if is_training:
            corr = vari_len_acc_compute(labels, message_, valid_len)
            acc = corr * 1.0
        else:
            acc = acc_eval_compute(labels, aux_inpute, message_, valid_len, pad_id)

        for step, step_output in enumerate(receiver_output_):
            l = self.criterion(step_output.contiguous().view(
                batch_size, -1), labels[:, step + 1])
            accumulate_loss += l
        return accumulate_loss, {"acc": acc}


class MyLoss_spk_v2:
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(
        self,
        sender_input: torch.Tensor,
        message,
        receiver_input: torch.Tensor,
        receiver_output: torch.Tensor,
        labels: torch.Tensor,
        aux_inpute: Dict[str, torch.Tensor],
        is_training: True,
        eos_id,
        pad_id,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:

        # cut length for computing loss: cutting_length(message, receiver_output, labels)
        message_, receiver_output_ = cutting_length(message, receiver_output, labels)

        accumulate_loss = 0

        batch_size = sender_input.size(0)
        # acc test
        # count eos, not count sos
        valid_len = (labels == eos_id).nonzero()[:, -1]
        # if is_training:
        #     corr = vari_len_acc_compute(labels, message_, valid_len)
        #     acc = corr * 1.0
        # else:
        #     acc = acc_eval_compute(labels, aux_inpute, message_, valid_len, pad_id)

        corr = vari_len_acc_compute(labels, message_, valid_len)
        acc = corr * 1.0
        multi_acc = acc_eval_compute(labels, aux_inpute, message_, valid_len, pad_id)

        for step, step_output in enumerate(receiver_output_):
            l = self.criterion(step_output.contiguous().view(
                batch_size, -1), labels[:, step + 1])
            accumulate_loss += l
        return accumulate_loss, {"acc": acc, 'multi_acc':multi_acc}