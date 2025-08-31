# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from typing import Iterable, Optional, Tuple
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from torchtext.vocab import build_vocab_from_iterator
from itertools import chain




def get_dataloader(
        train_dataset,
        batch_size: int = 32,
        num_workers: int = 1,
        is_distributed: bool = False,
        seed: int = 111,
        drop_last: bool = True,
        shuffle: bool = True,
) -> Iterable[
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
]:
    "Returning an iterator for tuple(sender_input, labels, receiver_input)."

    train_sampler = None
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True, seed=seed
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    return train_loader


class MyDataset(Dataset):
    """Meaning-Utterance dataset."""

    def __init__(self, csv_file, language):
        """
        :param csv_file: Path to the csv file
        """
        self.get_language_type(language)
        self.raw_samples = pd.read_csv(csv_file, sep='\t', names=['meaning', 'utterance'])
        self._init_dataset()
        self.max_meaning_len, self.max_uttr_len = self.get_max_len()

    def get_language_type(self, language):


        if 'initial_none' in language:
            self.language = 'initial_none' 
        elif 'initial_long' in language:
            self.language = 'initial_long'   
        elif 'initial_local' in language:
            self.language = 'initial_local'
        elif 'final_none' in language:  
            self.language = 'final_none'
        elif 'final_long' in language:
            self.language = 'final_long'
        elif 'final_local' in language:
            self.language = 'final_local'
        elif 'final_uniform' in language and 'long' not in language and 'local' not in language:
            self.language = 'final_uniform'
        elif 'final_skewed' in language and 'long' not in language and 'local' not in language:
            self.language = 'final_skewed'
        elif 'initial_medium' in language and 'long' not in language and 'local' not in language:
            self.language = 'initial_medium'
        elif 'initial_test' in language:
            self.language = 'initial_test'
        elif 'final_uniformlong' in language:
            self.language = 'final_uniformlong'
        elif 'final_uniformlocal' in language:
            self.language = 'final_uniformlocal'
        elif 'final_skewedlong' in language:
            self.language = 'final_skewedlong'
        elif 'final_skewedlocal' in language:
            self.language = 'final_skewedlocal'
        elif 'initial_mediumlong' in language:
            self.language = 'initial_mediumlong'
        elif 'initial_mediumlocal' in language:
            self.language = 'initial_mediumlocal'
        elif 'initial' in language and '_' not in language:
            self.language = 'initial'         
        else:
            self.language = 'None'
            print('Language type not defined')

    def __len__(self):
        return len(self.raw_samples)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        meaning = self.samples['meaning'].iloc[idx]
        meaning_tensor = torch.tensor(self.vocab_meaning(meaning))

        utterance = self.samples['utterance'].iloc[idx]
        utterance_ = utterance + ['<PAD>']*(self.max_uttr_len - len(utterance))
        utterance_tensor = torch.tensor(self.vocab_utterance(utterance_))

        language_dict = {'initial': 0, 'initial_long': 1, 'initial_local': 2, 'initial_none': 3, 'final_long': 4, 'final_local': 5, 'final_none': 6, 'final_uniform': 7, 'final_skewed': 8, 'initial_medium': 9, 'initial_test': 10, 'final_uniformlong': 11, 'final_uniformlocal': 12, 'final_skewedlong': 13, 'final_skewedlocal': 14, 'initial_mediumlong': 15, 'initial_mediumlocal': 16}
        lang_code = torch.tensor(language_dict[self.language])

        #!!! need to be modified when adding new inanimate nouns
        sample = [meaning_tensor, utterance_tensor, {}, {'language': lang_code, 'mk_idx': self.get_mk_index(), 'inanimate_idx1': self.get_inanimate_index1(), 'inanimate_idx2': self.get_inanimate_index2(), 'inanimate_idx3': self.get_inanimate_index3()}]
        # sample = [meaning_tensor, utterance_tensor, {}, {'language': lang_code}]


        return sample

    def _init_dataset(self):
        # Preprocessing
        meaning_fn = lambda x: x.split()
        utterance_fn = lambda x: ['<SOS>'] + x.split() + ['<EOS>']

        self.samples = self.raw_samples.copy()
        self.samples['meaning'] = self.samples['meaning'].apply(meaning_fn)
        self.samples['utterance'] = self.samples['utterance'].apply(utterance_fn)

        # build vocab
        counter_meaning = Counter()
        counter_utterance = Counter()
        for ind, content in self.raw_samples.iterrows():
            m, u = content
            counter_meaning.update(m.split())
            counter_utterance.update(u.split())

        self.vocab_meaning = build_vocab_from_iterator(iter([[i] for i in counter_meaning]))
        self.vocab_utterance = build_vocab_from_iterator(iter([[i] for i in counter_utterance]),
                                                         specials=['<SOS>', '<EOS>', '<PAD>'])

        print(f"{self.vocab_utterance(['<SOS>', '<EOS>', '<PAD>'])}")
        
    def get_special_index(self):
        return self.vocab_utterance(['<SOS>', '<EOS>', '<PAD>'])

    def get_vocab_size(self):
        return len(self.vocab_meaning), len(self.vocab_utterance)

    def get_max_len(self):
        max_l_meaning = max(self.samples['meaning'].apply(lambda x: len(x)))
        max_l_uttr = max(self.samples['utterance'].apply(lambda x: len(x)))
        return max_l_meaning, max_l_uttr

    def get_mk_index(self):
        return self.vocab_utterance['mk']

    def get_inanimate_index1(self):
        return self.vocab_utterance['Inanimate_1']

    def get_inanimate_index2(self):
        return self.vocab_utterance['Inanimate_2']
    
    def get_inanimate_index3(self):
        return self.vocab_utterance['Inanimate_3']
    
    # def get_inanimate_index4(self):
    #     return self.vocab_utterance['Inanimate_4']
    
    # def get_inanimate_index5(self):
    #     return self.vocab_utterance['Inanimate_5']


from typing import (
    List,
    Optional,
    Sequence,
    TypeVar,
)

# No 'default_generator' in torch/__init__.pyi
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch._C import Generator


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

UNTRACABLE_DATAFRAME_PIPES = ['batch',  # As it returns DataChunks
                              'groupby',   # As it returns DataChunks
                              '_dataframes_as_tuples',  # As it unpacks DF
                              'trace_as_dataframe',  # As it used to mark DF for tracing
                              ]



# def my_selected_split(dataset: Dataset[T], lengths: Sequence[int], n_fixed,
#                       generator: Optional[Generator] = default_generator, selected=False) -> List[Subset[T]]:
#     r"""
#     ************************
#     If selected=True: every element appears in the train set
#     ************************

#     Randomly split a dataset into non-overlapping new datasets of given lengths.
#     Optionally fix the generator for reproducible results, e.g.:

#     >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

#     Args:
#         dataset (Dataset): Dataset to be split
#         lengths (sequence): lengths of splits to be produced
#         generator (Generator): Generator used for the random permutation.
#     """
#     # Cannot verify that dataset is Sized
#     if sum(lengths) != len(dataset):
#         raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

#     # Splitting the dataset into two categories based on the assumption that the first half
#     # belongs to one category and the second half belongs to another category.
#     n_total = len(dataset)
#     n_half = n_total // 2

#     # Get the indices for both categories
#     category1_indices = list(range(0, n_half))
#     category2_indices = list(range(n_half, n_total))

#     # Shuffle the indices within each category
#     category1_indices = [i for i in randperm(n_half, generator=generator).tolist()]
#     category2_indices = [n_half + i for i in randperm(n_total - n_half, generator=generator).tolist()]
#     # print('category1_indices:', category1_indices)
#     # print('category2_indices:', category2_indices)

#     # Determine the number of samples to take from each category for the train set
#     train_len = lengths[0]
#     n_cat1_train = train_len // 2
#     n_cat2_train = train_len - n_cat1_train

#     # Combine the selected number of samples from both categories for the train set
#     train_indices = category1_indices[:n_cat1_train] + category2_indices[:n_cat2_train]
#     print("train_indice1", category1_indices[:n_cat1_train])
#     print("train_indice2", category2_indices[:n_cat2_train])
#     print("len_train_indice1", len(category1_indices[:n_cat1_train]))
#     print("len_train_indice2", len(category2_indices[:n_cat2_train]))

#     # Combine the remaining samples from both categories for the test set
#     test_len = lengths[1]
#     n_cat1_test = test_len // 2
#     n_cat2_test = test_len - n_cat1_test

#     test_indices = category1_indices[n_cat1_train:n_cat1_train + n_cat1_test] + category2_indices[n_cat2_train:n_cat2_train + n_cat2_test]

#     print("test_indice1", category1_indices[n_cat1_train:n_cat1_train + n_cat1_test])
#     print("test_indice2", category2_indices[n_cat2_train:n_cat2_train + n_cat2_test])
#     print("len_test_indice1", len(category1_indices[n_cat1_train:n_cat1_train + n_cat1_test]))
#     print("len_test_indice2", len(category2_indices[n_cat2_train:n_cat2_train + n_cat2_test]))

#     # Combine the train and test indices
#     indices = train_indices + test_indices

#     if selected:
#         # ensure every element appears in the train set
#         meaning_list = dataset.vocab_meaning.get_itos()
#         meaning_samples = dataset.samples['meaning']

#         train_samples = dataset.samples.loc[indices[:train_len]]['meaning']

#         def df_set(x, s):
#             return s.update(x)
#         s = set()
#         train_samples.apply(lambda x: df_set(x, s))
#         meaning_set = set(meaning_list)
#         if s == set(meaning_set):
#             print('all items included')
#         else:
#             print(meaning_set.difference(s))
#             print('add all items manually')

#     # Create the subsets
#     subset_indices = [indices[offset - length:offset] for offset, length in zip(_accumulate(lengths), lengths)]
#     subsets = [Subset(dataset, indices) for indices in subset_indices]

#     return subsets


def my_selected_split(dataset: Dataset[T], lengths: Sequence[int], n_fixed,
                      generator: Optional[Generator] = default_generator, selected=False) -> List[Subset[T]]:
    r"""
    ************************
    If selected=True: every element appears in the train set
    ************************

    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Check if lengths add up to the remaining dataset after fixing the first n_fixed samples
    if sum(lengths) != len(dataset) - n_fixed:
        raise ValueError("Sum of input lengths does not equal the length of the input dataset minus the fixed n_fixed samples!")

    # First n_fixed samples are fixed in the training set
    fixed_train_indices = list(range(n_fixed))

    # Remaining dataset after the first n_fixed samples
    remaining_indices = list(range(n_fixed, len(dataset)))
    n_remaining = len(remaining_indices)

    # Splitting the remaining dataset into two categories
    n_half_remaining = n_remaining // 2

    # Get the indices for both categories from the remaining dataset
    category1_indices = [n_fixed + i for i in randperm(n_half_remaining, generator=generator).tolist()]
    category2_indices = [n_fixed + n_half_remaining + i for i in randperm(n_remaining - n_half_remaining, generator=generator).tolist()]

    # Determine the number of samples to take from each category for the train set (excluding the fixed n_fixed)
    train_len = lengths[0]  # lengths[0] is the total train size minus fixed n_fixed samples
    n_cat1_train = train_len // 2
    n_cat2_train = train_len - n_cat1_train

    # Combine fixed indices with the selected indices from both categories
    train_indices = fixed_train_indices + category1_indices[:n_cat1_train] + category2_indices[:n_cat2_train]
    
    print("train_indice0", fixed_train_indices)
    print("train_indice1", category1_indices[:n_cat1_train])
    print("train_indice2", category2_indices[:n_cat2_train])
    print("len_train_indice1", len(category1_indices[:n_cat1_train]))
    print("len_train_indice2", len(category2_indices[:n_cat2_train]))

    # Combine the remaining samples from both categories for the test set
    test_len = lengths[1]
    n_cat1_test = test_len // 2
    n_cat2_test = test_len - n_cat1_test

    test_indices = category1_indices[n_cat1_train:n_cat1_train + n_cat1_test] + category2_indices[n_cat2_train:n_cat2_train + n_cat2_test]

    print("test_indice1", category1_indices[n_cat1_train:n_cat1_train + n_cat1_test])
    print("test_indice2", category2_indices[n_cat2_train:n_cat2_train + n_cat2_test])
    print("len_test_indice1", len(category1_indices[n_cat1_train:n_cat1_train + n_cat1_test]))
    print("len_test_indice2", len(category2_indices[n_cat2_train:n_cat2_train + n_cat2_test]))

    # Now we have two sets of indices: `train_indices` and `test_indices`
    # Create the subsets directly using these indices
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    if selected:
        # ensure every element appears in the train set
        meaning_list = dataset.vocab_meaning.get_itos()
        meaning_samples = dataset.samples['meaning']

        train_samples = dataset.samples.loc[train_indices]['meaning']

        def df_set(x, s):
            return s.update(x)
        s = set()
        train_samples.apply(lambda x: df_set(x, s))
        meaning_set = set(meaning_list)
        if s == set(meaning_set):
            print('all items included')
        else:
            print(meaning_set.difference(s))
            print('add all items manually')

    # Return the train and test subsets
    return [train_subset, test_subset]



def find_idx_from_df(item, samples):
    for i in range(len(samples)):
        if item in samples.iloc[i]:
            return i, samples.iloc[i]
    return -1, None

def find_idx_from_df_equal(item, samples):
    for i in range(len(samples)):
        if item == samples.iloc[i]:
            return i, samples.iloc[i]
    return -1, None

def v2_find_idx_from_df(item, samples, item_list, flgs):
    for i in range(len(samples)):
        if item in samples.iloc[i]:
            it1, it2, it3 = samples.iloc[i]
            return i, samples.iloc[i]
    return -1, None

def update_flg_for_list(items, samples, flgs):
    if len(samples) != len(flgs):
        print('ERROR FLG')
        return flgs

    for item in items:
        for i in range(len(samples)):
            if item == samples[i]:
                flgs[i] = 1
    return flgs
