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
        elif 'initial_uniform' in language and 'long' not in language and 'local' not in language:
            self.language = 'initial_uniform'
        elif 'initial_skewed' in language and 'long' not in language and 'local' not in language:
            self.language = 'initial_skewed'
        elif 'initial_medium' in language and 'long' not in language and 'local' not in language:
            self.language = 'initial_medium'
        elif 'initial_mediumlocal' in language:
            self.language = 'initial_mediumlocal'
        elif 'initial_mediumlong' in language:
            self.language = 'initial_mediumlong'
        elif 'initial_test' in language:
            self.language = 'initial_test'
        elif 'initial_uniformlong' in language:
            self.language = 'initial_uniformlong'
        elif 'initial_uniformlocal' in language:
            self.language = 'initial_uniformlocal'
        elif 'initial_skewedlong' in language:
            self.language = 'initial_skewedlong'
        elif 'initial_skewedlocal' in language:
            self.language = 'initial_skewedlocal'
        elif 'initial_mediumlong' in language:
            self.language = 'initial_mediumlong'
        elif 'initial_mediumlocal' in language:
            self.language = 'initial_mediumlocal'
        elif 'initial' in language and '_' not in language:
            self.language = 'initial'         
        elif 'final_skewed' in language and 'long' not in language and 'local' not in language:
            self.language = 'final_skewed'
        elif 'final_skewedlong' in language:
            self.language = 'final_skewedlong'
        elif 'final_skewedlocal' in language:
            self.language = 'final_skewedlocal' 
        elif 'final_uniform' in language and 'long' not in language and 'local' not in language:
            self.language = 'final_uniform'
        elif 'final_uniformlong' in language:
            self.language = 'final_uniformlong'
        elif 'final_uniformlocal' in language:
            self.language = 'final_uniformlocal'     
        elif 'final_mediumlong' in language:
            self.language = 'final_mediumlong'
        elif 'final_mediumlocal' in language:
            self.language = 'final_mediumlocal'    
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

        language_dict = {'initial': 0, 'initial_long': 1, 'initial_local': 2, 'initial_none': 3, 'final_long': 4, 'final_local': 5, 'final_none': 6, 'initial_uniform': 7, 'initial_skewed': 8, 'initial_medium': 9, 'initial_test': 10, 'initial_uniformlong': 11, 'initial_uniformlocal': 12, 'initial_skewedlong': 13, 'initial_skewedlocal': 14, 'initial_mediumlong': 15, 'initial_mediumlocal': 16, 'final_skewed': 17, 'final_skewedlong': 18, 'final_skewedlocal': 19, 'final_uniform': 20, 'final_uniformlong': 21, 'final_uniformlocal': 22, 'final_mediumlong': 23, 'final_mediumlocal': 24}
        lang_code = torch.tensor(language_dict[self.language])

        #!!! need to be modified when adding new inanimate nouns
        sample = [meaning_tensor, utterance_tensor, {}, {'language': lang_code, 'mk_idx': self.get_mk_index(), 'inanimate_idx1': self.get_inanimate_index1(), 'inanimate_idx2': self.get_inanimate_index2(), 'inanimate_idx3': self.get_inanimate_index3()}]

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
    
    def get_inanimate_index4(self):
        return self.vocab_utterance['Inanimate_4']
    
    def get_inanimate_index5(self):
        return self.vocab_utterance['Inanimate_5']


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
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    # Ensure the first n_fixed indices are in the train set
    first_n_indices = list(range(n_fixed))

    # Shuffle the remaining indices for the test set
    remaining_indices = randperm(len(dataset) - n_fixed, generator=generator).tolist()

    # Combine the first n_fixed indices with the shuffled remaining indices
    indices = list(chain(first_n_indices, remaining_indices))

    # indices = randperm(sum(lengths), generator=generator).tolist()

    if selected:
        # ensure every element appears in the train set

        meaning_list = dataset.vocab_meaning.get_itos()
        meaning_samples = dataset.samples['meaning']


        # solution 3: check current selected indices if all item appears
        train_len = lengths[0]
        train_samples = dataset.samples.loc[indices[:train_len]]['meaning']

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
            # # solution 1: more overlap
            # flgs = [0]*len(meaning_list)
            # selected_idx = []
            # selected_samples = []
            # for i in range(len(meaning_list)):
            #     if flgs[i] == 0:
            #         idx, samp = find_idx_from_df(meaning_list[i], meaning_samples)
            #         selected_idx.append(idx)
            #         selected_samples.append(samp)
            #         update_flg_for_list(samp, meaning_list, flgs)


            #Solution 2: less overlap

            # verb = [i for i in meaning_list if 'VERB_' in i]
            # random.shuffle(verb)
            # verb_ = set(verb)
            # # modified !!!!!!!!!!!!!
            # noun = [i for i in meaning_list if 'ANIMATE_' in i and 'INANIMATE_' not in i]
            # random.shuffle(noun)
            # noun_ = set(noun)

            # adposition = [i for i in meaning_list if 'ADPOSITION_' in i]
            # # random.shuffle(adposition)
            # # adposition_ = set(adposition)

            # adjective = [i for i in meaning_list if 'ADJECTIVE_' in i]
            # # random.shuffle(adjective)
            # # adjective_ = set(adjective)

            # inanimate = [i for i in meaning_list if 'INANIMATE_' in i]  
            # # random.shuffle(inanimate)
            # # inanimate_ = set(inanimate)

            # sample_select = []
            # for i in range(min(len(noun)//2, len(verb))):
            #     sample_select.append([noun[2*i], verb[i],noun[2*i+1]])
            #     noun_.discard(noun[2*i])
            #     noun_.discard(noun[2*i+1])
            #     verb_.discard(verb[i])

            # if len(verb_) > 0:
            #     for v in verb_:
            #         if len(noun_) > 0:
            #             n1 = noun_.pop()
            #             tmp_noun = noun.copy()
            #             tmp_noun.pop(tmp_noun.index(n1))
            #             n2 = tmp_noun[random.randint(0, len(tmp_noun))]
            #             sample_select.append([v, n1, "NONE1", "NONE2", "NONE3", n2, adp, adj, ina])
            #         else:
            #             tmp_noun = noun.copy()
            #             n1 = tmp_noun[random.randint(0, len(tmp_noun))]
            #             tmp_noun.pop(tmp_noun.index(n1))
            #             n2 = tmp_noun[random.randint(0, len(tmp_noun))]
            #             sample_select.append([n1, v, n2])
            # else:
            #     for n in noun_:
            #         noun_.discard(n)
            #         v = verb[random.randint(0, len(verb))]
            #         if len(noun_) > 0:
            #             n2 = noun_.pop()
            #         else:
            #             tmp_noun = noun.copy()
            #             tmp_noun.pop(tmp_noun.index(n))
            #             n2 = tmp_noun[random.randint(0, len(tmp_noun))]
            #         sample_select.append([n, v, n2])

            # selected_idx = []
            # for s in sample_select:
            #     idx, samp = find_idx_from_df_equal(s, meaning_samples)
            #     selected_idx.append(idx)

            # for idx in selected_idx:
            #     indices.insert(0, indices.pop(indices.index(idx)))

    # return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

    subset_indices = [indices[offset - length:offset] for offset, length in zip(_accumulate(lengths), lengths)]

    # Create subsets
    subsets = [Subset(dataset, indices) for indices in subset_indices]

    return subsets


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
