import torch.nn as nn

from .baseRNN_embed import BaseRNN

import torch
from torch.nn.modules.dropout import _DropoutNd


class WordDropout_v2(_DropoutNd):
    """
    During training, replaces specific elements of the input tensor with
    `dropout_constant` if they match the given IDs.
    Input is expected to be a 2D tensor of indices representing tokenized sentences.
    During evaluation, this module is a no-op, returning `input`.

    Args:
        word_dropout_p (float): probability of an element to be replaced with `dropout_constant`. Default: 0.1.
        dropout_constant (int): Value to replace dropped out elements with.
        mask_ids (list): List of token IDs to be masked.

    Shape:
        - Input: `(N, T)`. `N` is the batch dimension and `T` is the number of indices per sample.
        - Output: `(N, T)`. Output is of the same shape as input

    Examples:
        >>> bs = 32
        >>> sent_len = 512  # Max len of padded sentences
        >>> V = 10000  # Vocab size
        >>> mask_ids = [100, 101, 102]
        >>> m = WordDropout(p=0.2, dropout_constant=0, mask_ids=mask_ids)
        >>> input = torch.randint(0, V, (bs, sent_len))
        >>> output = m(input)
    """
    def __init__(self, word_dropout_p, dropout_constant, mask_ids):
        super(WordDropout, self).__init__()
        self.word_dropout_p = word_dropout_p
        self.dropout_constant = dropout_constant
        self.mask_ids = mask_ids

    def forward(self, input):
        if not self.training or not self.word_dropout_p:
            return input

        mask = torch.empty_like(input).bernoulli_(self.word_dropout_p).bool()
        specific_mask = torch.isin(input, torch.tensor(self.mask_ids)).bool()
        final_mask = mask & specific_mask

        input = torch.where(final_mask, torch.full_like(input, self.dropout_constant), input)

        return input

        
class WordDropout(_DropoutNd):
    # derived from https://gist.github.com/JohnGiorgi/c030de1dd8cb84ad0970d1cc87e2ed86 
    """During training, randomly replaces some of the elements of the input tensor with
    `dropout_constant` with probability `p` using samples from a Bernoulli distriution. Each channel
    will be zerored out independently on every forward call.
    Input is expected to be a 2D tensor of indices representing tokenized sentences.
    During evaluation, this module is a no-op, returning `input`.
    Args:
        p (float): probability of an element to be replaced with `dropout_constant`. Default: 0.1.
        dropout_constant (int): Value to replace dropped out elements with.
    Shape:
        - Input: `(N, T)`. `N` is the batch dimension and `T` is the number of indices per sample.
        - Output: `(N, T)`. Output is of the same shape as input
    Examples:
        >>> bs = 32
        >>> sent_len = 512  # Max len of padded sentences
        >>> V = 10000  # Vocab size
        >>> m = nn.WordDropout(p=0.2)
        >>> input = torch.randint(0, V, (bs, sent_len))
        >>> output = m(input)
    """
    def __init__(self, word_dropout_p, dropout_constant):
        super(WordDropout, self).__init__(word_dropout_p)
        self.dropout_constant = dropout_constant

    def forward(self, input):
        # if not self.training or not self.p:
        #     return input
        #!!! both training and inference
        if not self.p:
            return input

        keep = torch.empty_like(input).bernoulli_(1 - self.p).bool()
        input = torch.where(keep, input, torch.empty_like(input).fill_(self.dropout_constant))

        return input
    

class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    def __init__(self, vocab_size, max_len, emb_size, hidden_size, word_dropout_p, dropout_constant,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False,
                 embedding=None, update_embedding=True):
        super(EncoderRNN, self).__init__(vocab_size, max_len, emb_size, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, emb_size)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(emb_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        self.word_drop = WordDropout(word_dropout_p, dropout_constant)
        self.dropout_constant = dropout_constant

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """

        drop = self.word_drop(input_var)
        embedded = self.embedding(drop)

        embedded = self.input_dropout(embedded)

        # freeze the embeddings of the tokens replaced by the dropout_constant value?
        mask = drop == self.dropout_constant
        embedded[mask] = embedded[mask].detach()

        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden

