import torch
from torch import nn
from src.models.base import BaseModel
from asteroid.masknn import norms


class DullModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.conv = nn.Conv1d(
            1,
            self.n_src,
            kernel_size=(3,),
            padding=1,
        )

    def forward(self, x):
        x = self.conv(x)
        return {"preds": x}

    def separate(self, mixture: torch.Tensor):
        with torch.no_grad():
            return self(mixture)


class SingleRNN(nn.Module):
    """Module for a RNN block.

    Inspired from https://github.com/asteroid-team/asteroid/blob/master/asteroid/masknn/recurrent.py

    Args:
        rnn_type (str): Select from ``'RNN'``, ``'LSTM'``, ``'GRU'``. Can
            also be passed in lowercase letters.
        input_size (int): Dimension of the input feature. The input should have
            shape [batch, seq_len, input_size].
        hidden_size (int): Dimension of the hidden state.
        n_layers (int, optional): Number of layers used in RNN. Default is 1.
        dropout (float, optional): Dropout ratio. Default is 0.
        bidirectional (bool, optional): Whether the RNN layers are
            bidirectional. Default is ``False``.
    """

    def __init__(
        self, rnn_type, input_size, hidden_size, n_layers=1, dropout=0, bidirectional=False
    ):
        super(SingleRNN, self).__init__()
        assert rnn_type.upper() in ["RNN", "LSTM", "GRU"]
        rnn_type = rnn_type.upper()
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn = getattr(nn, rnn_type)(
            input_size,
            hidden_size,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bool(bidirectional),
        )

    @property
    def output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(self, inp):
        """Input shape [batch, seq, feats]"""
        self.rnn.flatten_parameters()  # Enables faster multi-GPU training.
        output = inp
        rnn_output, _ = self.rnn(output)
        return rnn_output


class ReccurentBlock(nn.Module):
    """Dual-Path RNN Block as proposed in [1].

        Args:
            in_chan (int): Number of input channels.
            hid_size (int): Number of hidden neurons in the RNNs.
            norm_type (str, optional): Type of normalization to use. To choose from
                - ``'gLN'``: global Layernorm
                - ``'cLN'``: channelwise Layernorm
            bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN.
            rnn_type (str, optional): Type of RNN used. Choose from ``'RNN'``,
                ``'LSTM'`` and ``'GRU'``.
            num_layers (int, optional): Number of layers used in each RNN.
            dropout (float, optional): Dropout ratio. Must be in [0, 1].
    """

    def __init__(
            self,
            in_chan,
            hid_size,
            norm_type="gLN",
            bidirectional=True,
            rnn_type="LSTM",
            use_mulcat=False,
            num_layers=1,
            dropout=0,
    ):
        super(self).__init__()
        if use_mulcat:
            # IntraRNN block and linear projection layer (always bi-directional)
            self.intra_RNN = MulCatRNN(
                rnn_type,
                in_chan,
                hid_size,
                num_layers,
                dropout=dropout,
                bidirectional=True,
            )
            # InterRNN block and linear projection layer (uni or bi-directional)
            self.inter_RNN = MulCatRNN(
                rnn_type,
                in_chan,
                hid_size,
                num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        else:
            self.intra_RNN = SingleRNN(
                rnn_type,
                in_chan,
                hid_size,
                num_layers,
                dropout=dropout,
                bidirectional=True,
            )
            self.inter_RNN = SingleRNN(
                rnn_type,
                in_chan,
                hid_size,
                num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        self.intra_linear = nn.Linear(self.intra_RNN.output_size, in_chan)
        self.intra_norm = norms.get(norm_type)(in_chan)

        self.inter_linear = nn.Linear(self.inter_RNN.output_size, in_chan)
        self.inter_norm = norms.get(norm_type)(in_chan)

    def forward(self, x):
        """Input shape : [batch, feats, chunk_size, num_chunks]"""
        B, N, K, L = x.size()
        output = x  # for skip connection
        # Intra-chunk processing
        x = x.transpose(1, -1).reshape(B * L, K, N)
        x = self.intra_RNN(x)
        x = self.intra_linear(x)
        x = x.reshape(B, L, K, N).transpose(1, -1)
        x = self.intra_norm(x)
        output = output + x
        # Inter-chunk processing
        x = output.transpose(1, 2).transpose(2, -1).reshape(B * K, L, N)
        x = self.inter_RNN(x)
        x = self.inter_linear(x)
        x = x.reshape(B, K, L, N).transpose(1, -1).transpose(2, -1).contiguous()
        x = self.inter_norm(x)
        return output + x


class DualPathRNN(nn.Module):
    def __init__(
            self,
            in_chan,
            n_src,
            hid_size=128,
            chunk_size=100,
            n_repeats=6,
            norm_type="gLN",
            mask_act="relu",
            bidirectional=True,
            rnn_type="LSTM",
            use_mulcat=False,
            num_layers=1,
            dropout=0,
    ):
        super().__init__()
        self.in_chan = in_chan
        self.hid_size = hid_size
        self.chunk_size = chunk_size
        self.n_repeats = n_repeats
        self.n_src = n_src
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_mulcat = use_mulcat

        self.blocks = []
        for x in range(self.n_repeats):
            self.blocks += ReccurentBlock(
                in_chan=in_chan,
                hid_size=hid_size,
                norm_type=norm_type,
                bidirectional=bidirectional,
                rnn_type=rnn_type,
                use_mulcat=use_mulcat,
                num_layers=num_layers,
                dropout=dropout,
            )
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, n_feat, chunk_size, n_chunks)
        return: Tensor of shape (batch_size, n_feat, chunk_size, n_chunks)
                after n_repeats Reccurent Blocks
        """

        assert len(x.shape) == 4
        return self.blocks(x)


class SpectrogrammEncoder(BaseModel):

    def __init__(self,
                 n_src: int,
                 in_chan: int,
                 bn_chan: int,
                 chunk_size: int,
                 chunking_stride: int,
                 n_fft: int,
                 window_size: int,
                 hop_size: int,
                 windom: torch.Tensor = None
                 ):
        super().__init__(n_src=n_src)
        self.in_chan = in_chan
        self.bn_chan = bn_chan
        self.chunk_size = chunk_size
        self.chunking_stride = chunking_stride
        self.n_fft = n_fft
        self.window_size = window_size
        self.hop_size = hop_size
        self.window = windom

        if self.windom is None:
            self.windom = torch.hann_window(window_size)

        # TODO: what is num_output_channels after STFT?
        # ...

    def forward(self, x):
        """
        :torch.Tensor x: raw audio
        """

        # STFT
        # bottleneck
        # unfold (chunking)
        pass


class SpectrogrammDecoder(BaseModel):

    def __init__(self,
                 n_src: int,
                 n_chunks: int,
                 chunk_size: int,
                 hop_size: int
                 ):
        super().__init__(n_src=n_src)

    def forward(self, x):
        raise NotImplemented

