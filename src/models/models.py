import sys

import asteroid.masknn.recurrent
import torch
from torch import nn
from torch.nn.functional import fold, unfold

from src.models.base import BaseModel, BaseMaskingNetwork
from asteroid.masknn import norms

import asteroid.masknn.recurrent

from torch.profiler import profile, record_function, ProfilerActivity


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
        self,
        rnn_type,
        input_size,
        hidden_size,
        n_layers=1,
        dropout=0,
        bidirectional=False
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
        super().__init__()
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


class MaskerDPRNN(BaseModel):
    def __init__(
        self,
        n_src,
        in_chan,
        bn_chan=128,
        out_chan=1,
        hid_size=128,
        chunk_size=100,
        hop_size=None,
        n_repeats=4,
        norm_type="gLN",
        mask_act="sigmoid",
        bidirectional=True,
        rnn_type="LSTM",
        use_mulcat=False,
        num_layers=1,
        dropout=0,
    ):
        super().__init__(n_src)
        self.in_chan = in_chan
        out_chan = out_chan if out_chan is not None else in_chan
        self.out_chan = out_chan
        self.bn_chan = bn_chan
        self.hid_size = hid_size
        self.chunk_size = chunk_size
        hop_size = hop_size if hop_size is not None else chunk_size // 2
        self.hop_size = hop_size
        self.n_repeats = n_repeats
        self.n_src = n_src
        self.norm_type = norm_type
        self.mask_act = mask_act
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_mulcat = use_mulcat

        layer_norm = norms.get(norm_type)(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        self.net = []
        for x in range(self.n_repeats):
            self.net += [ReccurentBlock(
                in_chan=bn_chan,
                hid_size=hid_size,
                norm_type=norm_type,
                bidirectional=bidirectional,
                rnn_type=rnn_type,
                use_mulcat=use_mulcat,
                num_layers=num_layers,
                dropout=dropout,
            )]
        self.net = nn.Sequential(*self.net)

        self.first_out = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(bn_chan, n_src * bn_chan, 1)
        )
        self.net_out = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Sigmoid())
        self.mask_net = nn.Conv1d(bn_chan, out_chan, 1, bias=False)

        # TODO: adapt for other arg's values
        if mask_act == "sigmoid":
            self.output_act = nn.Sigmoid()
        else:
            raise ValueError

    def forward(self, x):
        # bottleneck conv and chunking
        batch, in_chan, n_frames = x.size()
        x = self.bottleneck(x)  # [batch, bn_chan, n_frames]
        x = unfold(
            x.unsqueeze(-1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )
        n_chunks = x.shape[-1]
        x = x.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)

        # Dual Path RNN
        assert len(x.shape) == 4
        x = self.net(x)

        # Masking
        x = self.first_out(x)
        x = x.reshape(batch * self.n_src, self.bn_chan, self.chunk_size, n_chunks)

        # Reverse chunking
        # [batch, out_chan, chunk_size, n_chunks] -> [batch, out_chan, n_frames]
        to_unfold = self.bn_chan * self.chunk_size
        x = fold(
            x.reshape(batch * self.n_src, to_unfold, n_chunks),
            (n_frames, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        # Apply gating
        x = x.reshape(batch * self.n_src, self.bn_chan, -1)
        x = self.net_out(x) * self.net_gate(x)

        # Compute mask
        score = self.mask_net(x)
        est_mask = self.output_act(score)
        est_mask = est_mask.view(batch, self.n_src, self.out_chan, n_frames)

        assert est_mask.shape == (batch, self.n_src, self.out_chan, n_frames)

        return est_mask


class WavEncoder(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_filters: int = 64,    # default values were taken from https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/dprnn_tasnet.py#L69
        kernel_size: int = 16,
        stride: int = 8,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_chan,
            n_filters,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        for p in self.parameters():
            nn.init.xavier_normal_(p)

    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor of shape (batch, 1, time)
        :return:  Tensor of shape (batch, n_filters, n_frames)
                  n_frames might not be equal to time
        """
        return self.conv(x)


class WavDecoder(nn.Module):
    def __init__(
        self,
        n_filters: int = 64,
        out_chan: int = 1,
        kernel_size: int = 16,
        stride: int = 8,
        padding: int = 0,
        output_padding: int = 0,
    ):
        super().__init__()
        self.transpose_conv = nn.ConvTranspose1d(
            in_channels=n_filters,
            out_channels=out_chan,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        )
        for p in self.parameters():
            nn.init.xavier_normal_(p)

    def forward(self, x: torch.Tensor):
        """
        :param x: masked time-fequency representation of model's output
            of shape (batch, n_src, n_filters, n_frames)
            or (n_src, n_filters, n_frames)
        :return: decoded signals of shape (batch, n_src, time)
        """
        if x.ndim == 3:
            return self.transpose_conv(x)
        elif x.ndim == 4:
            view_as = (-1,) + x.shape[-2:]
            out = self.transpose_conv(x.reshape(view_as))
            view_as = x.shape[:-2] + (-1,)
            return out.view(view_as)


class DPRNN(BaseMaskingNetwork):
    def __init__(
            self,
            n_src,
            in_chan,

            # encoder, decoder params:
            n_filters=64,
            kernel_size=16,
            stride=8,
            encoder_type=WavEncoder,
            decoder_type=WavDecoder,

            bn_chan=128,
            out_chan=1,

            # chunker params:
            hid_size=128,
            chunk_size=100,
            hop_size=None,

            # RNN params:
            n_repeats=6,
            norm_type="gLN",
            mask_act="sigmoid",
            bidirectional=True,
            rnn_type="LSTM",
            use_mulcat=False,
            num_layers=1,
            dropout=0,
    ):
        super(BaseMaskingNetwork, self).__init__(n_src)

        self.encoder = encoder_type(
            in_chan=in_chan,
            n_filters=n_filters,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.decoder = decoder_type(
            n_filters=n_filters,
            out_chan=out_chan,
            kernel_size=kernel_size,
            stride=stride,
        )

        # self.masker = MaskerDPRNN(
        #     n_src, n_filters, bn_chan,
        #     out_chan, hid_size, chunk_size, hop_size,
        #     n_repeats, norm_type, mask_act, bidirectional,
        #     rnn_type, use_mulcat, num_layers, dropout
        # )
        self.masker = asteroid.masknn.recurrent.DPRNN(
            n_src=n_src,
            in_chan=n_filters,
            bn_chan=bn_chan,
            out_chan=out_chan,
            hid_size=hid_size,
            chunk_size=chunk_size,
            hop_size=hop_size,
            n_repeats=n_repeats,
            norm_type=norm_type,
            mask_act=mask_act,
            bidirectional=bidirectional,
            rnn_type=rnn_type,
            use_mulcat=use_mulcat,
            num_layers=num_layers,
            dropout=dropout,
        )

        super().__init__(
            n_src=n_src,
            encoder=self.encoder,
            decoder=self.decoder,
            masker=self.masker,
        )

    def forward(self, x):
        out = super().forward(x)
        return {
            "preds": out
        }


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
