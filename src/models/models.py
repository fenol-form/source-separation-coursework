import torch
from torch import nn
from torch.nn.functional import fold

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


class DualPathRNN(BaseModel):
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
        super().__init__(n_src)
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
            self.blocks += [ReccurentBlock(
                in_chan=in_chan,
                hid_size=hid_size,
                norm_type=norm_type,
                bidirectional=bidirectional,
                rnn_type=rnn_type,
                use_mulcat=use_mulcat,
                num_layers=num_layers,
                dropout=dropout,
            )]
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, n_feat, chunk_size, n_chunks)
        return: Tensor of shape (batch_size, n_feat, chunk_size, n_chunks)
                after n_repeats Reccurent Blocks
        """

        assert len(x.shape) == 4
        return self.blocks(x)


class WavEncoder(BaseModel):
    def __init__(
        self,
        n_src,
        in_chan,
        bn_chan,
        hop_size,
        chunk_size,
    ):
        super().__init__(n_src)
        self.in_chan = in_chan
        self.bn_chan = bn_chan
        self.hop_size = hop_size
        self.chunk_size = chunk_size

        self.encoder_conv = nn.Conv1d(in_chan, bn_chan, kernel_size=1)
        self.chunker = nn.Unfold(
            kernel_size=(chunk_size, 1),
            padding=(chunk_size, 0),
            stride=(hop_size, 1),
        )

    def forward(self, x):
        """
        :param x: Tensor of shape (batch_size, in_channels, time)
        :return: Tensor of shape (batch_size, bn_channels, chunk_size, num_chunks)
        """

        batch, n_filters, n_frames = x.size()
        output = self.encoder_conv(x)  # [batch, bn_chan, n_frames]
        output = self.chunker(output.unsqueeze(-1))
        n_chunks = output.shape[-1]
        output = output.reshape(batch, self.bn_chan, self.chunk_size, n_chunks)
        return output


class WavDecoder(BaseModel):
    def __init__(
        self,
        n_src,
        bn_chan,
        out_chan,
        chunk_size,
        hop_size,
    ):
        super().__init__(n_src)

        self.bn_chan = bn_chan
        self.out_chan = out_chan
        self.hop_size = hop_size
        self.chunk_size = chunk_size

        self.first_out = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(bn_chan, n_src * bn_chan, 1)
        )
        self.net_out = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Tanh())
        self.net_gate = nn.Sequential(nn.Conv1d(bn_chan, bn_chan, 1), nn.Sigmoid())
        self.mask_net = nn.Conv1d(bn_chan, out_chan, 1, bias=False)
        self.output_act = nn.ReLU()

    def forward(self, x, n_frames):
        """
        :param x: output of Dual Path RNN of shape (batch, bn_chan, chunk_size, n_chunks)
        :param n_frames: initial len of audio
        :return: Tensor of shape (batch, n_src, out_chan, n_frames)
        """
        batch, _, chunk_size, n_chunks = x.size()
        output = self.first_out(x)
        output = output.reshape(batch * self.n_src, self.bn_chan, self.chunk_size, n_chunks)

        # Overlap and add:
        # [batch, out_chan, chunk_size, n_chunks] -> [batch, out_chan, n_frames]
        to_unfold = self.bn_chan * self.chunk_size
        output = fold(
            output.reshape(batch * self.n_src, to_unfold, n_chunks),
            (n_frames, 1),
            kernel_size=(self.chunk_size, 1),
            padding=(self.chunk_size, 0),
            stride=(self.hop_size, 1),
        )

        # Apply gating
        output = output.reshape(batch * self.n_src, self.bn_chan, -1)
        output = self.net_out(output) * self.net_gate(output)
        # Compute mask
        score = self.mask_net(output)
        est_mask = self.output_act(score)
        est_mask = est_mask.view(batch, self.n_src, self.out_chan, n_frames)
        return est_mask


class DPRNN(BaseModel):
    def __init__(
        self,
        n_src,
        in_chan,
        bn_chan=128,
        out_chan=1,
        hid_size=128,
        chunk_size=100,
        hop_size=None,
        n_repeats=6,
        norm_type="gLN",
        mask_act="relu",
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

        self.encoder = WavEncoder(
            n_src, in_chan, bn_chan, hop_size, chunk_size
        )

        self.decoder = WavDecoder(
            n_src, bn_chan, out_chan, chunk_size, hop_size
        )

        self.net = DualPathRNN(
            bn_chan, n_src, hid_size, chunk_size,
            n_repeats, norm_type, mask_act,
            bidirectional, rnn_type, use_mulcat,
            num_layers, dropout
        )

    def forward(self, x):
        batch, in_chan, n_frames = x.size()
        output = self.encoder(x)
        output = self.net(output)
        output = self.decoder(output, n_frames)

        assert output.shape == (batch, self.n_src, self.out_chan, n_frames)
        output = output.squeeze(2)
        assert output.shape == (batch, self.n_src, n_frames)

        return {
            "preds": output
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

