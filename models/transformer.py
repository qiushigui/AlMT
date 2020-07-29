"""Transformer Model"""

import math
import sys
import numpy as np
import mindspore as ms
from mindspore import Tensor

class TransformerModel(ms.cell):
    """
    Args: (TransformerEncoder, TransformerDecoder)
    """
    def __init__(self, args, encoder, decoder):
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
    
    @staticmethod
    def add_args(parser):
        # TODO

    @classmethod
    def build_model(cls, args):
        """build a model"""

        encoder = TransformerEncoder(args)
        decoder = TransformerDecoder(args)

        return cls(args, encoder. decoder)

    def construct(self, src_tokens, prev_output_tokens):
        """Forward"""
        encoder_out = self.encoder(src_tokens)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out

