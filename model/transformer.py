import torch
import torch.nn as nn
from blocks import TransformerBlock, Encoder, Decoder, DecoderBlock

class Transformer(nn.Module): 
    def __init__(
            self, src_vocab_size, trg_vocab_size, 
            src_pad_idx, trg_pad_idx, embed_size=256,
            num_layers=6, forward_expansion=4, heads=8, 
            dropout=0, device='cuda', max_len=100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, embed_size, num_layers,
                               heads, device, forward_expansion, dropout, max_len)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, 
                               heads, forward_expansion, dropout, device, max_len)
        
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg): 
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
    