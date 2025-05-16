import torch
from attention import scaled_dot_product_attention, MultiHeadAttention

class Encoder: 
    def __init__(self, num_layers, d_model, num_heads, ff_dim, input_vocab_size, max_seq_len): 
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.input_vocab_size = input_vocab_size
        self.max_seq_len = max_seq_len


    def embed_input(self): 
        pass
    
    def add_positional_encoding(self): 
        pass

    def build_encoder_layer(self): 
        pass

    def forward(self): 
        pass

