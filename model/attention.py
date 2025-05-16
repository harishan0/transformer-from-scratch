import torch

def scaled_dot_product_attention(Q, K, V, mask): 
    # output[i,j] represents strenght of relationship between word i and j in sequence
    # Mask is used to nullify words where j > i, so we aren't allowing future words to impact previous/current words
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device)) # seq_len x seq_len
    if mask is not None: 
        scores = scores.masked_fill(mask == 0, float('-1e20'))

    attn = torch.nn.functional.softmax(scores, dim=-1)
    # Each row is a vector embedding of word i's relationship w rest of sentence
    output = torch.matmul(attn, V)
    return output, attn

class MultiHeadAttention(torch.nn.Module): 
    def __init__(self, d_model, num_heads): 
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = torch.nn.Linear(d_model, d_model)
        self.W_K = torch.nn.Linear(d_model, d_model)
        self.W_V = torch.nn.Linear(d_model, d_model)
        self.W_O = torch.nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None): 
        batch_size = Q.size(0)

        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        context, attn = scaled_dot_product_attention(Q, K, V, mask)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        output = self.W_O(context)
        return output, attn
