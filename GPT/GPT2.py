from re import A
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math

from transformers import Conv1D


""" References
    * youtube : https://www.youtube.com/watch?v=XQSVJu8EpZ4
    * code : https://amaarora.github.io/2020/02/18/annotatedGPT2.html
    * + code : https://github.com/hugman/DL_NLP_101/tree/main/Part4_NLP_101/practice
"""

def clones(module, N):
    # Produce N identical layers
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Conv1D(nn.Module):
    def __init__(self, nx, nf):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


def gelu_new(x):
    # code : https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    return 0.5*x*(1.0 + torch.tanh(math.sqrt(2.0/math.pi)*(x+0.044715*torch.pow(x,3.0))))


class GPT2MLP(nn.Module):
    def __init__(self, d_model, nx, dropout):
        super().__init__()
        self.c_fc = Conv1D(d_model, nx)     # linear 1
        self.c_proj = Conv1D(nx, d_model)   # linear 2
        self.act = gelu_new
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # nx = dim_feedforward
        x = self.c_fc(x)    # output:  [d_model, nx]
        x = self.act(x)     # glue
        x = self.c_proj(x)  # output : [seq_len, d_model]
        return self.dropout(x)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, bias=True):
        super().__init__()    
        self.n_head = n_head
        self.d_model = d_model
        # Why use conv1D? 
        # 실제로 conv1d와 linear 썼을 때 결과물은 똑같다 (kernel_size = 1)
        # 본 hugging face에서 Conv1D를 사용. 복사하기 편하도록 linear 대신 사용함
        self.c_attn = Conv1D(d_model, d_model*3)    # linear, weight multiply part is replaced with Conv1D
        
        self.dropout = nn.Dropout(0.1)
        self.c_proj = Conv1D(d_model, d_model)      # linear, weight multiply part is replaced with Conv1D
        
        # Assume that d_v always equals d_k
        assert d_model % n_head == 0
        self.d_k = d_model//self.n_head     # d_model = 768, n_heads = 12, ---> d_k = 64

    def split_heads(self, x):
        new_shape = x.size()[:-1] + (self.n_head, self.d_k)
        x = x.view(*new_shape)
        # permute : 모든 차원들을 맞교환한다. 
        return x.permute(0, 2, 1, 3)    # [B, heads, seq_len, d_k]  
    
    def _attn(self, q, k, v, mask=None):
        # [seq_len, d_model] * [d_model, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1))   # 2개의 차원을 교환 
        scores = scores/math.sqrt(v.size(-1))   # scaling by root
        nd, ns = scores.size(-2), scores.size(-1)
        
        # masking
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
        if mask != None:
            mask = (1.0 - mask)*-1e4
            scores = scores + mask
        
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        outputs = torch.matmul(scores, v)   # [seq_len, seq_len] * [seq_len, d_model]
        return outputs, scores
        
    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2)*x.size(-1),)
        return x.view(*new_shape)
        
    def forward(self, x, mask):
        x = self.c_attn(x)  
        q, k, v = x.split(self.d_model, dim=2)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        out, scores = self._attn(q,k,v, mask)
        out = self.merge_heads(out)
        out = self.c_proj(out)
        return out, scores
    

class GPT2_TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1):
        super(GPT2_TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(d_model=d_model, n_head=n_head, bias=True)
        self.mlp = GPT2MLP(d_model=d_model, nx=dim_feedforward, dropout=dropout)
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)
        
    def forward(self, x, look_ahead_mask):
        # 1) layernorm and masked multihead
        nx = self.ln_1(x)
        a, attn_scores = self.attn(nx, mask=look_ahead_mask)
        x = x + a
        
        # 2) layernorm and MLP
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, attn_scores
        

class GPT2Decoder(nn.Module):
    """ Decoder Block of GPT2 - a stack of N layers """
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward=None):
        super(GPT2Decoder, self).__init__()
        self.num_layers = num_layers
        if dim_feedforward == None:
            dim_feedforward = 4*d_model
        
        # prepare N sub-blocks
        a_layer = GPT2_TransformerBlock(d_model=d_model, n_head=num_heads, dim_feedforward=dim_feedforward)
        self.layers = clones(a_layer, self.num_layers)
        
    def forward(self, x, look_ahead_mask=None):
        # x : [B, tar_seq_len, d_model]
        # enc_output : [B, src_seq_len, d_model]
        # look_ahead_mask
        layers_attn_scores = []
        for layer in self.layers:
            x, attn_scores = layer(x, look_ahead_mask)
            layers_attn_scores.append(attn_scores)
        
        return x, layers_attn_scores


class GPT2(nn.Module):
    """ GPT2 model """
    def __init__(self,
                 vocab_size,
                 num_layers,
                 emb_dim,
                 d_model,
                 num_heads,
                 max_seq_length,
                 ):
        super().__init__()
        self.max_seq_len = max_seq_length
        self.dropout_rate = 0.1
        self.dim_feedforward = 4*d_model    
        self.tokens = 0
        
        # input
        self.wte = nn.Embedding(vocab_size, emb_dim)            # vocab size -> emb_dim
        self.wpe = nn.Embedding(self.max_seq_len, emb_dim)      # position -> emb_dim
        self.emb_dropout = nn.Dropout(self.dropout_rate)
        self.register_buffer("position_ids", torch.arange(self.max_seq_len).expand((1,-1)))
        
        # Transformers part
        self.blocks = GPT2Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=self.dim_feedforward
        )
        self.ln_f = nn.LayerNorm(d_model)
        
        # output
        self.head = nn.Linear(emb_dim, vocab_size, bias=False)
        
    def forward(self, input_ids):
        # B = Batch size
        B, seq_len = input_ids.size()
        assert seq_len <= self.max_seq_len, "Input sequence length exceed model's maximum input length"
        
        # ----- INPUT (EMBEDDING PART)
        token_embeddings = self.wte(input_ids)      # each index maps to a (learnable) vector
        seq_length = input_ids.shape[1]
        position_ids =self.position_ids[:, :seq_length]
        position_embeddings = self.wpe(position_ids)        # each index maps to a (learnable) vector
        x = self.emb_dropout(token_embeddings + position_embeddings)
        
        # ----- TRANSFORMER part
        lookahead_mask = self.look_ahead_mask(seq_len).to(x.device)
        x, layer_attn_scores = self.blocks(x, look_ahead_mask=lookahead_mask)
        x = self.ln_f(x)    # layer norm on the final Transformer block
        
        # ----- OUTPUT part
        logits = self.head(x)
        
        return logits
    
    def look_ahead_mask(self, tgt_len:int) -> torch.FloatTensor:
        # 왼쪽 아래 삼각형이 0으로 채워짐
        mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.int), diagonal=1)
        mask = 1 - mask     # reverse
        return mask


def cp_weight(src, tar, copy_bias=True, include_eps=False):
    assert tar.weight.size() == src.weight.size(), "Not compatible parameter size"
    tar.load_state_dict( src.state_dict() )
    
    if include_eps:
        # in case of LayerNorm. 
        with torch.no_grad():
            tar.eps = src.eps  
            

def cp_gpt2_transformer_block_weights(src, tar):
    ## src: huggingface GPT2 - Transformer model 
    ## tar: my GPT2 - model - core weights

    ## layer normalization at top transformer block 
    cp_weight(src.transformer.ln_f, tar.ln_f, include_eps=True) # ln_f

    ## layer weights
    for layer_num, src_block in enumerate(src.transformer.h):
        # <<< MultiHeadAttention (Conv1D's parameters) >>>
        cp_weight(src_block.attn.c_attn,        tar.blocks.layers[layer_num].attn.c_attn) # c_attn
        cp_weight(src_block.attn.c_proj,        tar.blocks.layers[layer_num].attn.c_proj) # c_proj

        # same dropout for attention, residual, and others
        #tar.blocks.layers[layer_num].attn.dropout.load_state_dict( src_block.attn.attn_dropout )

        # <<< MLP >>
        cp_weight(src_block.mlp.c_fc,       tar.blocks.layers[layer_num].mlp.c_fc) # c_fc
        cp_weight(src_block.mlp.c_proj,     tar.blocks.layers[layer_num].mlp.c_proj) # c_proj
        #tar.blocks.layers[layer_num].mlp.dropout.load_state_dict(src_block.mlp.dropout) # dropout

        # layer normalization parameters
        cp_weight(src_block.ln_1, tar.blocks.layers[layer_num].ln_1, include_eps=True) # ln_1
        cp_weight(src_block.ln_2, tar.blocks.layers[layer_num].ln_2, include_eps=True) # ln_2

    return tar


def cli_main():
    ## GPT2 model
    from transformers import GPT2Model, GPT2LMHeadModel
    from transformers import GPT2Tokenizer
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    inputs = tokenizer("Hello, My dog is cute", return_tensors="pt")
    
    ## huggingface model
    hg_model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    ## my model
    my_model = GPT2(
        vocab_size=hg_model.config.vocab_size,
        num_layers=hg_model.config.n_layer,
        emb_dim=hg_model.config.n_embd,
        d_model=hg_model.config.n_embd,
        num_heads=hg_model.config.n_head,
        max_seq_length=hg_model.config.n_ctx,
    )
    
    ## copy hyperparameters
    
    # INPUT embedding
    ## copy embeddings from hugginface to my gpt2
    my_model.wte.load_state_dict(hg_model.transformer.wte.state_dict())
    my_model.wpe.load_state_dict(hg_model.transformer.wpe.state_dict())
    
    # OUTPUT embedding
    ## copy to output vocab
    my_model.head.load_state_dict(hg_model.lm_head.state_dict())
    
    # TRANSFORMER 
    ## Transformer block copy
    my_model = cp_gpt2_transformer_block_weights(hg_model, my_model)
    
    # 학습 mode로 on
    hg_model.eval()
    my_model.eval()
    
    with torch.no_grad():
        hg_outputs = hg_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )
        
        my_output = my_model(
            input_ids = inputs.input_ids
        )
        
        assert torch.all(torch.eq(hg_outputs.logits, my_output)), "Not same result!"
        print("Same results! -- huggingface and my code")
        
        

if __name__ == '__main__':
    cli_main()