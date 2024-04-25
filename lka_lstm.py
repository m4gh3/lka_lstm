import torch
import torch.nn as nn
import torch.nn.functional as F

#this file defines the LKA_LSTM layer and LKA_LSTM_LM model

#lka feature map
def f_map(x):
    alpha = torch.linalg.vector_norm(x, dim=-1 ).unsqueeze(-1)
    x = F.normalize(x, dim=-1 )
    return torch.cat([(1-0.5**alpha)*x, torch.ones(x.shape[:-1]+(1,)).to(x.device)], dim=-1 )


def mh_lka(q, k, v ):

    #F.normalize(q, dim=-1 )
    #F.normalize(k, dim=-1 )
    q = f_map(q)
    k = f_map(k)

    kv = torch.einsum('b n d, b n e -> b n d e', k, v ).cumsum(1)
    k_ = k.cumsum(1)
    qk_ = torch.einsum('bnd,bnd->bn', q, k_ ).unsqueeze(-1)
    qkv = torch.einsum('bnd,bnde->bne', q, kv )

    return (qkv)/(qk_)



class LKA_LSTM(nn.Module):
   

    def __init__(self, in_sz, hid_sz, n_h, out_sz ):
       
        super().__init__()

        self.l0q = nn.Linear(in_sz, hid_sz*n_h )
        self.l0k = nn.Linear(in_sz, hid_sz*n_h )
        self.l0v = nn.Linear(in_sz, hid_sz*n_h )
        self.lstm = nn.LSTM(hid_sz, hid_sz, batch_first=True )
        self.l1 = nn.Linear(hid_sz*n_h, out_sz )
        self.l2 = nn.Linear(out_sz, out_sz )
        self.n_h = n_h
        self.hid_sz = hid_sz
        self.ln = nn.LayerNorm(out_sz)


    def forward(self, x ):

        n_h = self.n_h
        hid_sz = self.hid_sz
        
        b = x.shape[0]
        q = self.l0q(x).view(b, -1, n_h, hid_sz ).permute(0,2,1,3).reshape(b*n_h, -1, hid_sz )
        k = self.l0k(x).view(b, -1, n_h, hid_sz ).permute(0,2,1,3).reshape(b*n_h, -1, hid_sz )
        v = self.l0v(x).view(b, -1, n_h, hid_sz ).permute(0,2,1,3).reshape(b*n_h, -1, hid_sz )

        o = mh_lka(q, k, v )
        o_, _ = self.lstm(o)
        o = o + o_

        o = o.view(b, n_h, -1, hid_sz ).permute(0,2,1,3).reshape(b, -1, hid_sz*n_h )
        o  = self.l1(o)
        o  = self.l2(F.gelu(o))

        return self.ln(o)



class LKA_LSTM_LM(nn.Module):


    def __init__(self, vocab_sz, emb_sz, hid_sz, n_h ):

        super().__init__()

        self.emb = nn.Embedding(vocab_sz, emb_sz  )
        self.lka_lstm_stack = nn.ModuleList([LKA_LSTM(emb_sz, hid_sz, n_h, emb_sz ) for i in range(8) ])
        self.map_to_tok = nn.Linear(emb_sz, vocab_sz )


    def forward(self, x ):

        x = self.emb(x)

        for lka_lstm in self.lka_lstm_stack:
            x = lka_lstm(x) + x

        return self.map_to_tok(x)
