import torch
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, mask=None):
        return input + self.module(input)

class MaskedSkipConnection(nn.Module):

    def __init__(self, module):
        super(MaskedSkipConnection, self).__init__()
        self.module = module

    def forward(self, input, mask=None):
        return input + self.module(input, None, mask)

class MaskedSequential(nn.Sequential):
    def forward(self, x, mask):
        for module in self._modules.values():
            x = module(x, mask)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)

class PositionalDecoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalDecoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x - self.pe[:,:x.size(1),:]
        return self.dropout(x)

class StMultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None,
            is_vrp=False
    ):
        super(StMultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.is_vrp = is_vrp

        self.spatial_attention = MultiHeadAttention(n_heads, input_dim, embed_dim, val_dim, key_dim)
        self.temporal_attention = MultiHeadAttention(n_heads, input_dim, embed_dim, val_dim, key_dim)

        #self.positional_encode = PositionalEncoding(input_dim, dropout=0.0)
        #self.positional_decode = PositionalDecoding(input_dim, dropout=0.0)

        self.fixed_emb = torch.nn.Linear(input_dim, input_dim)
        self.fushed = torch.nn.Linear(2*input_dim, input_dim)

    def forward(self, q, h=None, mask=None):

        if h is None:
            h = q  # compute self-attention #TODO

        # h should be (batch_size, time, graph_size, input_dim)
        batch_size, time, graph_size, input_dim = q.size()
        shape_spatial = (batch_size, time, graph_size, input_dim)
        shape_temporal = (batch_size, graph_size, time, input_dim)

        spatial = q.contiguous().view(batch_size*time, graph_size, input_dim)


        if self.is_vrp:
            temporal = q[:,:,1:,:].transpose(1, 2).contiguous().view(batch_size * (graph_size-1), time, input_dim)
            shape_temporal = (batch_size, graph_size-1, time, input_dim)
            fixed_out = self.fixed_emb(q[:,:,0,:])
        else:
            temporal = q.transpose(1,2).contiguous().view(batch_size*graph_size, time, input_dim)

        temporal_mask = mask[:,None,:,:].expand(batch_size,graph_size,graph_size,graph_size).contiguous().\
            view(batch_size*graph_size,graph_size,graph_size)
        spatial_out = self.spatial_attention(spatial, mask=temporal_mask).view(shape_spatial)
        temporal_out = self.temporal_attention(temporal).view(shape_temporal).transpose(1, 2)

        if self.is_vrp:
            temporal_out = torch.cat((fixed_out[:,:,None,:], temporal_out), 2)

        emb_cat = self.fushed(torch.cat((spatial_out, temporal_out), dim=-1))
        #emb_cat = torch.add(spatial_out, temporal_out)
        fusion = torch.sigmoid(emb_cat)#TODO: Implement fusuion

        return fusion

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, time, graph_size, input_dim)
        if len(q.size()) == 4:
            batch_size, time, graph_size, input_dim = q.size()
        # h should be (batch_size, graph_size, input_dim)
        else:
            batch_size, graph_size, input_dim = q.size()

        n_query = q.size(1)
        assert q.size(0) == batch_size
        #assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out


class MultiHeadConv2DAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            elem_dim=20,
            val_dim=None,
            key_dim=None,
    ):
        super(MultiHeadConv2DAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.elem_dim = elem_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim*elem_dim, key_dim*elem_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim*elem_dim, key_dim*elem_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim*elem_dim, val_dim*elem_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim*elem_dim, embed_dim*elem_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, time, graph_size, input_dim)
        #if len(h.size()) == 4:
        batch_size, time, graph_size, input_dim = h.size()
        # h should be (batch_size, graph_size, input_dim)
        #else:
        #    batch_size, graph_size, input_dim = h.size()

        n_query = q.size(2)
        assert q.size(0) == batch_size
        assert q.size(3) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(batch_size*time, graph_size*input_dim)
        qflat = q.contiguous().view(batch_size*time, graph_size*input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, time, -1)
        shp_q = (self.n_heads, batch_size, time, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim * self.elem_dim),
            self.W_out.view(-1, self.embed_dim * self.elem_dim)
        ).view(batch_size, n_query, self.elem_dim, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out



class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask=None):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(MaskedSequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
            st_attention=False,
            is_vrp=False
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            MaskedSkipConnection(
                StMultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                    is_vrp=is_vrp
                ) if st_attention else
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )

class StMultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(StMultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                StMultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )

class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512,
            st_attention=False,
            is_vrp=False,
    ):
        super(GraphAttentionEncoder, self).__init__()



        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        self.is_st_attention = st_attention
        self.positional_encode = PositionalEncoding(d_model=embed_dim, dropout=0.0)

        self.layers = MaskedSequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization, st_attention, is_vrp)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):

        #assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:-1], -1) if self.init_embed is not None else x

        #if self.is_st_attention:
        #    batch_size, time, graph_size, input_dim = h.size()
        #    shape_temporal = (batch_size, graph_size, time, input_dim)
        #
        #    h = h.transpose(1, 2).contiguous().view(batch_size * graph_size, time, input_dim)
        #    h = self.positional_encode(h).view(shape_temporal).transpose(1, 2).contiguous()

        h = self.layers(h, mask)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )