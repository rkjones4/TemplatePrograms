import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# penalties for violating different constraints
MIN_LL_THRESH = -1000
MIN_LL_PEN = -10000
MASK_LL_PEN = -100000.

def load_inf_net(domain):
    from inf_prob_model import ProbProgInfNet
    return ProbProgInfNet(domain)
        
def load_vis_encoder(domain):

    if domain.name == 'omni':
        return load_omni_vis_encoder(domain)

    elif domain.name == 'shape':
        if domain.args.vin_type == 'prim':
            net =  PrimEncoder(
                max_prim_encs = domain.args.max_prim_enc,
                max_num_prims = domain.executor.MAX_PRIMS,
                prim_num_params = 6,
                od = domain.args.hidden_dim,
                pd = 16,
                num_layers = 4,
                num_heads = 8,
                dropout = domain.args.dropout,
            )
        elif domain.args.vin_type == 'voxel':
            net = V3DCNN(
                inp_dim = domain.executor.VOXEL_DIM,
                max_prim_encs = domain.args.max_prim_enc,                
                out_dim = domain.args.hidden_dim,                
                drop= domain.args.dropout
            )
        else:
            assert False, f'bad vin type {domain.args.vin_type}'
        net.device = domain.device
        return net

    elif domain.name == 'layout':
    
        inp_shape = domain.executor.get_input_shape()
        
        assert len(inp_shape) == 3
        
        return V2DCNN(inp_shape[2], domain.args.hidden_dim, domain.args.dropout)

    else:
        assert False
    
def load_omni_vis_encoder(domain):
    inp_shape = domain.executor.get_input_shape()

    assert len(inp_shape) == 3

    enc = OV2DCNN(inp_shape[2], domain.args.hidden_dim, domain.args.dropout)

    return enc
    
def parse_inds(
    dpi, batch, nvi, beams
):

    batch_ind = dpi // (nvi * beams)
    res = dpi - (batch_ind * nvi * beams)
    struct_ind = res // nvi
    res = res - (struct_ind * nvi)
    vis_ind = res % nvi
    
    return batch_ind, struct_ind, vis_ind


class TDecNet(nn.Module):
    def __init__(
        self,
        domain,
        num_venc_tokens,
        max_seq_len
    ):
        super(TDecNet, self).__init__()
        
        self.domain = domain
        args = domain.args

        self.ex = domain.executor
        self.beams = args.beams
        self.device = domain.device

        # max number of tokens from sequence
        self.ms = max_seq_len
        self.mp = num_venc_tokens
        
        self.nl = args.num_layers
        self.nh = args.num_heads
                
        self.bs = args.batch_size
        self.dropout = args.dropout
        
        # language number of tokens
        self.nt = domain.executor.TLang.get_num_tokens()
        self.hd = args.hidden_dim 
        
        self.token_enc_net = nn.Embedding(self.nt, self.hd)        
        self.token_head = SDMLP(self.hd, self.nt, self.dropout)
        
        self.pos_enc = nn.Embedding(self.ms+self.mp, self.hd)
        self.pos_arange = torch.arange(self.ms+self.mp).unsqueeze(0)
        
        self.attn_mask = self.generate_attn_mask()

        self.attn_layers = nn.ModuleList(
            [AttnLayer(self.nh, self.hd, self.dropout) for _ in range(self.nl)]
        )        
        
    def generate_attn_mask(self):
        return _generate_attn_mask(self)

    def generate_key_mask(self, num):
        return _generate_key_mask(self, num)

    # main training function, takes in codes from encoder + sequence
    def infer_prog(self, codes, seq):        
        return _infer_prog(self, codes, seq)

    # used during eval, a bit faster
    def fast_infer_prog(self, codes, seq, drange, bsinds):        
        exp_bpreds = _fast_infer_prog(
            self,
            codes,
            seq[:, :(bsinds.max() + 1)]
        )
        bpreds = exp_bpreds[drange, bsinds]
        return bpreds

class DMLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim, DP):
        super(DMLP, self).__init__()
        
        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)
        self.d1 = nn.Dropout(p=DP)
        self.d2 = nn.Dropout(p=DP)
                
    def forward(self, x):
        x = self.d1(F.relu(self.l1(x)))
        x = self.d2(F.relu(self.l2(x)))
        return self.l3(x)

class MLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim):
        super(MLP, self).__init__()
        
        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)
                
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

    
# small dropout MLP
class SDMLP(nn.Module):
    def __init__(self, ind, odim, DP):
        super(SDMLP, self).__init__()
        
        self.l1 = nn.Linear(ind, odim)
        self.l2 = nn.Linear(odim, odim)
        self.d1 = nn.Dropout(p=DP)
                
    def forward(self, x):
        x = self.d1(F.leaky_relu(self.l1(x), 0.2))
        return self.l2(x)


class PrimEncoder(nn.Module):
    def __init__(
        self, max_prim_encs, max_num_prims, prim_num_params, od, pd, num_layers, num_heads, dropout
    ):
        super(PrimEncoder, self).__init__()
        self.mpe = max_prim_encs 
        self.mnp = max_num_prims
        self.pnp = prim_num_params
        self.hd = od
        self.pd = pd

        self.mp = self.mpe + (self.mnp * self.pnp)
        self.ms = 0

        self.nl = num_layers
        self.nh = num_heads

        self.dropout = dropout

        self.attn_mask = self.generate_attn_mask()
        
        self.attn_layers = nn.ModuleList(
            [AttnLayer(self.nh, self.hd, self.dropout) for _ in range(self.nl)]
        )

        self.param_enc_net = SDMLP(1,self.pd, 0.)
        self.attr_emb = nn.Embedding(self.pnp, self.pd)
        self.prim_enc_net = SDMLP(self.pd * self.pnp, self.pd, 0.)

        self.lift_enc_net = SDMLP(self.pd, self.hd, self.dropout)

        self.enc_tokens_emb = nn.Embedding(self.mpe, self.hd)

        self.attr_arange = torch.arange(self.pnp).view(1,1,-1).repeat(1,self.mnp,1)
        self.emb_arange = torch.arange(self.mpe)
                
    def generate_attn_mask(self):
        return _generate_encoder_attn_mask(self)

    def generate_key_mask(self, num):
        return _generate_encoder_key_mask(self, num)

    def forward(self, x):
        if len(x.shape) == 4:
            fx = x.view(-1,x.shape[2],x.shape[3])
            return self._forward(fx).contiguous()
        else:
            return self._forward(x)
                            
    def _forward(self, x):
        assert len(x.shape) == 3

        mask = (x.abs().sum(dim=-1) > 0.01).float()
        penc_exp = self.param_enc_net(x.unsqueeze(-1))

        penc_flat = penc_exp.view(penc_exp.shape[0],-1,penc_exp.shape[-1])
        attr_enc = self.attr_emb(self.attr_arange.to(self.device)) 

        attr_flat = attr_enc.view(1,-1,attr_enc.shape[-1])
        
        prim_emb_exp = self.prim_enc_net(penc_exp.view(penc_exp.shape[0],penc_exp.shape[1],-1))
        
        prim_emb_flat = prim_emb_exp.unsqueeze(2).repeat(1,1,self.pnp,1).view(
            penc_exp.shape[0],-1,penc_exp.shape[-1]
        )
        
        comb_emb = penc_flat + attr_flat + prim_emb_flat
        
        lemb = self.lift_enc_net(comb_emb)

        _enc_tokens = self.enc_tokens_emb(self.emb_arange.to(self.device))

        enc_tokens = _enc_tokens.unsqueeze(0).repeat(lemb.shape[0],1,1)

        out = torch.cat((enc_tokens, lemb),dim=1)

        attn_mask = self.attn_mask.to(self.device)

        seq_lens = [int(l) * self.pnp for l in mask.sum(dim=1).tolist()]

        key_mask = self.generate_key_mask(seq_lens).to(self.device)
        
        for attn_layer in self.attn_layers:        
            out = attn_layer(out, attn_mask, key_mask)
            
        emb_out = out[:,:self.mpe,:]

        return emb_out

        
# 2D pixel CNN encoder
class V2DCNN(nn.Module):
    def __init__(self, inp_dim, code_size, drop):

        super(V2DCNN, self).__init__()

        self.inp_dim = inp_dim
        
        self.conv1 = nn.Conv2d(
            in_channels=self.inp_dim, out_channels=32, kernel_size=3, stride=(1, 1), padding=(1, 1)
        )
        self.b1 = nn.BatchNorm2d(num_features=32)
        
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=(1, 1), padding=(1, 1)
        )        
        self.b2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=(1, 1), padding=(1, 1)
        )                                                  
        self.b3 = nn.BatchNorm2d(num_features=128)
        
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, stride=(1, 1), padding=(1, 1)
        )                                                   
        self.b4 = nn.BatchNorm2d(num_features=256)

        self._encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(drop),
            self.b1,
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(drop),
            self.b2,
            self.conv3,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(drop),
            self.b3,
            self.conv4,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(drop),
            self.b4,
        )
        
        self.ll = DMLP(256, 256, 256, code_size, drop)
                        
    def forward(self, x):
        x1 = x.view(-1, self.inp_dim, 64, 64)
        x2 = self._encoder(x1)

        x2 = x2.view(-1, 256, 16)            
        x2 = x2.transpose(1, 2)
                        
        return self.ll(x2)


# 2D pixel CNN encoder
class OV2DCNN(nn.Module):
    def __init__(self, inp_dim, code_size, drop):

        super(OV2DCNN, self).__init__()

        self.inp_dim = inp_dim
        self.code_size = code_size

        self.conv1 = nn.Conv2d(
            in_channels=self.inp_dim, out_channels=32, kernel_size=3, stride=(1, 1), padding=(1, 1)
        )
        self.b1 = nn.BatchNorm2d(num_features=32)
        
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=(1, 1), padding=(1, 1)
        )        
        self.b2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=(1, 1), padding=(2, 2)
        )                                                  
        self.b3 = nn.BatchNorm2d(num_features=128)
        
        self._encoder = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(drop),
            self.b1,
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(drop),
            self.b2,
            self.conv3,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(drop),
            self.b3,
        )
        
        self.ll = DMLP(128, 128, code_size, code_size, drop)
            
            
    def forward(self, x):
        x1 = x.view(-1, self.inp_dim, 28, 28)
        x2 = self._encoder(x1)
        
        x2 = x2.view(-1, 128, 16)            
        x2 = x2.transpose(1, 2)
        
        o = self.ll(x2)

        return o

######## TRANSFORMER

class AttnLayer(nn.Module):
    def __init__(self, nh, hd, dropout):
        super(AttnLayer, self).__init__()
        self.nh = nh
        self.hd = hd

        self.self_attn = torch.nn.MultiheadAttention(self.hd, self.nh)

        self.l1 = nn.Linear(hd, hd)
        self.l2 = nn.Linear(hd, hd)

        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)
        self.d3 = nn.Dropout(dropout)        

        self.n1 = nn.LayerNorm(hd)
        self.n2 = nn.LayerNorm(hd)
                
    def forward(self, _src, attn_mask, key_padding_mask):
        
        src = _src.transpose(0, 1)
            
        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=attn_mask,
            key_padding_mask = key_padding_mask
        )[0]
        
        
        src = src + self.d1(src2)
        src = self.n1(src)
        src2 = self.l2(self.d2(F.leaky_relu(self.l1(src), .2)))
        src = src + self.d2(src2)
        src = self.n2(src)

        return src.transpose(0, 1)

    
# generate attention mask for transformer auto-regressive training
# first mp spaces have fully connected attention, as they are the priming sequence of visual encoding
def _generate_attn_mask(net):
    sz = net.mp + net.ms
    mask = (torch.triu(torch.ones(sz, sz)) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).T
    mask[:net.mp, :net.mp] = 0.
    return mask

def _generate_encoder_attn_mask(net):
    sz = net.mp + net.ms
    mask = torch.zeros(sz, sz).float()
    return mask

# generate key mask for transformer auto-regressive training
def _generate_key_mask(net, num):
    sz = net.mp + net.ms
    mask = torch.zeros(num, sz).bool()
    return mask

def _generate_encoder_key_mask(net, seq_lens):
    sz = net.mp + net.ms
    mask = torch.zeros(len(seq_lens), sz)
    for i, j in enumerate(seq_lens):
        mask[i,j+1:] = 1.0
        
    return mask.bool()
    
# main forward process of transformer, encode tokens, add PE, run through attention, predict tokens with MLP
def _infer_prog(net, codes, seq, seq_lens=None):
    token_encs = net.token_enc_net(seq).view(-1, net.ms, net.hd)
                                            
    out = torch.cat((codes.view(codes.shape[0], net.mp, net.hd), token_encs), dim = 1)        
    out += net.pos_enc(net.pos_arange.repeat(codes.shape[0], 1).to(net.device))
        
    attn_mask = net.attn_mask.to(net.device)

    if seq_lens is not None:
        key_mask = net.generate_key_mask(seq_lens).to(net.device)
    else:
        key_mask = net.generate_key_mask(codes.shape[0]).to(net.device)
        
    for attn_layer in net.attn_layers:        
        out = attn_layer(out, attn_mask, key_mask)
        
    seq_out = out[:,net.mp:,:]

    token_out = net.token_head(seq_out)
        
    return token_out

def _fast_infer_prog(net, codes, seq):
      
    token_encs = net.token_enc_net(seq)
    
    out = torch.cat((codes.view(codes.shape[0], net.mp, net.hd), token_encs), dim = 1)

    fms = out.shape[1]
    
    out += net.pos_enc(
        net.pos_arange[:,:fms].repeat(
            codes.shape[0], 1
        ).to(net.device)
    )

    attn_mask = net.attn_mask[:fms,:fms].to(net.device)

    key_mask = net.generate_key_mask(codes.shape[0])[:,:fms].to(net.device)
    
    for attn_layer in net.attn_layers:        
        out = attn_layer(out, attn_mask, key_mask)
        
    seq_out = out[:,net.mp:,:]
    
    token_out = net.token_head(seq_out)
    
    return token_out

celoss = torch.nn.CrossEntropyLoss(reduction='none')

def calc_token_loss(preds, targets, weights):
    
    loss = (celoss(preds, targets) * weights).sum() / (weights.sum() + 1e-8)

    with torch.no_grad():
        corr = ((preds.argmax(dim=1) == targets).float() * weights).sum()
        total = weights.sum()
        
    return loss, corr, total


class V3DCNN(nn.Module):
    def __init__(
            self,
            inp_dim,
            max_prim_encs,
            out_dim,
            drop
    ):

        assert inp_dim == 64
        assert max_prim_encs == 4
        self.out_dim = out_dim        

        self.inp_dim = inp_dim
        
        super(V3DCNN, self).__init__()
                                                
        # Encoder architecture
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b1 = nn.BatchNorm3d(num_features=32)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b2 = nn.BatchNorm3d(num_features=64)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b3 = nn.BatchNorm3d(num_features=128)
        self.conv4 = nn.Conv3d(in_channels=128, out_channels=256,
                               kernel_size=4, stride=(1, 1, 1), padding=(2,
                                                                         2, 2))
        self.b4 = nn.BatchNorm3d(num_features=256)

        
        self.conv5 = nn.Conv3d(
            in_channels=256, out_channels=256,
            kernel_size=4, stride=(1, 1, 1), padding=(2, 2, 2))
        self.b5 = nn.BatchNorm3d(num_features=256)
            
        self._encoder = nn.Sequential(
                self.conv1,
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 2)),
                nn.Dropout(drop),
                self.b1,
                self.conv2,
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 2)),
                nn.Dropout(drop),
                self.b2,
                self.conv3,
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 2)),
                nn.Dropout(drop),
                self.b3,
                self.conv4,
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 2)),
                nn.Dropout(drop),
                self.b4,
                self.conv5,
                nn.ReLU(),
                nn.MaxPool3d(kernel_size=(2, 2, 2)),
                nn.Dropout(drop),
                self.b5,
        )
        
        self.ll = DMLP(256, 256, 256, self.out_dim // 2, drop)
            
    def forward(self, x):
        x1 = x.view(-1, 1, self.inp_dim, self.inp_dim, self.inp_dim)
        x2 = self._encoder(x1)
        x2 = x2.view(-1, self.out_dim, 8)            
        x2 = x2.transpose(1, 2)        
        x3 = self.ll(x2)
        o = x3.view(-1, 4, self.out_dim)
        return o
    
