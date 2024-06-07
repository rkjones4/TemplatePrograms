import time
import random
import math
from tqdm import tqdm
import dill
import utils
import torch
import json
from copy import deepcopy
import model_utils as mu

def split_beam_search(
    net,
    inp_batch,
    temp_beams,
    exp_beams,    
):

    pixels = inp_batch['vdata']
    codes = net.encode(pixels)        
        
    prob_codes = (codes).view(
        pixels.shape[0], pixels.shape[1], codes.shape[1], codes.shape[2]
    ).view(pixels.shape[0], -1, codes.shape[2])
                
    struct_preds = net.infer_struct_eval(prob_codes, temp_beams, 1)
    
    exp_codes = codes.view(
        pixels.shape[0], pixels.shape[1], codes.shape[1], codes.shape[2]
    )

    final_preds = {}
    final_log_info = {'struct': {}, 'param': {}}
    
    for si in range(pixels.shape[1]):

        assert len(struct_preds) == 1
        si_struct_preds = {0:[(ll, sisp.clone()) for ll,sisp in struct_preds[0]]}
        
        deriv_preds, struct_log_info = infer_deriv_eval(
            net,
            exp_codes[:,si:si+1,:,:],
            si_struct_preds,
            temp_beams,
            exp_beams
        )

        split_preds, split_log_info = infer_param_eval(
            net,
            exp_codes[:,si:si+1,:,:],
            deriv_preds,
            si_struct_preds,
            temp_beams,
            exp_beams,
            struct_log_info
        )

        for (i,j,k), V in split_preds.items():
            assert k == 0
            final_preds[(i,j,si)] = V
            
        final_log_info['struct'].update(split_log_info['struct'])
        
        for (i,j,k), V in split_log_info['param'].items():
            final_log_info['param'][(i,j,si)] = V
                
    ret_info = net.make_batch_ret_info(
        final_preds,
        final_log_info,
        pixels,
        temp_beams,
        inp_batch['extra_gt_data']
    )

    eval_res = net.make_new_result(ret_info)
    
    return ret_info, eval_res


def infer_deriv_eval(net, exp_codes, struct_preds, temp_beams, exp_beams):


    struct_info, struct_log_info = parse_struct_preds(
        net, struct_preds, exp_codes, temp_beams, exp_beams
    )
            
    max_left = net.deriv_net.ms - struct_info['bsinds'].min().item() + 1

    deriv_preds = net.ar_eval_logic(
        net.deriv_net,
        struct_info,
        exp_beams,
        0,
        max_left,
    )        
        
    for dpi, fp in deriv_preds.items():            
        fp.sort(reverse=True, key=lambda a: a[0])
        deriv_preds[dpi] = fp[:exp_beams]
                
    return deriv_preds, struct_log_info

def parse_struct_preds(net, struct_preds, exp_codes, temp_beams, exp_beams):

    
    batch = exp_codes.shape[0]
    nvi = exp_codes.shape[1]
    nsp = temp_beams
    ndp = exp_beams   

    bprefix = torch.zeros(
        batch,
        nsp,
        nvi,
        ndp,
        exp_codes.shape[-2],
        exp_codes.shape[-1],
        device=net.device
    ).float()
        
    bseqs = torch.zeros(
        batch, 
        nsp, 
        nvi,
        ndp,
        net.deriv_net.ms,
        device=net.device
    ).long()

    bpp_left = torch.zeros(
        batch,
        nsp,
        nvi,
        ndp,            
        device=net.device
    ).float()

    blls = torch.ones(
        batch,
        nsp,
        nvi,
        ndp,            
        device=net.device
    ).float() * mu.MIN_LL_PEN

    bsinds = torch.zeros(
        batch,            
        nsp,
        nvi,
        ndp,
        device=net.device
    ).long()

    struct_log_info = {}
        
    for bi, SP in struct_preds.items():
        for si, (s_ll, P) in enumerate(SP):
                
            out, num = net.ex.conv_struct_out_to_deriv_inp(P)
                    
            bseqs[bi,si,:,:,:out.shape[0]] = out
            bpp_left[bi,si,:,:] = num
            blls[bi,si,:,0] = s_ll
            bsinds[bi,si,:,:] = out.shape[0] - 1

            struct_log_info[(bi, si)] = net.ex.TLang.tensor_to_tokens(P)

        for ci in range(nvi):
            bprefix[bi,:,ci,:] = exp_codes[bi, ci]
        

    ttnn = net.TTNN_struct
    nloc_val = net.ex.TLang.T2I[f'{net.ex.STRUCT_LOC_TOKEN}_1']
                                
    bpp_nloc = torch.ones(
        batch,
        nsp,
        nvi,
        ndp,
        device=net.device
    ).long() * nloc_val

    pbs = batch * nsp * nvi * ndp
        
    bprefix = bprefix.view(
        pbs,
        exp_codes.shape[-2],
        exp_codes.shape[-1]
    )

    bseqs = bseqs.view(
        pbs,
        net.deriv_net.ms
    )

    bpp_left = bpp_left.flatten()
    bsinds = bsinds.flatten()
    bpp_nloc = bpp_nloc.flatten()
    blls = blls.flatten()        
        
    struct_info = {
        'bprefix': bprefix,
        'bseqs': bseqs,
        'bpp_left': bpp_left,
        'bsinds': bsinds,
        'bpp_nloc': bpp_nloc,
        'blls': blls,
        'batch': batch * nsp * nvi,
        'ttnn': ttnn
    }
        
    return struct_info, struct_log_info

def infer_param_eval(
    net, exp_codes, deriv_preds, struct_preds, temp_beams, exp_beams, struct_log_info
):

    deriv_info, deriv_log_info = parse_deriv_preds(
        net,
        deriv_preds,
        struct_preds,
        struct_log_info,
        exp_codes,
        temp_beams,
        exp_beams
    )                        
        
    max_left = net.param_net.ms - deriv_info['bsinds'].min().item() + 1
            
    param_preds = net.ar_eval_logic(
        net.param_net,
        deriv_info,
        exp_beams,
        0,
        max_left,
    )
            
    final_preds, final_log_info = parse_final_preds(
        net,
        param_preds,
        struct_log_info,
        exp_codes.shape[0],
        exp_codes.shape[1],
        temp_beams,
        exp_beams,
        'param'
    )

    return final_preds, final_log_info
        
def parse_deriv_preds(
    net, deriv_preds, struct_preds, struct_log_info, exp_codes, temp_beams, exp_beams
):
    
    
    batch = exp_codes.shape[0]
    nvi = exp_codes.shape[1]
    nsp = temp_beams
    ndp = exp_beams        

    bprefix = torch.zeros(
        batch,
        nsp,
        nvi,
        ndp,
        exp_codes.shape[-2],
        exp_codes.shape[-1],
        device=net.device
    ).float()
        
    bseqs = torch.zeros(
        batch, 
        nsp, 
        nvi,
        ndp,
        net.param_net.ms,
        device=net.device
    ).long()

    bpmask = torch.zeros(
        batch, 
        nsp, 
        nvi,
        ndp,
        net.param_net.ms,
        device=net.device
    ).long()

    bpp_left = torch.zeros(
        batch,
        nsp,
        nvi,
        ndp,            
        device=net.device
    ).float()

    blls = torch.ones(
        batch,
        nsp,
        nvi,
        ndp,            
        device=net.device
    ).float() * mu.MIN_LL_PEN

    bsinds = torch.zeros(
        batch,            
        nsp,
        nvi,
        ndp,
        device=net.device
    ).long()

    deriv_log_info = {}
    
    add_end_loc = True
        
    for dpi, fp in deriv_preds.items():
            
        batch_ind, struct_ind, vis_ind = parse_inds(
            dpi, batch, nvi, temp_beams, exp_beams
        )
            
        if (batch_ind, struct_ind) not in struct_log_info:
            continue
            
        no_deriv = (
            net.ex.STRUCT_LOC_TOKEN not in \
            ' '.join(struct_log_info[(batch_ind, struct_ind)])
        ) 
                        
        if no_deriv:                

            s_ll, P = struct_preds[batch_ind][struct_ind]

            try:
                out, num  = net.ex.conv_deriv_out_to_param_inp(P)
            except Exception as e:
                out = None

            if out is None:
                continue
                
            bseqs[batch_ind,struct_ind,vis_ind,:,:out.shape[0]] = out
            bpp_left[batch_ind,struct_ind,vis_ind,:] = num
            blls[batch_ind,struct_ind,vis_ind,0] = s_ll
            bsinds[batch_ind,struct_ind,vis_ind,:] = out.shape[0] - 1
                
        else:

            for deriv_ind, (d_ll, P) in enumerate(fp):

                try:
                    out, num = net.ex.conv_deriv_out_to_param_inp(P)

                except Exception as e:
                    out = None

                if out is None:
                    continue
                    
                bseqs[batch_ind,struct_ind,vis_ind,deriv_ind,:out.shape[0]] = out
                bpp_left[batch_ind,struct_ind,vis_ind,deriv_ind] = num
                blls[batch_ind,struct_ind,vis_ind,deriv_ind] = d_ll
                bsinds[batch_ind,struct_ind,vis_ind,deriv_ind] = out.shape[0] -1
                                        

    for bi in range(batch):
        for ci in range(nvi):
            bprefix[bi,:,ci,:] = exp_codes[bi, ci]
                

    bpp_nloc = torch.ones(
        batch,
        nsp,
        nvi,
        ndp,
        device=net.device
    ).long() * net.ex.TLang.T2I[f'{net.ex.PARAM_LOC_TOKEN}_1'] 

    pbs = batch * nsp * nvi * ndp
        
    bprefix = bprefix.view(
        pbs,
        exp_codes.shape[-2],
        exp_codes.shape[-1]
    )

    bseqs = bseqs.view(
        pbs,
        net.param_net.ms
    )
    
    bpmask = bpmask.view(
        pbs,
        net.param_net.ms
    )

    bpp_left = bpp_left.flatten()
    bsinds = bsinds.flatten()
    bpp_nloc = bpp_nloc.flatten()
    blls = blls.flatten()
    
    deriv_info = {
        'bprefix': bprefix,
        'bseqs': bseqs,
        'bpmask': bpmask,
        'bpp_left': bpp_left,
        'bsinds': bsinds,
        'bpp_nloc': bpp_nloc,
        'blls': blls,
        'batch': batch * nsp * nvi,
        'ttnn': net.TTNN_full
    }
    
    return deriv_info, deriv_log_info


def parse_final_preds(
    net, deriv_preds, struct_log_info, batch, nvi, temp_beams, exp_beams, log_field
):

    final_log_info = {'struct': {}, log_field: {}}
    final_preds = {}
                
    for dpi, fp in deriv_preds.items():

        batch_ind, struct_ind, vis_ind = parse_inds(
            dpi, batch, nvi, temp_beams, exp_beams
        )

        fkey = (batch_ind, struct_ind, vis_ind)

        final_preds[fkey] = []
            
        if (batch_ind, struct_ind) not in struct_log_info:
            continue

        final_log_info['struct'][(batch_ind, struct_ind)] = \
            struct_log_info[(batch_ind, struct_ind)]
                        
        final_log_info[log_field][fkey] = []
            
        for beam_ind, (_, pred) in enumerate(fp):

            try:
                tokens = net.ex.TLang.tensor_to_tokens(pred)
                
                full_prog, deriv = net.ex.cs_instantiate(
                    tokens,
                    rci=True
                )
            except Exception as e:
                continue
                
                
            final_preds[fkey].append(full_prog)
            final_log_info[log_field][fkey].append(deriv)

    return final_preds, final_log_info
    
def parse_inds(
    dpi, batch, nvi, temp_beams, exp_beams
):

    batch_ind = dpi // (nvi * temp_beams)
    res = dpi - (batch_ind * nvi * temp_beams)
    struct_ind = res // nvi
    res = res - (struct_ind * nvi)
    vis_ind = res % nvi
    
    return batch_ind, struct_ind, vis_ind
