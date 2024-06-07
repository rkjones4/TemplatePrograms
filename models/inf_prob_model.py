import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.model_utils as mu
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import utils

VERBOSE = False

class ProbProgInfNet(nn.Module):
    def __init__(
        self,
        domain
    ):

        super(ProbProgInfNet, self).__init__()

        self.vis_mode = None
        self.domain = domain
        args = self.domain.args

        self.mode = 'recon'        
        
        self.device = domain.device

        self.mp = args.max_prim_enc
        self.hd = args.hidden_dim
        self.ex = domain.executor
                
        self.ar_mode = 'beam'

        self.encoder = mu.load_vis_encoder(domain)
        
        sn_ms = args.max_struct_tokens
        dn_ms = sn_ms + args.max_deriv_tokens
        pn_ms = sn_ms + args.max_deriv_tokens + args.max_param_tokens + args.max_param_tokens

        self.eval_info = None
        self.eval_count = {}
        self.eval_res = {}
        self.eval_pres = {}
        self.num_write = args.num_write
        
        self.struct_net = mu.TDecNet(
            domain,
            self.mp * args.max_vis_inputs,
            sn_ms
        )

        self.deriv_net = mu.TDecNet(
            domain,
            self.mp,
            dn_ms
        )
        
        self.param_net = mu.TDecNet(
            domain,
            self.mp,
            pn_ms
        )
                    
        self.TTNN_full = torch.zeros(
            self.ex.TLang.nt,
            device=self.device
        ).long()

        self.TTNN_struct = torch.zeros(
            self.ex.TLang.nt,
            device=self.device
        ).long()
        
        self.TTNN_prob_struct = torch.zeros(
            self.ex.TLang.nt,
            device=self.device
        ).long()
        
        for t in self.ex.TLang.tokens.keys():
            tind = self.ex.TLang.T2I[t]
            self.TTNN_full[tind] = self.ex.TLang.get_num_inp(t)
            nsi = self.ex.TLang.get_num_struct_inp(t)
            if nsi is None:
                assert domain.name == 'shape'
                nsi = 1
            self.TTNN_struct[tind] = nsi
            self.TTNN_prob_struct[tind] = self.ex.TLang.get_num_prob_struct_inp(t)
                    
    ###################
    ### TRAIN LOGIC ###
    ###################
    
    def encode(self, pixels):
        return self.encoder(pixels)
    
    def get_codes(self, batch):
        vdata = batch['vdata']        
        codes = self.encode(vdata)

        prob_codes = codes.view(
            vdata.shape[0], vdata.shape[1], codes.shape[1], codes.shape[2]
        ).view(vdata.shape[0], -1, codes.shape[2])                    
        
        return codes, prob_codes

    def make_struct_preds(self, batch, prob_codes, res):
        struct = batch['struct_seq']
        struct_weight = batch['struct_seq_weight']
        
        struct_preds = self.struct_net.infer_prog(
            prob_codes,
            struct
        )
        
        flat_struct_preds = struct_preds[:,:-1,:].reshape(-1,struct_preds.shape[2])
        flat_struct_targets = struct[:,1:].flatten()    
        flat_struct_weights = struct_weight[:,1:].flatten()
        
        loss, corr, total = mu.calc_token_loss(
            flat_struct_preds,
            flat_struct_targets,
            flat_struct_weights
        )

        res['struct_loss'] = loss
        res['struct_corr'] = corr
        res['struct_total'] = total

    def make_deriv_preds(self, batch, codes, res):

        E_deriv = batch['deriv_seq']
        E_deriv_weight = batch['deriv_seq_weight']
        
        deriv = E_deriv.view(
            E_deriv.shape[0] * E_deriv.shape[1],
            E_deriv.shape[2]
        )        
        
        deriv_preds = self.deriv_net.infer_prog(
            codes,
            deriv
        )

        flat_deriv_preds = deriv_preds[:,:-1,:].reshape(-1, deriv_preds.shape[2])
        
        flat_deriv_targets = E_deriv[:,:,1:].flatten()    
        flat_deriv_weights = E_deriv_weight[:,:,1:].flatten()        
        
        loss, corr, total = mu.calc_token_loss(
            flat_deriv_preds,
            flat_deriv_targets,
            flat_deriv_weights
        )

        res['deriv_loss'] = loss
        res['deriv_corr'] = corr
        res['deriv_total'] = total


    def make_param_preds(self, batch, codes, res):

        E_param = batch['param_seq']
        E_param_weight = batch['param_seq_weight']
        
        param = E_param.view(
            E_param.shape[0] * E_param.shape[1],
            E_param.shape[2]
        )                
        
        param_preds = self.param_net.infer_prog(
            codes,
            param
        )
        
        flat_param_preds = param_preds[:,:-1,:].reshape(-1, param_preds.shape[2])
        
        flat_param_targets = E_param[:,:,1:].flatten()    
        flat_param_weights = E_param_weight[:,:,1:].flatten()                
        
        loss, corr, total = mu.calc_token_loss(
            flat_param_preds,
            flat_param_targets,
            flat_param_weights
        )
        
        res['param_loss'] = loss
        res['param_corr'] = corr
        res['param_total'] = total

    # Loss logic for main reconstruction networks
    def model_train_batch(self, batch):

        res = {}
        
        codes, prob_codes = self.get_codes(
            batch,
        )
        
        self.make_struct_preds(batch, prob_codes, res)
        
        self.make_deriv_preds(batch, codes, res)

        self.make_param_preds(batch, codes, res)            

        loss = res['struct_loss'] + res['deriv_loss'] + res['param_loss']
                    
        res['loss'] = loss

        res = {k:v.item() for k,v in res.items()}

        return loss, res

    # Loss logic for main generation networks used during finetuning
    def ws_model_train_batch(self, batch):

        res = {}
        
        codes = torch.zeros(
            batch['vdata'].shape[0] * batch['vdata'].shape[1],
            self.mp,
            self.hd,
            device=batch['vdata'].device            
        ).float()
        
        self.make_struct_preds(
            batch,
            codes.view(batch['vdata'].shape[0], -1, self.hd),
            res
        )
        
        self.make_deriv_preds(batch, codes, res)

        self.make_param_preds(batch, codes, res)
        
        loss = res['struct_loss'] + res['deriv_loss'] + res['param_loss']
                    
        res['loss'] = loss

        res = {k:v.item() for k,v in res.items()}

        return loss, res

    # Loss logic for mean aggregation networks for few-shot generation
    def magg_model_train_batch(self, batch):

        res = {}
        
        codes, _ = self.get_codes(
            batch,
        )

        exp_codes = codes.view(
            batch['vdata'].shape[0], batch['vdata'].shape[1], codes.shape[1], codes.shape[2]
        )
        
        mean_codes = exp_codes.mean(dim=1)
        
        exp_codes = mean_codes.view(
            mean_codes.shape[0], 1, mean_codes.shape[1], mean_codes.shape[2]
        ).repeat(
            1, self.domain.args.max_vis_inputs, 1, 1
        ).view(
            -1, mean_codes.shape[1], mean_codes.shape[2]
        )
        
        self.make_deriv_preds(batch, exp_codes, res)

        self.make_param_preds(batch, exp_codes, res)            

        loss = res['deriv_loss'] + res['param_loss']
                    
        res['loss'] = loss

        res = {k:v.item() for k,v in res.items()}

        return loss, res

    
    ###################
    ### EVAL LOGIC ###
    ###################
    
    def eval_infer_progs(
        self,
        pixels,            
        beams,
        min_gpp_len,
        max_dv_len,
        extra_gt_data
    ):
        
        if beams is None:
            beams = self.domain.args.beams

        vdata = pixels
        codes = self.encode(pixels)        
        
        prob_codes = (codes).view(
            pixels.shape[0], pixels.shape[1], codes.shape[1], codes.shape[2]
        ).view(pixels.shape[0], -1, codes.shape[2])
            
        exp_codes = codes.view(
            pixels.shape[0], pixels.shape[1], codes.shape[1], codes.shape[2]
        )

        struct_preds = self.infer_struct_eval(prob_codes, beams, min_gpp_len)

        deriv_preds, struct_log_info = self.infer_deriv_eval(
            exp_codes,
            struct_preds,
            beams,
            max_dv_len
        )

        final_preds, final_log_info = self.infer_param_eval(
            exp_codes,
            deriv_preds,
            struct_preds,
            beams,
            struct_log_info
        )

        ret_info = self.make_batch_ret_info(
            final_preds,
            final_log_info,
            pixels,
            beams,
            extra_gt_data
        )

        return ret_info

    def get_seed_codes(self, inp_group, num_gen):
        args = self.domain.args

        if self.model_name == 'blank_gen':
            return torch.zeros(
                num_gen, self.mp, args.hidden_dim, device=self.device
            ).float()
        
        elif self.model_name == 'magg':

            codes, _ = self.get_codes(
                {'vdata': inp_group},
            )
            
            exp_codes = codes.view(
                inp_group.shape[0], inp_group.shape[1], codes.shape[1], codes.shape[2]
            )
 
            mean_codes = exp_codes.mean(dim=1)

            exp_codes = mean_codes.repeat(num_gen, 1, 1)
            return exp_codes

        else:
            assert False
        
    def sample_gen_from_seed(
        self,
        codes,
        struct_seq,
        num_gen
    ):

        struct_preds = {}

        prev_ar_mode = self.ar_mode
        self.ar_mode = 'sample'
        
        for i in range(num_gen):
            struct_preds[i] = [(0.0, struct_seq.clone())]

        deriv_preds, struct_log_info = self.infer_deriv_eval(
            codes.unsqueeze(1),
            struct_preds,
            1,
            self.deriv_net.ms,
        )
        
        final_preds, _ = self.infer_param_eval(
            codes.unsqueeze(1),
            deriv_preds,
            struct_preds,
            1,
            struct_log_info,
        )
        self.ar_mode = prev_ar_mode

        gens = []
        
        for _v in final_preds.values():
            try:
                expr = ' '.join(_v[0])
                pixels = self.ex.execute(expr)
                gens.append((pixels, _v[0]))
            except Exception as e:
                pass
            
        return gens

    def sample_gens(
        self,
        first_num_gens,
    ):

        seen_structs = self.seen_structs
            
        nvi = self.domain.args.max_vis_inputs
        
        prev_ar_mode = self.ar_mode
        self.ar_mode = 'sample'

        struct_seed_codes = torch.zeros(
            first_num_gens,
            self.mp * nvi,
            self.hd,
            device=self.device
        ).float()

        first_deriv_seed_codes = torch.zeros(
            first_num_gens,
            1,
            self.mp,
            self.hd,
            device=self.device
        ).float()
        
        first_struct_preds = self.infer_struct_eval(
            struct_seed_codes, 1, 1
        )
        
        ref_struct_preds = {} 
        
        for k,v in first_struct_preds.items():
            if len(v) == 0:
                continue
            
            assert len(v) == 1
            rv = v[0][1].detach().clone()
            ref_struct_preds[k] = [(0., rv)]        
            
        first_deriv_preds, first_struct_log_info = self.infer_deriv_eval(
            first_deriv_seed_codes,
            first_struct_preds,
            1,
            self.deriv_net.ms
        )        
        
        first_final_preds, first_final_log_info = self.infer_param_eval(
            first_deriv_seed_codes,
            first_deriv_preds,
            first_struct_preds,
            1,
            first_struct_log_info
        )

        keep_final_preds = {}
        struct_preds = {}        

        # try to sample slightly more derivations per template program
        grace_num = self.grace_num

        last_ind = nvi + grace_num - 1
        
        for i in range(first_num_gens):
            if (i,0,0) not in first_final_preds:
                continue
            
            try:
                expr = ' '.join(first_final_preds[(i,0,0)][0])
                struct = first_final_log_info['struct'][(i,0)]
            except Exception as e:
                continue
            
            if tuple(struct) in seen_structs:
                continue
            
            P = self.ex.prog_cls(self.ex)
            try:
                P.run(expr.split())
                pixels = P.make_image()
            except Exception as e:
                continue

            if not self.ex.check_valid_prog(P):
                continue

            sp_ind = len(struct_preds)
            struct_preds[sp_ind] = ref_struct_preds[i]
            keep_final_preds[(sp_ind,0,last_ind)] = first_final_preds[(i,0,0)]
            if VERBOSE:
                print(f"Kept {i}")
                
        num_gens = len(struct_preds)
        if VERBOSE:
            print(f"First stage filtered {first_num_gens} -> {num_gens}")            
                        
        deriv_seed_codes = torch.zeros(
            num_gens,
            nvi + grace_num - 1,
            self.mp,
            self.hd,
            device=self.device
        ).float()
            
        deriv_preds, struct_log_info = self.infer_deriv_eval(
            deriv_seed_codes,
            struct_preds,
            1,
            self.deriv_net.ms,
        )

        final_preds, final_log_info = self.infer_param_eval(
            deriv_seed_codes,
            deriv_preds,
            struct_preds,
            1,
            struct_log_info,
        )
        
        final_preds.update(keep_final_preds)
        
        self.ar_mode = prev_ar_mode

        gens = []        
                
        for i in range(num_gens):
            group = []
            group_sigs = set()
            
            for j in range(nvi+grace_num):                
                if len(group) >= nvi:
                    break
                
                if (i,0,j) not in final_preds:                    
                    continue
                
                expr = None
                try:
                    expr = ' '.join(final_preds[(i,0,j)][0])
                    struct = final_log_info['struct'][(i,0)]
                except Exception as e:
                    if VERBOSE:
                        print(f"Failed info lookup for {(i,j)} with {e}")
                        print(final_preds.keys())
                        print(final_log_info['struct'].keys())
                        
                    continue
                
                P = self.ex.prog_cls(self.ex)
                try:
                    P.run(expr.split())
                    pixels = P.make_image()
                    sig = P.get_state_sig()                    
                except Exception as e:                    
                    continue

                if torch.isnan(pixels).any():
                    continue

                if not self.ex.check_valid_prog(P):
                    continue

                if sig in group_sigs:
                    continue
                                        
                group_sigs.add(sig)

                group.append(
                    (pixels, expr, struct)
                )

            if len(group) < nvi:
                if VERBOSE:
                    print(f"missed filling group {i} after {j} iters")
                continue
            else:
                if VERBOSE:
                    print(f'took {j} iters to fill group {i}')
                
            infos = [{'expr': g[1], 'struct': g[2]} for g in group]

            try:
                infd = self.ex.make_infer_data(infos, self.domain.args)
            except Exception as e:
                if VERBOSE:
                    print(f"Failed conversion to infer data with {e}")
                continue

            seen_structs.add(tuple(infos[0]['struct']))
            gens.append(infd)
                            
        return gens
                
    def parse_pred_info(
        self, batch_ind, struct_ind, vis_ind,
        final_preds, final_log_info, pixels, egt
    ):
        vis_mval = self.domain.init_metric_val()
        vis_exec = torch.zeros(self.ex.get_input_shape()).to(pixels[0].device)
        vis_info = {}
                
        for beam_ind, beam_prog in enumerate(final_preds[
            (batch_ind, struct_ind, vis_ind)
        ]):

            expr = ' '.join(beam_prog)

            if egt is not None:
                _egt = egt[batch_ind][vis_ind]
            else:
                _egt = None

            ex_struct_info = None
            ex_deriv_info = None
            ex_param_info = None
            
            if 'struct' in final_log_info:
                ex_struct_info = final_log_info['struct'][
                    (batch_ind, struct_ind)
                ]

            if 'param' in final_log_info:
                ex_param_info = final_log_info['param'][
                    (batch_ind, struct_ind, vis_ind) 
                ][beam_ind]

            try:
                ex_deriv_info = self.ex.find_deriv(expr, ex_struct_info)
            except:
                pass
                
            prog_info = {
                'struct': ex_struct_info,
                'deriv': ex_deriv_info,
                'param': ex_param_info
            }

            _exec, _mval, _minfo = self.domain.get_vis_metric(
                expr,
                pixels[batch_ind, vis_ind],
                extra=_egt,
                prog_info=prog_info
            )

            if _exec is None or _mval is None:
                continue
                                        
            if self.domain.comp_metric(_mval, vis_mval):
                vis_mval = _mval
                vis_exec = _exec
                vis_info = _minfo
                vis_info['expr'] = expr

                if ex_struct_info is not None:
                    vis_info['struct'] = ex_struct_info
                    
                if ex_deriv_info is not None:
                    vis_info['deriv'] = ex_deriv_info
                    
                if ex_param_info is not None:
                    vis_info['param'] = ex_param_info
                
        if len(vis_info) == 0:
            return None, None, None
        return vis_mval, vis_exec, vis_info
                    
    def make_inst_ret_info(
        self, batch_ind, final_preds, final_log_info, pixels, beams, extra_gt_data
    ): 

        best_avg_match = self.domain.init_metric_val()
        best_mvals = []
        best_execs = []
        best_info = []

        nvi = pixels.shape[1]        

        for struct_ind in range(beams):

            struct_mvals = []
            struct_execs = []
            struct_info = []
            failed_inner = False
            for vis_ind in range(nvi):
                
                vis_mval, vis_exec, vis_info = self.parse_pred_info(
                    batch_ind, struct_ind, vis_ind,
                    final_preds, final_log_info, pixels, extra_gt_data
                )

                if vis_mval is None:
                    failed_inner = True
                    break
                                    
                struct_mvals.append(vis_mval)
                struct_execs.append(vis_exec)
                struct_info.append(vis_info)

            if failed_inner:
                continue
                
            avg_match = torch.tensor(struct_mvals).float().mean().item()
            
            if self.domain.comp_metric(avg_match, best_avg_match):
                best_execs = struct_execs
                best_mvals = struct_mvals
                best_info = struct_info
                best_avg_match = avg_match

        return {
            'execs': best_execs,
            'mvals': best_mvals,
            'info': best_info
        }

    
    def make_batch_ret_info(
        self, final_preds, final_log_info, pixels, beams, extra_gt_data
    ):
        R = {}

        batch_num = pixels.shape[0]        

        for batch_ind in range(batch_num):
            r = self.make_inst_ret_info(
                batch_ind, final_preds, final_log_info, pixels, beams, extra_gt_data
            )
            for k,v in r.items():
                if k not in R:
                    R[k] = []
                R[k].append(v)

        return R

    def parse_final_preds(
        self, deriv_preds, struct_log_info, batch, nvi, beams, log_field
    ):
        final_log_info = {'struct': {}, log_field: {}}
        final_preds = {}
                
        for dpi, fp in deriv_preds.items():

            batch_ind, struct_ind, vis_ind = mu.parse_inds(
                dpi, batch, nvi, beams
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
                    tokens = self.ex.TLang.tensor_to_tokens(pred)
                
                    full_prog, deriv = self.ex.cs_instantiate(
                        tokens,
                        rci=True
                    )
                except Exception as e:
                    continue
                                
                final_preds[fkey].append(full_prog)
                final_log_info[log_field][fkey].append(deriv)


        return final_preds, final_log_info
                

    def infer_param_eval(
            self, exp_codes, deriv_preds, struct_preds, beams, struct_log_info
    ):

        deriv_info, deriv_log_info = self.parse_deriv_preds(
            deriv_preds,
            struct_preds,
            struct_log_info,
            exp_codes,
            beams
        )                        
        
        max_left = self.param_net.ms - deriv_info['bsinds'].min().item() + 1
            
        param_preds = self.ar_eval_logic(
            self.param_net,
            deriv_info,
            beams,
            0,
            max_left,
        )
            
        final_preds, final_log_info = self.parse_final_preds(
            param_preds,
            struct_log_info,
            exp_codes.shape[0],
            exp_codes.shape[1],
            beams,
            'param'
        )

        return final_preds, final_log_info
        
    def parse_deriv_preds(
        self, deriv_preds, struct_preds, struct_log_info, exp_codes, beams
    ):
        
        batch = exp_codes.shape[0]
        nvi = exp_codes.shape[1]
        nsp = beams
        ndp = beams        

        bprefix = torch.zeros(
            batch,
            nsp,
            nvi,
            ndp,
            exp_codes.shape[-2],
            exp_codes.shape[-1],
            device=self.device
        ).float()
        
        bseqs = torch.zeros(
            batch, 
            nsp, 
            nvi,
            ndp,
            self.param_net.ms,
            device=self.device
        ).long()

        bpmask = torch.zeros(
            batch, 
            nsp, 
            nvi,
            ndp,
            self.param_net.ms,
            device=self.device
        ).long()

        bpp_left = torch.zeros(
            batch,
            nsp,
            nvi,
            ndp,            
            device=self.device
        ).float()

        blls = torch.ones(
            batch,
            nsp,
            nvi,
            ndp,            
            device=self.device
        ).float() * mu.MIN_LL_PEN

        bsinds = torch.zeros(
            batch,            
            nsp,
            nvi,
            ndp,
            device=self.device
        ).long()

        deriv_log_info = {}

        add_end_loc = True
        
        for dpi, fp in deriv_preds.items():

            batch_ind, struct_ind, vis_ind = mu.parse_inds(
                dpi, batch, nvi, beams
            )

            if (batch_ind, struct_ind) not in struct_log_info:
                continue
            
            no_deriv = (
                self.ex.STRUCT_LOC_TOKEN not in \
                ' '.join(struct_log_info[(batch_ind, struct_ind)])
            ) 
                        
            if no_deriv:                

                s_ll, P = struct_preds[batch_ind][struct_ind]

                try:
                    out, num  = self.ex.conv_deriv_out_to_param_inp(P)
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
                        out, num = self.ex.conv_deriv_out_to_param_inp(P)

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
            device=self.device
        ).long() * self.ex.TLang.T2I[f'{self.ex.PARAM_LOC_TOKEN}_1'] 

        pbs = batch * nsp * nvi * ndp
        
        bprefix = bprefix.view(
            pbs,
            exp_codes.shape[-2],
            exp_codes.shape[-1]
        )

        bseqs = bseqs.view(
            pbs,
            self.param_net.ms
        )

        bpmask = bpmask.view(
            pbs,
            self.param_net.ms
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
            'ttnn': self.TTNN_full
        }
        
        return deriv_info, deriv_log_info
        
    def parse_struct_preds(self, struct_preds, exp_codes, beams):

        batch = exp_codes.shape[0]
        nvi = exp_codes.shape[1]
        nsp = beams
        ndp = beams        

        bprefix = torch.zeros(
            batch,
            nsp,
            nvi,
            ndp,
            exp_codes.shape[-2],
            exp_codes.shape[-1],
            device=self.device
        ).float()
        
        bseqs = torch.zeros(
            batch, 
            nsp, 
            nvi,
            ndp,
            self.deriv_net.ms,
            device=self.device
        ).long()

        bpp_left = torch.zeros(
            batch,
            nsp,
            nvi,
            ndp,            
            device=self.device
        ).float()

        blls = torch.ones(
            batch,
            nsp,
            nvi,
            ndp,            
            device=self.device
        ).float() * mu.MIN_LL_PEN

        bsinds = torch.zeros(
            batch,            
            nsp,
            nvi,
            ndp,
            device=self.device
        ).long()

        struct_log_info = {}

        
        for bi, SP in struct_preds.items():
            for si, (s_ll, P) in enumerate(SP):
                
                out, num = self.ex.conv_struct_out_to_deriv_inp(P)
                    
                bseqs[bi,si,:,:,:out.shape[0]] = out
                bpp_left[bi,si,:,:] = num
                blls[bi,si,:,0] = s_ll
                bsinds[bi,si,:,:] = out.shape[0] - 1

                struct_log_info[(bi, si)] = self.ex.TLang.tensor_to_tokens(P)

            for ci in range(nvi):
                bprefix[bi,:,ci,:] = exp_codes[bi, ci]
        

        ttnn = self.TTNN_struct
        nloc_val = self.ex.TLang.T2I[f'{self.ex.STRUCT_LOC_TOKEN}_1']
                            
        bpp_nloc = torch.ones(
            batch,
            nsp,
            nvi,
            ndp,
            device=self.device
        ).long() * nloc_val

        pbs = batch * nsp * nvi * ndp
        
        bprefix = bprefix.view(
            pbs,
            exp_codes.shape[-2],
            exp_codes.shape[-1]
        )

        bseqs = bseqs.view(
            pbs,
            self.deriv_net.ms
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
            
    def infer_deriv_eval(self, exp_codes, struct_preds, beams, max_dv_len):

        struct_info, struct_log_info = self.parse_struct_preds(
            struct_preds, exp_codes, beams
        )
            
        max_left = self.deriv_net.ms - struct_info['bsinds'].min().item() + 1

        deriv_preds = self.ar_eval_logic(
            self.deriv_net,
            struct_info,
            beams,
            0,
            min(max_dv_len, max_left),
        )        
        
        for dpi, fp in deriv_preds.items():            
            fp.sort(reverse=True, key=lambda a: a[0])
            deriv_preds[dpi] = fp[:beams]
                
        return deriv_preds, struct_log_info
            
        
    def infer_struct_eval(self, prefix, beams, min_gpp_len):
        batch = prefix.shape[0]
        
        bprefix = prefix.view(
            prefix.shape[0], 1, prefix.shape[1], prefix.shape[2]
        ).repeat(1, beams, 1, 1).view(
            beams * batch, prefix.shape[1], prefix.shape[2]
        )
        
        info = {
            'batch': batch,
            'bprefix': bprefix
        }

        info['ttnn'] = self.TTNN_prob_struct
        info['bqc'] = torch.ones(
            batch * beams, device=self.device
        ).long() * 2
        info['bqc_extra'] = 0
            
        struct_preds = self.ar_eval_logic(
            self.struct_net,
            info,
            beams,
            min_gpp_len,
            self.struct_net.ms,
        )        
        
        for i, fp in struct_preds.items():            
            fp.sort(reverse=True, key=lambda a: a[0])
            struct_preds[i] = fp[:beams]

        return struct_preds
    
    def ar_eval_logic(
        self,
        net,
        info,
        beams,
        min_len,
        max_len,
    ):
        
        if self.ar_mode == 'sample':
            return self.ar_sample_logic(
                net,
                info,
                beams,
                min_len,
                max_len,
            )
        
        batch = info['batch']

        bprefix = info['bprefix']

        if 'bseqs' not in info:                                
            bseqs = torch.zeros(batch * beams, net.ms, device=self.device).long()
            bseqs[:,0] = self.ex.TLang.T2I[self.ex.START_TOKEN]
        else:
            bseqs = info['bseqs']

        if 'bpp_left' not in info:        
            bpp_left = torch.ones(batch * beams, device=self.device).float()
        else:
            bpp_left = info['bpp_left']

        if 'bpp_nloc' not in info:
            bpp_nloc = torch.zeros(batch * beams, device=self.device).long()
        else:
            bpp_nloc = info['bpp_nloc']

        if 'bsinds' not in info:
            bsinds = torch.zeros(batch * beams, device=self.device).long()            
        else:
            bsinds = info['bsinds']

        if 'ttnn' not in info:
            TTNN = self.TTNN_full
        else:
            TTNN = info['ttnn']

        if 'blls' not in info:
            # batch log liks
            blls = torch.zeros(batch, beams, device=self.device)        
            blls[:,1:] += mu.MIN_LL_PEN
            blls = blls.flatten()
        else:
            blls = info['blls']

        if 'bqc' not in info:
            bqc = torch.ones(
                batch * beams,
                device=self.device
            ).long()
            _extra = 1
        else:
            bqc = info['bqc']
            _extra = info['bqc_extra']
            
        # [batch, beam, O]

        max_token_num = self.struct_net.nt
        
        fin_progs = {i:[] for i in range(batch)}

        fin_count = torch.zeros(
            batch,
            device=self.device
        )
        
        break_cond = torch.zeros(batch, beams, device=self.device).bool()
        
        fin_lls = [[] for _ in range(batch)]
        
        dummy_arange = torch.arange(beams * batch, device=bprefix.device)        

        max_ind = bseqs.shape[1] - 1 
        
        for PL in range(max_len-1): 

            break_cond = break_cond | (bpp_left.view(batch, beams) <= 0)
            
            E_blls = blls.view(batch, beams)
            
            for i in (fin_count >= beams).nonzero().flatten():
                fin_nll = -1 * torch.tensor([
                    np.partition(fin_lls[i], beams-1)[beams-1]
                ], device=self.device)

                if (E_blls[i] < fin_nll).all():
                    break_cond[i] = True
            
            if break_cond.all():     
                break
            
            bpreds = net.fast_infer_prog(
                bprefix,
                bseqs,
                dummy_arange,
                bsinds
            )
            
            bdist = torch.log(torch.softmax(bpreds, dim = 1) + 1e-8)            
            
            beam_liks, beam_choices = torch.topk(bdist, beams)
            
            next_liks = (beam_liks + blls.view(-1, 1)).view(batch, -1)

            E_ll, E_ranked_beams = torch.sort(next_liks,1,True)

            blls = E_ll[:,:beams].flatten()

            ranked_beams = E_ranked_beams[:,:beams]

            R_beam_choices = beam_choices.view(batch, -1)

            nt = torch.gather(R_beam_choices,1,ranked_beams).flatten()

            old_index = (torch.div(ranked_beams, beams).float().floor().long() + (torch.arange(batch, device=self.device) * beams).view(-1, 1)).flatten()
            
            bseqs  = bseqs[old_index].clone()
            bsinds = bsinds[old_index].clone() + 1
            bsinds = torch.clamp(bsinds, 0, max_ind)
            bseqs[dummy_arange, bsinds] = nt

            bprefix = bprefix[old_index]

            bqc = bqc[old_index].clone()
            bqc -= 1            
            bqc += TTNN[nt]

            bpp_left = bpp_left[old_index].clone()
            bpp_nloc = bpp_nloc[old_index].clone()
            
            p_fin_inds = (bqc == 0.).nonzero().flatten()
            
            if p_fin_inds.shape[0] > 0:

                bpp_left[p_fin_inds] -= 1
                bqc[p_fin_inds] += 1                
                bsinds[p_fin_inds] += 1                

                bsinds = torch.clamp(bsinds, 0, max_ind)                
                
                bseqs[
                    dummy_arange[p_fin_inds],
                    bsinds[p_fin_inds]
                ] = bpp_nloc[p_fin_inds]
                
                bpp_nloc[p_fin_inds] += 1
                bpp_nloc = torch.clamp(bpp_nloc, 0, max_token_num-1)
                
            fin_inds = (bpp_left == 0.).nonzero().flatten().tolist()

            for i in fin_inds:
                                                                                
                if blls[i] > mu.MIN_LL_THRESH:

                    if bsinds[i] > min_len:
                        beam_ind = i // beams
                        _ll = blls[i].item()
                        fin_progs[beam_ind].append((
                            _ll,
                            bseqs[i,:bsinds[i]+_extra]
                        ))
                        fin_count[beam_ind] += 1
                        fin_lls[beam_ind].append(-1 * _ll)                        
                        
                blls[i] += mu.MIN_LL_PEN
                bqc[i] += 1
                
        return fin_progs

    def ar_sample_logic(
        self,
        net,
        info,
        beams,
        min_len,
        max_len,
    ):

        assert beams == 1
        
        batch = info['batch']

        bprefix = info['bprefix']

        if 'bseqs' not in info:                                
            bseqs = torch.zeros(batch, net.ms, device=self.device).long()
            bseqs[:,0] = self.ex.TLang.T2I[self.ex.START_TOKEN]
        else:
            bseqs = info['bseqs']

        if 'bpp_left' not in info:        
            bpp_left = torch.ones(batch, device=self.device).float()
        else:
            bpp_left = info['bpp_left']

        if 'bpp_nloc' not in info:
            bpp_nloc = torch.zeros(batch, device=self.device).long()
        else:
            bpp_nloc = info['bpp_nloc']

        if 'bsinds' not in info:
            bsinds = torch.zeros(batch, device=self.device).long()            
        else:
            bsinds = info['bsinds']

        if 'ttnn' not in info:
            TTNN = self.TTNN_full
        else:
            TTNN = info['ttnn']

        if 'bqc' not in info:
            bqc = torch.ones(
                batch,
                device=self.device
            ).long()
            _extra = 1
            
        else:
            bqc = info['bqc']
            _extra = info['bqc_extra']
            
        # [batch, beam, O]

        max_token_num = self.struct_net.nt
        
        fin_progs = {i:[] for i in range(batch)}

        in_count = (bpp_left > 0).float()
                        
        max_ind = bseqs.shape[1] - 1 

        dummy_arange = torch.arange(batch, device=bprefix.device)        
                        
        for PL in range(max_len-1):            

            barange = (in_count > 0).nonzero().flatten()
            if (in_count < 1.).all():
                break
            
            bpreds = net.fast_infer_prog(
                bprefix[barange],
                bseqs[barange],
                dummy_arange[:barange.shape[0]],
                bsinds[barange]
            )
                
            bdist = torch.softmax(bpreds, dim = 1)            
            nt = torch.distributions.categorical.Categorical(bdist).sample()

            bsinds += 1
            bsinds = torch.clamp(bsinds, 0, max_ind)

            bseqs[barange, bsinds[barange]] = nt

            bqc[barange] -= 1            
            bqc[barange] += TTNN[nt]
            
            p_fin_inds = (bqc[barange] == 0.).nonzero().flatten()
            
            if p_fin_inds.shape[0] > 0:
                
                bp_fin_inds = barange[p_fin_inds]
                
                bpp_left[bp_fin_inds] -= 1
                bqc[bp_fin_inds] += 1
                
                bsinds[bp_fin_inds] += 1                
                bsinds = torch.clamp(bsinds, 0, max_ind)                
                
                bseqs[
                    bp_fin_inds,
                    bsinds[bp_fin_inds]
                ] = bpp_nloc[bp_fin_inds]
                
                bpp_nloc[bp_fin_inds] += 1
                bpp_nloc = torch.clamp(bpp_nloc, 0, max_token_num-1)
                
            fin_inds = (bpp_left[barange] == 0.).nonzero().flatten()

            for _i in fin_inds:
                
                beam_ind = barange[_i].item()
                
                _ll = 0.0
                fin_progs[beam_ind].append((
                    _ll,
                    bseqs[beam_ind,:bsinds[beam_ind]+_extra]
                ))
                in_count[beam_ind] = 0.                        
                
        return fin_progs
    
                
    def model_eval_fn(
        self,
        vdata,
        beams=None,
        min_gpp_len=0,
        max_dv_len=1000,
        ret_info = False
    ):
        
        try:
            eval_info = self.eval_infer_progs(
                vdata['vdata'],
                beams,
                min_gpp_len,
                max_dv_len,
                vdata['extra_gt_data']
            )
            
        except Exception as e:
            print(f"Failed to eval infer progs with {e}")
            if ret_info:
                return {}, {}
            else:
                return {}

        if self.vis_mode is not None:
            if 'vis_vdata' in vdata:
                self.record_vis_logic(vdata['vis_vdata'], eval_info)
            else:
                self.record_vis_logic(vdata['vdata'], eval_info)

        try:
            eval_res = self.make_new_result(eval_info)
        except Exception as e:
            utils.log_print(f'!!!\nFailed to make result with {e}\n!!!', self.domain.args)
            if ret_info:
                return {}, {}
            else:
                return {}
        
        if ret_info:
            return eval_info, eval_res
        else:
            return eval_res
            
    def make_new_result(self, eval_info):
        res = {
            'dv_cnt': 0.,
            'mval_cnt': 0.,
            'gpp_cnt': 0.,
            'prm_cnt': 0.,
            
            'mval': 0.,

            'rec_cnt': 0.,
            'match': 0.,
            'prog_len': 0.,
            
            'gpp_len': 0.,
            'dv_len': 0.,
            'prm_len': 0.,
            'nh_amt': 0.,

            'errs': 0.
        }
        
        for mval_list in eval_info['mvals']:
            for mval in mval_list:
                res['mval'] += mval
                res['mval_cnt'] += 1.

        if len(eval_info['info']) == 0:
            res['errs'] += 1
                
        for info_list in eval_info['info']:

            if len(info_list) == 0:
                res['errs'] += 1
                continue
            
            if 'struct' not in info_list[0]:
                res['errs'] += 1
                continue
            
            if len(info_list) > 1 and \
               'struct' in info_list[0] and \
               'struct' in info_list[1]:
                assert info_list[0]['struct'] == info_list[1]['struct']

            res['gpp_cnt'] += 1.
            res['gpp_len'] += len(info_list[0]['struct'])

            for _info in info_list:
                if 'match' in _info:
                    
                    res['match'] += _info['match']
                    res['prog_len'] += _info['prog_len']
                    res['rec_cnt'] += 1.
            
            if True:
                if 'shared_cnt' not in res:
                    res['shared_cnt'] = 0.
                    res['static_cnt'] = 0.
                    
                saw_prim = False
                saw_share = False
                
                struct = info_list[0]['struct']

                for t in struct:
                    if self.ex.TLang.get_out_type(t) in self.ex.DEF_PARAM_TYPES:
                        saw_prim = True
                        break

                if self.ex.SHARED_TOKEN in ' '.join(struct):
                    saw_share = True

                res['shared_cnt'] += float(saw_share)
                res['static_cnt'] += float(saw_prim)
                    
            
            res['nh_amt'] += len([
                t for t in info_list[0]['struct'] if
                (
                    self.ex.STRUCT_LOC_TOKEN in t or \
                    self.ex.PARAM_LOC_TOKEN in t or\
                    self.ex.HOLE_TOKEN == t
                )
            ])

            
            if True:
                derivs = []
                for _info in info_list:
                    if 'expr' in _info:                        
                        _p = self.ex.find_deriv(_info['expr'], _info['struct'])
                        
                        derivs.append(
                            _p
                        )
                            
                for deriv in derivs:
                    res['dv_len'] += sum([len(dv.split()) for dv in deriv])
                    res['dv_cnt'] += 1                
                                
                        
            for _info in info_list:
                if 'param' in _info:
                    res['prm_len'] += len(_info['param'])
                    res['prm_cnt'] += 1
                
        return res

    def init_vis_logic(self):
        set_name, itn = self.vis_mode        
        
        if self.vis_mode not in self.eval_count:
            self.eval_count[self.vis_mode] = 0
            self.eval_res[self.vis_mode] = []
            self.eval_pres[self.vis_mode] = []

    def record_vis_logic(self, vdata, eval_info):
        for i in range(vdata.shape[0]):
            if self.eval_count[self.vis_mode] >= self.num_write:
                return

            info = eval_info['info'][i]
                
            preds = eval_info['execs'][i]
            targets = vdata[i]
            self.eval_count[self.vis_mode] += 1
            self.eval_res[self.vis_mode].append(list(zip(preds, targets)))
            

    def save_vis_logic(self):
        set_name, itn = self.vis_mode
        args = self.domain.args

        try:
                    
            name = f'{args.outpath}/{args.exp_name}/vis/{set_name}'
            pname = f'{args.outpath}/{args.exp_name}/progs/{set_name}'

            if len(self.eval_res[self.vis_mode]) == 0:
                return
        
            self.flush_group_images(self.eval_res[self.vis_mode], name, itn, vis_size = 2)

            self.eval_res.pop(self.vis_mode)
            self.eval_pres.pop(self.vis_mode)
            
        except Exception as e:
            utils.log_print(f"Failed to save vis for {set_name} ({itn}) with: {e}", args)

    def flush_group_progs(self, progs, name, itn):
        for i, prog in enumerate(progs):
            with open(f'{name}_itn_{itn}_b_{i}.txt', 'w') as f:
                f.write(prog)
            
    def flush_group_images(
            self, images, name, itn, vis_size=4, group_size = 5
    ):
        i = 0
        while len(images) > 0:
            batch = images[:vis_size]
            images = images[vis_size:]
            self._flush_group_images(batch,f'{name}_b_{i}_itn_{itn}', group_size)
            i += 1

    def _flush_group_images(self, images,name, group_size):
                
        fig, axes = self.ex.make_plot_render(
            len(images) * 2, group_size, (16,12)
        )
        
        for I,group in enumerate(images):

            for j, (pred,target) in enumerate(group):
            
                i = 2*I
                ii = (2*I) + 1

                if target is not None:
                    self.ex.vis_on_axes(axes[i,j], target.cpu().numpy())

                if pred is not None:
                    self.ex.vis_on_axes(axes[ii,j], pred.cpu().numpy())        
            
        plt.savefig(name)
        plt.close()
                
