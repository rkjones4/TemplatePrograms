import random
from tqdm import tqdm
import math
import torch
import gutils as gu
import tutils as tu
import lutils as lu
import eutils as eu
import numpy as np
import time
import matplotlib.pyplot as plt

BASE_CONFIG = {
    
    'MAX_HOLES': 4,
    'NUM_SHARED_TOKENS': 4,

    'EXTRA_DERIV_SAMP_RATIO': 10,
    'BREAK_EARLY': True,
    'SP_PATIENCE': 5,

    'CAT_REL_TYPES': ['indep', 'static', 'reuse'],
    'VERBOSE': False,

    'START_TOKEN': 'START',
    'HOLE_TOKEN': 'HOLE',
    'PARAM_LOC_TOKEN' : '$',
    'STRUCT_LOC_TOKEN' : '&',
    'SHARED_TOKEN' : '%',    
    'BLANK_TOKEN' : '#',
    'HOLE_PARAM_TOKEN' : '@',

    'USE_SIG_CHECK': True,
    'SPEC_GROUP_SAMPLE_TYPES': [],    
}

class Token:
    def __init__(self, name, inp_types, out_type, use_type):
        self.name = name
        self.inp_types = inp_types
        self.out_type = out_type
        self.use_type = use_type

    def num_inp_tokens(self):
        if self.inp_types == '':
            return 0
        else:            
            return len(self.inp_types.split(','))

    def num_struct_inp_tokens(self, struct_types):
        if self.inp_types == '':
            return 0
        else:
            return len([
                x for x in self.inp_types.split(',')
                if x in struct_types
            ])
        
class TokenLang:
    def __init__(self, executor):
        self.ex = executor
        self.tokens = {}
        self.float_samplers = {}
        self.type_sampler = {}
        
    def add_token(self, name, inp_types, out_type, use_type='def'):
        assert use_type in ('def', 'inp_only', 'out_only')

        t = Token(name, inp_types, out_type, use_type)
        self.tokens[name] = t

    def add_type_sampler(self, tname, sample_fn):
        self.type_sampler[tname] = sample_fn
        
    def add_float_sampler(
        self, name, sample_fn
    ):
        assert name in self.tokens
        self.float_samplers[name] = sample_fn

    def make_params(self):
        self.params = set([
            t.name for t in self.tokens.values() \
            if t.inp_types == '' and t.use_type != 'inp_only'
        ])

    def make_ot2t(self):
        OT2T = {}
        for t in self.tokens.values():
            if t.use_type == 'inp_only':
                continue
            ot = t.out_type
            if ot not in OT2T:
                OT2T[ot] = []
            OT2T[ot].append(t.name)

        self.OT2T = OT2T
        
    def make_t2ipc(self):
        self.T2IPC = {
            t.name:t.num_inp_tokens() for t in self.tokens.values()
        }
        self.T2SIPC = {
            t.name:t.num_struct_inp_tokens(self.ex.DEF_STRUCT_TYPES)
            for t in self.tokens.values()
        }

    def add_shared_tokens(self, num):
        for i in range(num):
            self.add_token(f'{self.ex.SHARED_TOKEN}_{i}', '', 'shared')
        self.init()
        
    def init(self):
        self.make_params()
        self.make_t2ipc()
        self.make_ot2t()
        if 'float' in self.OT2T:
            self.float_vals = [float(f) for f in self.OT2T['float']]
        self.make_token_maps()

        self.ex.extra_tlang_logic()                
        
    def make_token_maps(self):
        t2i = {}
        self.nt_inp = 0
        self.nt_out = 0
        for ut in ['def', 'out_only', 'inp_only']:            
            for t in self.tokens.values():
                if t.use_type == ut:
                    t2i[t.name] = len(t2i)

                    self.nt_inp += 1

                    if ut in ('def', 'out_only'):
                        self.nt_out += 1

        self.T2I = t2i
        self.I2T = {v:k for k,v in self.T2I.items()}
        self.nt = self.nt_inp
    
    def get_num_tokens(self):
        return self.nt_inp

    def tensor_to_tokens(self, tensor, keep_last=False):
        if keep_last:
            return [self.I2T[t.item()] for t in tensor]
        else:
            return [self.I2T[t.item()] for t in tensor[:-1]]
    
    def tokens_to_tensor(self, tokens):
        return torch.tensor([self.T2I[t] for t in tokens])
        
    def sample_param_val(self, ptype, last_fn, param_pos, global_scope, local_scope):

        if ptype in self.type_sampler:
            return self.type_sampler[ptype](global_scope, local_scope)
        
        if ptype != 'float':
            return self.sample_uni_val(ptype)

        val = self.float_samplers[last_fn](
            self.float_vals, param_pos, global_scope, local_scope
        )

        if float(val) not in self.float_vals:
            val = str(lu.round_val(float(val), self.float_vals))
            
        assert float(val) in self.float_vals
        return val

    def group_sample_param_val(self, ptype, last_fn, param_pos, tokens):

        if ptype in self.type_sampler:
            return self.type_sampler[ptype](tokens)
        
        assert ptype != 'float'
        
        return self.sample_uni_val(ptype)
        
    def sample_uni_val(self, ptype):
        return random.choice(self.OT2T[ptype])

    def is_valid_float_param(self, val, last_fn):
        assert last_fn in self.float_constr

        fval = float(val)

        if fval <= self.float_constr[last_fn][0] or\
           fval >= self.float_constr[last_fn][1]:
            return False
        
        return True

    def get_num_inp(self, t):        
        if t in self.T2IPC:
            return self.T2IPC[t]
        else:
            return 0

    def get_num_prob_struct_inp(self, t):
        ot = self.get_out_type(t)
        if t == self.ex.START_TOKEN:
            return 0
        elif ot in self.ex.DEF_STRUCT_TYPES or ot == self.ex.HOLE_TOKEN:
            if self.T2SIPC[t] is None:
                return self.tokens[t].num_struct_inp_tokens(self.ex.DEF_STRUCT_TYPES)
            return self.T2SIPC[t]
        elif ot != 'float':
            # a non-float param is a "no-op"
            return 1
        else:
            return 0
        
    def get_num_struct_inp(self, t):
        if t in self.T2SIPC:
            return self.T2SIPC[t]
        else:
            return 0

    def get_num_inp_eq(self, t, eq):
        return self.tokens[t].inp_types.count(eq)

    def get_inp_types(self, t):

        if t not in self.tokens:
            return []
        
        return [ip for ip in self.tokens[t].inp_types.split(',') if len(ip) > 0]

    def get_out_type(self, t):
        if t not in self.tokens:
            return ''
        else:
            return self.tokens[t].out_type

class DummyProg:
    def __init__(self,img):
        self.img = img.cpu().detach()
        
    def make_image(self):
        return self.img

class BaseExecutor:

    def extra_tlang_logic(self):
        pass
    
    def make_plot_render(self, num, gs, fs):

        fig, axes = plt.subplots(num, gs, figsize=fs)

        for i in range(num):
            for j in range(gs):
                axes[i,j].set_xticks([])
                axes[i,j].set_yticks([])
                axes[i,j].axis('off')
                
        return fig, axes
        
    def base_init(self, prm_config):
    
        BASE_CONFIG.update(prm_config)

        for k,v in BASE_CONFIG.items():
            setattr(self,k,v)
            
    def make_infer_data(self, infos, args):
        return eu.InferData(self, infos, args)
            
    def cs_instantiate(self, tokens, rci=False):
        return tu.cs_instantiate(self, tokens, rci)

    def conv_struct_out_to_deriv_inp(self, tokens):
        return tu.conv_struct_out_to_deriv_inp(self, tokens)

    def find_deriv(self, expr, struct):
        return tu.find_deriv(self, expr, struct)
    
    def conv_deriv_out_to_param_inp(self, tokens):
        return tu.conv_deriv_out_to_param_inp(self, tokens)

    def make_batch(self, data, args):
        return tu.make_batch(self, data, args)
    
    def sample_cat_rel(self):
        return np.random.choice(self.CAT_REL_TYPES,p=self.CAT_REL_PROBS)

    def add_part_info(self, expr, struct):
        return eu.add_part_info(self, expr, struct)
    
    def sample_cat_rel_for_pt(self, pt):
        return np.random.choice(self.CAT_REL_TYPES,p=self.TYPED_CAT_REL_PROBS[pt])                            
    def group_prog_random_sample(
        self, num_progs, num_derivs_per_prog, vis_progs=False, use_pbar=True
    ):
        with torch.no_grad():
            return self._group_prog_random_sample(num_progs, num_derivs_per_prog, vis_progs, use_pbar)

    def _group_prog_random_sample(
        self, num_progs, num_derivs_per_prog, vis_progs, use_pbar
    ):
        
        max_tokens = self.MAX_TOKENS
        max_struct_tokens = self.MAX_STRUCT_TOKENS
        max_param_tokens = self.MAX_PARAM_TOKENS
        
        data = []
        start_time = time.time()
        
        if not use_pbar:
            pbar = None
        else:
            pbar = tqdm(total=num_progs)        
        
        sample_params = None
            
        if vis_progs:
            vis_imgs = []

        t = 0
            
        while len(data) < num_progs:        
                    
            if t > self.SP_PATIENCE or sample_params is None:
                sample_params = self.get_group_sample_params()
                t = 0
                
            t += 1

            tokens = self.sample_det_prog(sample_params)
            
            if tokens is None or len(tokens) >= max_tokens:                
                continue

            try:
                # add holes and relationships and distributions
                group_sampler = gu.GroupSampler(self, tokens, sample_params)
            except Exception as e:
                print(f"Failed to group sample {e}")
                continue
            
            rels = group_sampler.get_relations()

            rel_counts = {}

            for rt,_ in rels.values():
                if rt not in rel_counts:
                    rel_counts[rt] = 0.
                rel_counts[rt] += 1.

            rcount = 0            
            if 'reuse' in rel_counts:
                rcount = rel_counts['reuse']

            if rcount > self.NUM_SHARED_TOKENS:
                continue
            
            struct_prog_tokens = group_sampler.get_struct_prog_tokens()

            num_struct_tokens = len(struct_prog_tokens) + len(rels) + rcount
            
            if num_struct_tokens >= max_struct_tokens:
                continue
            
            valid = []
            seen = set()

            for dcnt in range(num_derivs_per_prog * self.EXTRA_DERIV_SAMP_RATIO):
                if self.BREAK_EARLY and dcnt == num_derivs_per_prog and len(valid) == 0:
                    break                            
            
                deriv = group_sampler.sample()

                if len(deriv.tokens) >= max_tokens:
                    continue

                if len(deriv.param_vals.keys()) >= max_param_tokens:                    
                    continue

                assert self.USE_SIG_CHECK

                try:
                    deriv.ex_prog()
                except Exception as e:
                    continue

                # Get the signature of all primitives
                sig = deriv.get_sig()

                if sig is not None and sig in seen:
                    continue

                seen.add(sig)
            
                # Do validity checks
                if deriv.is_valid():
                    valid.append(deriv)
                                            
                if len(valid) == num_derivs_per_prog:
                    break

            if len(valid) < num_derivs_per_prog:
                continue

            if vis_progs:                
                for v in valid:                    
                    vis_imgs.append(v.P.make_image())
                    
            data.append(
                tu.SampleData(
                    group_sampler, 
                    valid,
                )                
            )

            if pbar is not None:
                pbar.update(1)

            sample_params = None
            t = 0
            
        if vis_progs:
            self.render_group(vis_imgs, rows = len(data))
            
        return data
