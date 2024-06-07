import numpy as np
import torch
from copy import deepcopy
import random

class GroupSampler:
    def __init__(self, ex, det_tokens, sample_params):

        self.ex = ex

        group_tokens = add_holes(self.ex, det_tokens, sample_params['num_holes'])
        
        self.params = {}
        self.holes = {}
        self.skip_params = set()
        
        self.struct_prog_tokens = [ex.START_TOKEN]
        self.tokens = []

        self.prev_params = {}
        assert group_tokens.pop(0) == ex.START_TOKEN

        self.COUNTS = {
            'params': 0,
            'hole': 0
        }
        
        self.parse_group_tokens(
            ex.START_TOKEN,
            group_tokens
        )        
        
    def parse_group_tokens(self, token, expr):
        if token == self.ex.HOLE_TOKEN:
            token = f'{self.ex.STRUCT_LOC_TOKEN}_{self.COUNTS["hole"]}'
            self.COUNTS['hole'] += 1
            sc = StructContext(deepcopy(self.tokens))
            self.ex.build_struct_scope(sc, token)
            self.holes[token] = sc
            
        self.tokens.append(token)
    
        for ppos, inp in enumerate(self.ex.TLang.get_inp_types(token)):
            if inp in self.ex.DEF_STRUCT_TYPES:
                nt = expr.pop(0)
                self.struct_prog_tokens.append(nt)
                self.parse_group_tokens(nt, expr)

            else:
                ntoken = f'{self.ex.BLANK_TOKEN}_{inp}_{self.COUNTS["params"]}'
                self.COUNTS['params'] += 1

                gt = expr.pop(0)                

                self.params[ntoken] =\
                    ParamContext(
                        ex=self.ex,
                        name=ntoken,
                        gt=gt,
                        param_type=inp,
                        parent_fn=token,
                        param_pos = ppos,
                        prev_params=self.prev_params[inp] \
                        if inp in self.prev_params else [],
                        skip_params=self.skip_params
                    )
                
                self.tokens.append(ntoken)

                ptype = self.ex.TLang.get_out_type(gt)

                if inp not in self.prev_params:
                    self.prev_params[inp] = []
                
                self.prev_params[inp].append(ntoken)
                

    def get_struct_prog_tokens(self):
        return self.struct_prog_tokens

        
    def get_relations(self):
        return {k:v.relation for k,v in self.params.items() if v.relation is not None}

                
    def sample(self):
        return Group(
            self.ex,
            deepcopy(self.tokens),
            self.params,
            self.holes,
        )        


class ParamContext:

    def __init__(
        self, ex, name, gt, param_type, parent_fn, param_pos, prev_params, skip_params
    ):
        self.ex = ex
        self.name = name
        self.param_type = param_type
        self.last_fn = parent_fn
        self.param_pos = param_pos

        self.relation = None

        if self.param_type in self.ex.SPEC_GROUP_SAMPLE_TYPES:
            self.sample_fn = self.ex.make_spec_group_sampler(
                self, gt, param_type, parent_fn, param_pos, prev_params, skip_params
            )
        
        elif self.param_type == 'float' or \
             self.param_type in self.ex.FLOAT_PARAM_TYPES:

            self.sample_fn = self.ex.make_group_float_sampler(
                gt, param_type, parent_fn, param_pos
            )

        else:
            self.sample_cat_relation(gt, param_type, prev_params, skip_params)
            self.sample_fn = None
            
        if self.relation is not None:
            skip_params.add(self.name)
        
    def sample_cat_relation(self, gt, param_type, prev_params, skip_params):

        tp = self.ex.sample_cat_rel_for_pt(param_type)

        if tp == 'indep':
            self.relation = None
            
        elif tp == 'static':
            self.relation = ('static', gt)
                        
        elif tp == 'reuse':
            options = [p for p in prev_params if p not in skip_params]
            if len(options) == 0:
                self.relation = None
            else:
                self.relation = ('reuse', random.choice(options))
        else:
            assert False
    
    def sample_param(self, tokens):

        if self.sample_fn is not None:
            return self.sample_fn()
        
        elif self.relation is None:
            val = self.ex.TLang.group_sample_param_val(
                self.param_type,
                self.last_fn,
                self.param_pos,
                tokens,                
            )            
            return val
        elif self.relation[0] == 'static':
            return self.relation[1]
            
        elif self.relation[0] == 'reuse':
            return self.context_vals[self.relation[1]]
        else:
            assert False


class StructContext:
    def __init__(self, tokens):
        self.tokens = tokens
    
    def sample_hole(self, ex, counts):
        hs_tokens, hs_params, hs_struct = ex.group_sample_sub_prog(
            self.struct_scope, counts
        )
        return hs_tokens, hs_params, hs_struct
    

    
    
class Group:
    def __init__(
        self,  ex, struct_tokens, _params, _holes
    ):
        
        params = {}
        holes = {}
        params.update(_params)
        holes.update(_holes)
        
        self.ex = ex
        self.struct_derivs = {}

        tokens = []
        param_vals = {}
        context_vals = {}
        self.bck_map = {}

        COUNTS = {"hpcnt": 0}
        PARAM_LOC_COUNT = 0

        hpcnt = 0
        
        while len(struct_tokens) > 0:

            t = struct_tokens.pop(0)
                                
            if t in holes:
                
                hs_tokens, hs_params, hs_struct = holes[t].sample_hole(self.ex, COUNTS)
                                                
                self.struct_derivs[t] = hs_struct
                
                for k,v in hs_params.items():
                    tkn = f'{self.ex.PARAM_LOC_TOKEN}_{PARAM_LOC_COUNT}'
                    param_vals[tkn] = v
                    PARAM_LOC_COUNT += 1
                    self.bck_map[k] = tkn

                fhs_tokens = []
                for t in hs_tokens:
                    if t in self.bck_map:
                        fhs_tokens.append(self.bck_map[t])
                    else:
                        fhs_tokens.append(t)
                        
                tokens += fhs_tokens
                    
                tkn = None
                continue
            
            elif t in params:
                params[t].context_vals = context_vals
                val = params[t].sample_param(                    
                    tokens
                )

                tkn = f'{self.ex.PARAM_LOC_TOKEN}_{PARAM_LOC_COUNT}'
                param_vals[tkn] = val
                PARAM_LOC_COUNT += 1
                assert t not in context_vals
                context_vals[t] = val
                self.bck_map[t] = tkn
                
            else:
                tkn = t

            assert tkn is not None
            
            tokens.append(tkn)                
            
        self.tokens = tokens
        self.param_vals = param_vals
        
        self.prog = ' '.join([
            self.param_vals[t] if t in self.param_vals else t for t in self.tokens
        ])

        
    def ex_prog(self):
        self.P = self.ex.ex_prog(self.prog.split())
        
    def get_sig(self):
        return self.P.get_state_sig()

    def is_valid(self):
        return self.ex.check_valid_prog(self.P)

    


    

def remove_params(tokens, ex):

    struct = []

    PARAM_TYPES = ex.DEF_PARAM_TYPES
    
    for t in tokens:
        if ex.TLang.get_out_type(t) in PARAM_TYPES:
            continue
        struct.append(t)

    return struct


def fairly_sample_ind(ex, tokens):

    m = []
    w = []

    for st in range(len(tokens)):
        if (ex.TLang.get_num_inp(tokens[st]) == 0) or \
           tokens[st] in ex.SKIP_HOLE_TOKENS:
            m.append(0.)
            w.append(1.)
            continue
        else:
            m.append(1.)
            
        c = ex.TLang.get_num_inp(tokens[st])
        i = 0
        pen = 0
        start_pen = 10. if st == 0 else 0.
        while c > 0:
            i += 1
            c += ex.TLang.get_num_inp(tokens[i+st]) - 1
            if tokens[i+st] == ex.HOLE_TOKEN:
                pen += 10.

        
        w.append((i * 0.1 ) + 1. + pen + start_pen)
        
    P = 1/np.array(w)

    P = (P * m) + 1e-8
    P /= P.sum()
    
    ind = np.random.choice(range(len(tokens)), p=P)
    
    return ind


def add_hole(ex, tokens):

    if len(tokens) == 1:
        return tokens
    
    ind = fairly_sample_ind(ex, tokens)

    new_tokens = tokens[:ind] + [ex.HOLE_TOKEN]
    c = ex.TLang.get_num_inp(tokens[ind])
    
    ind += 1
    
    while c > 0:
        c -= 1        
        ntoken = tokens[ind]
        c += ex.TLang.get_num_inp(ntoken)
        ind += 1
        
    new_tokens += tokens[ind:]

    return new_tokens


def add_holes(ex, ftokens, num_holes):

    tokens = ftokens[1:]

    for _ in range(num_holes):
        tokens = add_hole(ex, tokens)
        
    return [ftokens[0]] + tokens


