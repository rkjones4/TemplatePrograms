import os
from tqdm import tqdm
import torch
import numpy as np
import json
import torch
import random
import sys
sys.path.append('../../executors/layout/')
sys.path.append('../../executors/common/')
#sys.path.append('../../domains/')
#sys.path.append('../../')

import ex_layout as ex_lay

ex = ex_lay.LayExecutor(
    {
        'MAX_PARAM_TOKENS': 128
    }
)

def r2v(v):
    rv = ex.D_FLTS[(ex.TD_FLTS - v).abs().argmin()]
    return str(rv)
    
def sample_part_infos(part_list, shared_vars):
    samples = {}
    
    for part in part_list:
        sp = part.sample(samples, shared_vars)
        sp.name = part.name
        samples[part.name] = sp

    return [s for s in samples.values() if s.prim_type is not None]
                

def get_op_params(op, v1, v2, K=None):
    if op == 'move':
        params = [
            ex.find_closest_float_val(v1, 'mxtype'),
            ex.find_closest_float_val(v2, 'mytype'),
        ]
        move_tokens = ex.make_move_tokens(params)        
        return move_tokens

    elif op == 'scale':
        params = [
            ex.find_closest_float_val(v1, 'swtype'),
            ex.find_closest_float_val(v2, 'shtype'),
        ]
        scale_tokens = ex.make_scale_tokens(params)        
        return scale_tokens

    elif op == 'symTranslate':
        params = [
            ex.find_closest_float_val(v1, 'mxtype'),
            ex.find_closest_float_val(v2, 'mytype'),
        ]
        sym_tokens = ex.make_symtranslate_tokens(params + [K])
        return sym_tokens

    else:
        assert False
    
class SamplePart:
    def __init__(
        self,       
    ):
        self.prim_type = None
        self.width = None
        self.height = None
        self.x_pos = None
        self.y_pos = None
        self.color = None
        self.sem_class = None
        self.top_tokens = None
        
    def get_tokens(self):
        tokens = []
        if self.color != 'grey':
            assert self.color in ex.D_COLORS
            tokens += ['color', self.color]

        if self.x_pos != 0. or self.y_pos != 0.:

            mparams = get_op_params('move', self.x_pos, self.y_pos)            
            tokens += mparams

        assert self.width < 1.0 and self.width > 0.
        assert self.height < 1.0 and self.height > 0.
            
        tokens += get_op_params('scale', self.width * 2, self.height * 2)

        assert self.prim_type in ex.D_PRIMS
        
        tokens += ['prim', self.prim_type]

        if self.top_tokens is not None:

            if 'symTranslate' in self.top_tokens:
                symparams = get_op_params(
                    'symTranslate',
                    self.top_tokens[1],
                    self.top_tokens[2],
                    self.top_tokens[3]
                )
                return symparams + tokens
            else:
                return list(self.top_tokens) + tokens
        
        return tokens
        
    def get_info(self):
        def rnd(v):
            return round(v,2)

        assert self.sem_class is not None
        
        return (self.prim_type,
               rnd(self.width), rnd(self.height),
               rnd(self.x_pos), rnd(self.y_pos),
                self.color, self.top_tokens, self.sem_class)
        
def get_static_val(logic, pname, pvals, psamps, svars):
    assert len(logic) == 1
    return logic[0]

def get_ref_val(logic, pname, pvals, psamps, svars):
    assert len(logic) == 2
    ref_name, ref_att = logic

    if ref_name == 'share':
        v = svars[ref_att]
    elif ref_name in psamps:
        v = getattr(psamps[ref_name], ref_att)
    else:
        assert pname == ref_name
        v = getattr(pvals, ref_att)

    return v

def _get_uni_val(a, b):
    p = random.random()
    return (a * p) + (b * (1-p))

def get_uni_val(logic, pname, pvals, psamps, svars):
    assert len(logic) == 2    
    a,b = logic
    return _get_uni_val(a, b)

def get_att(O, A, K):
    if A == 'center':
        assert K is False
        return 0.

    if A == 'center_width':
        assert K is True
        return O.x_pos
    
    elif A == 'center_height':
        assert K is True
        return O.y_pos
    
    elif A == 'bot':
        v = -1 * (O.height)
        if not K:
            return v
        else:
            cv = O.y_pos + v
            return cv
            
    elif A == 'top':
        v = 1 * (O.height)
        if not K:
            return v
        else:
            cv = O.y_pos + v
            return cv
        
    elif A == 'left':
        v = -1 * (O.width)
        if not K:
            return v
        else:
            cv = O.x_pos + v
            return cv
        
    elif A == 'right':
        v = 1 * (O.width)
        if not K:
            return v
        else:
            cv = O.x_pos + v
            return cv
        
    else:
        assert False, f'impl att {A}'
            

def calc_rel_pos(L, R, D):
    LO, LA = L
    RO, RA = R

    LV = get_att(LO, LA, False)
    RV = get_att(RO, RA, True)

    v = RV - (LV * D)
    
    return v

def get_rel_val(logic, pname, pvals, psamps, svars):
    assert len(logic) == 2
    return calc_rel_pos(
        (pvals, logic[0]),
        (psamps[logic[1][0]], logic[1][1]),
        1.0
    )
    
def get_prel_val(logic, pname, pvals, psamps, svars):

    assert len(logic) == 3
    rel_val = fn_map[logic[2][0]](logic[2][1:], pname, pvals, psamps, svars)
    return calc_rel_pos(
        (pvals, logic[0]),
        (psamps[logic[1][0]], logic[1][1]),
        rel_val
    )

def get_expr_val(logic, pname, pvals, psamps, svars):
    assert len(logic) == 3

    lhs = fn_map[logic[1][0]](logic[1][1:], pname, pvals, psamps, svars)
    rhs = fn_map[logic[2][0]](logic[2][1:], pname, pvals, psamps, svars)

    op = logic[0]

    if op == 'div':
        return lhs / (rhs+1e-8)

    elif op == 'add':
        return lhs + rhs

    elif op == 'mul':
        return lhs * rhs

    elif op == 'sub':
        return lhs - rhs
    
    else:
        assert False, f'impl {op}'


fn_map = {
    'static': get_static_val,
    'ref': get_ref_val,
    'uni': get_uni_val,
    'rel': get_rel_val,
    'expr': get_expr_val,
    'prel': get_prel_val
}
        
class Part:
    def __init__(
        self,
        name,
        prim_info,
        size_info,
        loc_info,
        color_info,
        sem_info,
        top_info=None,
        part_group=None    
    ):
        self.name = name
        self.prim_info = prim_info
        self.size_info = size_info
        self.loc_info = loc_info
        self.color_info = color_info
        self.sem_info = sem_info
        self.top_info = top_info
        self.part_group = part_group
        
    def get_val(self, logic, SP, samples, shared_vars):
        assert isinstance(logic, tuple)

        ltype = logic[0]
                
        assert ltype in fn_map

        v = fn_map[ltype](logic[1:], self.name, SP, samples, shared_vars)

        return v
        
    def sample(self, samples, shared_vars):
        SP = SamplePart()
        SP.part_group = self.part_group
        
        self.sample_prim_info(SP, samples, shared_vars)                
        self.sample_size_info(SP, samples, shared_vars)
        self.sample_loc_info(SP, samples, shared_vars)
        self.sample_color_info(SP, samples, shared_vars)

        self.sample_top_info(SP, samples, shared_vars)

        SP.sem_class = self.S2I[self.sem_info]
        
        return SP

    def sample_prim_info(self, SP, samples, shared_vars):
        SP.prim_type = self.get_val(self.prim_info, SP, samples, shared_vars)

    def sample_size_info(self, SP, samples, shared_vars):
        SP.width = self.get_val(self.size_info[0], SP, samples, shared_vars)
        SP.height = self.get_val(self.size_info[1], SP, samples, shared_vars)
        
    def sample_loc_info(self, SP, samples, shared_vars):
        SP.x_pos = self.get_val(self.loc_info[0], SP, samples, shared_vars)
        SP.y_pos = self.get_val(self.loc_info[1], SP, samples, shared_vars)

    def sample_color_info(self, SP, samples, shared_vars):
        SP.color = self.get_val(self.color_info, SP, samples, shared_vars)

    def sample_top_info(self, SP, samples, shared_vars):
        SP.top_tokens = get_top_val(self.top_info, SP, samples, shared_vars)

def get_top_val(info, SP, samples, shared_vars):
    if info is None:
        return None

    assert len(info) == 2

    if info[0] == 'static':
        return info[1]

    if info[0] == 'fn':
        return info[1](SP, samples, shared_vars)

    assert False, f'bad info {info}'

class Concept:
    def __init__(
        self,
        name,
        cat_vars,
        sample_var_fn,
    ):
        self.name = name
        self.cat_vars = cat_vars
        self.sample_var_fn = sample_var_fn
                
        self.parts = []
        self.S2I = {}

        self.pgl = {}
        
    def add_part(
        self,
        part
    ):

        if part.sem_info not in self.S2I:
            self.S2I[part.sem_info] = len(self.S2I)
        part.S2I = self.S2I
        
        self.parts.append(part)

    def get_base_types(self):

        dopts = []
        
        for att, att_opts in self.cat_vars.items():
            o1 = f'{att}:{att_opts[0]}'
            o2 = f'{att}:{att_opts[1]}'
            dopts.append((o1, o2))

        btypes = [','.join([a for a,_ in dopts])]

        for i in range(len(dopts)):
            nopts = [b if j==i else a for j,(a,b) in enumerate(dopts)]
            btypes.append(
                ','.join(nopts)
            )
                        
        return btypes

    def get_all_types(self):

        dopts = []
        
        for att, att_opts in self.cat_vars.items():
            opts = []
            for ao in att_opts:
                opts.append(
                    f'{att}:{ao}'
                )                
            dopts.append(tuple(opts))

        atypes = [None]

        for i in range(len(dopts)):
            ptypes = atypes
            atypes = []

            for ai in dopts[i]:
                for pt in ptypes:
                    if pt is not None:
                        atypes.append(f'{pt},{ai}')
                    else:
                        atypes.append(ai)
                                    
        return atypes
        
    def sample(self, char_type, ret_info=False):
        assert char_type is not None        

        cvars = {}

        for ai in char_type.split(','):
            att, att_opt = ai.split(':')
            assert att in self.cat_vars
            assert att_opt in self.cat_vars[att]
            cvars[att] = att_opt

        assert len(cvars) == len(self.cat_vars)

        svars = self.sample_var_fn(cvars)
        
        sample = sample_part_infos(self.parts, svars)        
        tokens = self.conv_sample_to_tokens(sample, svars)

        if ret_info:
            infos = [s.get_info() for s in sample]
            return tokens, infos
        
        return tokens

    def conv_sample_to_tokens(self, sample, svars):
        if len(self.pgl) == 0:
            return self.simple_conv_sample_to_tokens(sample)
        else:
            return self.pgl_conv_sample_to_tokens(sample, svars)
    
    def simple_conv_sample_to_tokens(self, sample):
        sub_progs = []
        for s in sample:
            sub_progs.append(s.get_tokens())

        return ex_lay.comb_sub_progs(sub_progs)

    def pgl_conv_sample_to_tokens(self, sample, svars):
        
        sub_groups = {'def': []}

        for s in sample:
            if s.part_group is not None:
                if s.part_group not in sub_groups:
                    sub_groups[s.part_group] = []
                sub_groups[s.part_group].append(s.get_tokens())
            else:
                sub_groups['def'].append(s.get_tokens())

        pgl_sub_groups = []

        for k, V in sub_groups.items():
            if len(V) == 0:
                continue
            cv = _comb_sub_progs(V)
            if k in self.pgl:
                pgl_sub_groups.append(self.pgl[k](svars) + cv)
            else:
                pgl_sub_groups.append(cv)

        P = comb_sub_progs(pgl_sub_groups)

        return P
        
    def add_part_group_logic(self, k,v):
        self.pgl[k] = v
        

def comb_sub_progs(sub_shapes):
    return ['START'] + _comb_sub_progs(sub_shapes)

def _comb_sub_progs(ss):
    if len(ss) == 0:
        return []
    
    if len(ss) == 1:
        return ss[0]
    
    return ['union'] + ss[0]  + _comb_sub_progs(ss[1:])    
        
def make_concept_map():

    import org_concepts as oc
    import man_concepts as mc
    
    CONCEPT_MAP = {
        # Organic
        'fish': oc.make_fish,
        'person': oc.make_person,
        'caterpillar': oc.make_caterpillar,
        'flower': oc.make_flower,
        'mushroom': oc.make_mushroom,
        'crab': oc.make_crab,
        'cat': oc.make_cat,
        'turtle': oc.make_turtle,
        'mouse': oc.make_mouse,
        'ladybug': oc.make_ladybug,

        # Manufactured
        'fridge': mc.make_fridge,
        'microwave': mc.make_microwave,
        'clock': mc.make_clock,
        'car': mc.make_car,
        'plane': mc.make_plane,
        'horiz_back': mc.make_horiz_back,
        'house': mc.make_house,        
        'side_chair': mc.make_side_chair, 
        'table': mc.make_table, 
        'bookshelf': mc.make_bookshelf,
    }

    return CONCEPT_MAP

def save_data(out_name, num_to_write):

    CONCEPT_MAP = make_concept_map()
    
    D = []
    TI = {}

    print(f"Saving to {out_name}")
    
    os.system(f'mkdir {out_name}')
    os.system(f'mkdir {out_name}/vis')
    
    seen = set()
    for name, concept_fn in CONCEPT_MAP.items():
        print(f"~~{name}~~")
        
        concept = concept_fn()
        
        all_types = concept.get_all_types()
        TI[name] = all_types
        
        for type_num, atype in tqdm(list(enumerate(all_types))):
            count = 0
            LV = []
            for j in range(10000):
                if count >= num_to_write:
                    break

                tokens, infos = concept.sample(atype, ret_info=True)

                P = ex_lay.Program(ex)

                try:
                    P.run(tokens)
                except Exception as e:
                    print(e)
                    continue
                
                img = ex.check_valid_prog(P, ret_vdata=True)

                if img is None:
                    continue

                sig = P.get_state_sig()

                if sig in seen:
                    continue

                seen.add(sig)
                D.append((name, atype, count, tokens, infos))
                if count < 10:
                    LV.append(img)
                count += 1
                
            # Save
            atypename = f'{out_name}/vis/{name}_{type_num}_{atype}'
            ex.render_group(LV, rows = 2, name = atypename)
            
    J = {
        'names': [],
        'tokens': [],
        'infos': []
    }

    print(f"Loaded {len(D)} examples")
    
    for nconcept, ntype, ncount, tokens, infos in D:
        name = f'{nconcept}_{ntype}_{ncount}'
        J['names'].append(name)
        J['tokens'].append(tokens)
        J['infos'].append(infos)

    json.dump(J, open(f'{out_name}/data.json', 'w'))
        
if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    out_name = sys.argv[1]
    num = int(sys.argv[2])

    with torch.no_grad():
        save_data(out_name, num)
        
