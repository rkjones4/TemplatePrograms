import torch
from copy import deepcopy

class SampleData:
    def __init__(self, sampler, valid, dummy=False):
        if dummy:
            self.sampler = None
            self.valid = None
            self.rels = None
            return
        
        self.sampler = sampler
        self.valid = valid
        self.rels = sampler.get_relations()

    def get_images(self):
        return [v.P.make_image().cpu() for v in self.valid]

    def make_cs_new_struct_info(self, ex, args):

        struct_prog_tokens = []

        static_tokens = {}
        shared_tokens = {}
        vars_to_shared = {}

        self_ref_rels = {}
        
        for relt, VALUE in self.rels.items():
            if len(VALUE) != 2:
                continue

            relation, reft = VALUE

            if 'float' in relt:
                continue

            if relation == 'reuse' and reft not in self.rels:                
                self_ref_rels[reft] = (relation, reft)

        self.rels.update(self_ref_rels)
        
        for t in self.sampler.tokens:
            if ex.BLANK_TOKEN in t:

                if 'float' in t:
                    continue

                elif t in self.rels:
                    rel = self.rels[t]
                                        
                    if rel[0] == 'static':
                        struct_prog_tokens.append(rel[1])
                        static_tokens[t] = rel[1]
                        
                    elif rel[0] == 'reuse':

                        rt = rel[1]
                        
                        if rt not in shared_tokens:
                            shared_tokens[rt] = f'{ex.SHARED_TOKEN}_{len(shared_tokens)}'

                        vars_to_shared[t] = shared_tokens[rt]
                        vars_to_shared[rt] = shared_tokens[rt]
                        struct_prog_tokens.append(shared_tokens[rt])
                        
            elif ex.STRUCT_LOC_TOKEN in t:
                struct_prog_tokens.append(ex.HOLE_TOKEN)

            else:
                struct_prog_tokens.append(t)

        struct_prog_tokens.append(ex.START_TOKEN)
                
        pre_struct_deriv_tokens = []
        pre_struct_deriv_weight = []

        hole_info = []
        
        for t in struct_prog_tokens:
            if t == ex.HOLE_TOKEN:
                ht = f'{ex.STRUCT_LOC_TOKEN}_{len(hole_info)}'
                hole_info.append(ht)
                pre_struct_deriv_tokens.append(ht)
            else:
                pre_struct_deriv_tokens.append(t)
                
            pre_struct_deriv_weight.append(0)
        
        assert len(pre_struct_deriv_tokens) == len(struct_prog_tokens)
        assert len(pre_struct_deriv_tokens) == len(pre_struct_deriv_weight)
        
        struct_deriv_info = []
        
        for v in self.valid:
            struct_deriv_tokens = deepcopy(pre_struct_deriv_tokens)
            struct_deriv_weight = deepcopy(pre_struct_deriv_weight)

            for hi in hole_info:
                struct_deriv_tokens.append(hi)
                struct_deriv_weight.append(0.)

                tkns = v.struct_derivs[hi]

                struct_deriv_tokens += tkns
                struct_deriv_weight += [1. for _ in tkns]

            assert len(struct_deriv_tokens) == len(struct_deriv_weight)
            struct_deriv_info.append((struct_deriv_tokens, struct_deriv_weight))
            
        return struct_prog_tokens, struct_deriv_info, shared_tokens, static_tokens, vars_to_shared
    
    def make_prob_prog_batch(self, ex, args):

        struct_prog_tokens, struct_deriv_info, shared_tokens, static_tokens, vars_to_shared = self.make_cs_new_struct_info(ex, args)
        
        param_deriv_info = []
        
        for v in self.valid:
            
            rbm = {v:k for k,v in v.bck_map.items()}
            seen_shared = set()
            
            hole_info = []
            
            param_deriv_tokens = []
            param_deriv_weight = []
            
            for t in v.tokens:

                if t in rbm and rbm[t] in static_tokens:
                    param_deriv_tokens.append(static_tokens[rbm[t]])
                    param_deriv_weight.append(0.)

                elif t in rbm and rbm[t] in vars_to_shared:
                    param_deriv_tokens.append(vars_to_shared[rbm[t]])
                    param_deriv_weight.append(0.)

                    if vars_to_shared[rbm[t]] not in seen_shared:
                        seen_shared.add(vars_to_shared[rbm[t]])
                        ht = f'{ex.PARAM_LOC_TOKEN}_{len(hole_info)}'
                        param_deriv_tokens.append(ht)
                        param_deriv_weight.append(0.)
                        hole_info.append((ht, t))
                        
                elif ex.PARAM_LOC_TOKEN in t:

                    ht = f'{ex.PARAM_LOC_TOKEN}_{len(hole_info)}'
                    param_deriv_tokens.append(ht)
                    param_deriv_weight.append(0.)
                    hole_info.append((ht, t))
                                                
                else:
                    param_deriv_tokens.append(t)
                    param_deriv_weight.append(0.)

            # everything seen atleast once
            assert len(seen_shared) == len(shared_tokens)
            
            param_deriv_tokens.append(ex.START_TOKEN)
            param_deriv_weight.append(0.)

            for ht, hi in hole_info:
                param_deriv_tokens.append(ht)
                param_deriv_weight.append(0.)

                param_deriv_tokens.append(v.param_vals[hi])
                param_deriv_weight.append(1.)
                
            assert len(param_deriv_tokens) == len(param_deriv_weight)
            
            param_deriv_info.append((param_deriv_tokens, param_deriv_weight))
            
            assert v.prog == ex.cs_instantiate(param_deriv_tokens)

        return struct_prog_tokens, struct_deriv_info, param_deriv_info

def cs_instantiate(ex, arpp_prog, rci):
    
    prog = [arpp_prog[0]]

    sst = {}
    Osst = {}
    im = {}
    pm = {}

    last = None
    cur = []

    cut_ind = None
    
    for t in arpp_prog[1:]:
        if t == ex.START_TOKEN:
            last = "START"
            continue

        if last is None:

            if ex.SHARED_TOKEN in t:

                if t not in sst:
                    sst[t] = len(prog)
                    continue

                else:
                    Osst[len(prog)] = sst[t]
                    
            
            if ex.PARAM_LOC_TOKEN in t:
                im[t] = len(prog)
                    
            prog.append(t)

        else:
            if ex.PARAM_LOC_TOKEN in t:
                if len(cur) > 0:
                    pm[last] = ' '.join(cur)                    
                    cur = []
                last = t
            else:
                cur.append(t)

        _lt = t

        
    if len(cur) > 0:
        pm[last] = ' '.join(cur)
       
    for pln, pli in im.items():
        prog[pli] = pm[pln]

    for c,o in Osst.items():
        prog[c] = prog[o]
                
    if rci:
        return prog, list(pm.values())
        
    return ' '.join(prog)



def make_batch(ex, data, args):

    B = {
        
        'struct_seq': torch.zeros(
            len(data),
            args.max_struct_tokens,
        ).long(),
        'struct_seq_weight': torch.zeros(
            len(data),
            args.max_struct_tokens,
        ).float(),

        'deriv_seq': torch.zeros(
            len(data),
            args.max_vis_inputs,
            args.max_struct_tokens + args.max_deriv_tokens,
        ).long(), 
        'deriv_seq_weight': torch.zeros(
            len(data),
            args.max_vis_inputs,
            args.max_struct_tokens + args.max_deriv_tokens,
        ).float(),

        'param_seq': torch.zeros(
            len(data),
            args.max_vis_inputs,
            args.max_struct_tokens + args.max_deriv_tokens + args.max_param_tokens + args.max_param_tokens,
        ).long(),
        'param_seq_weight': torch.zeros(
            len(data),
            args.max_vis_inputs,
            args.max_struct_tokens + args.max_deriv_tokens + args.max_param_tokens + args.max_param_tokens,
        ).float(),
        
        'vdata': torch.zeros(
            tuple([len(data), args.max_vis_inputs] + ex.get_input_shape())
        ).float()
    }

    for i,d in enumerate(data):

        struct_prog_tokens, struct_deriv_info, param_deriv_info = d.make_prob_prog_batch(ex, args)
        
        skip = ''

        if len(struct_prog_tokens) > args.max_struct_tokens:
            skip += 'struct_len'

        for (sd, _), (pd, _) in zip(struct_deriv_info, param_deriv_info):
            if len(sd) > args.max_struct_tokens + args.max_deriv_tokens:
                skip += 'Sderiv_len'

            if len(pd) > args.max_struct_tokens + args.max_deriv_tokens + args.max_param_tokens + args.max_param_tokens:
                skip += 'Pderiv_len'                    

        if skip:
            continue

    
        if ex.ex_name == 'shape':
            if 'vis_vdata' not in B:
                B['vis_vdata'] = []
            scenes = d.get_images()        
            images = [ex.conv_scene_to_vinput(scene) for scene in scenes]
            B['vis_vdata'].append(torch.stack([s.cpu() for s in scenes],dim=0))
        else:
            images = d.get_images()
            
        struct_prog = ex.TLang.tokens_to_tensor(struct_prog_tokens)

        B['struct_seq'][i,:struct_prog.shape[0]] = struct_prog
        B['struct_seq_weight'][i,:struct_prog.shape[0]] = 1.

        for j, ((sd, sw), (pd, pw), img) in enumerate(
            zip(struct_deriv_info, param_deriv_info, images)
        ):

            sdT = ex.TLang.tokens_to_tensor(sd)                        
            swT = torch.tensor(sw).float()

            pdT = ex.TLang.tokens_to_tensor(pd)
            pwT = torch.tensor(pw).float()
            
            B['deriv_seq'][i,j,:sdT.shape[0]] = sdT
            B['deriv_seq_weight'][i,j,:sdT.shape[0]] = swT

            B['param_seq'][i,j,:pdT.shape[0]] = pdT
            B['param_seq_weight'][i,j,:pdT.shape[0]] = pwT

            try:
                B['vdata'][i,j] = img
            except:
                assert len(img.shape) == 2
                img = img.unsqueeze(-1)
                B['vdata'][i,j] = img
            
    if 'vis_vdata' in B:
        B['vis_vdata'] = torch.stack(B['vis_vdata'],dim=0)
                
    return B
    
def conv_struct_out_to_deriv_inp(ex, tokens):

    hole_inds = (tokens == ex.TLang.T2I[ex.HOLE_TOKEN]).nonzero().flatten()

    lt = ex.TLang.T2I[f'{ex.STRUCT_LOC_TOKEN}_0']
    
    loc_vals = torch.clamp(torch.arange(
        hole_inds.shape[0], device=hole_inds.device
    ), 0, ex.MAX_HOLES-1) + lt
    
    tokens[hole_inds] = loc_vals

    if hole_inds.shape[0] > 0:    
        out = torch.cat((
            tokens,
            torch.ones(1,device=hole_inds.device).long() * lt
        ),dim=0)
    else:
        out = tokens
    
    return out, hole_inds.shape[0]

def conv_deriv_out_to_param_inp(ex, tokens):
    
    try:
        ind = (tokens[1:] == ex.TLang.T2I[ex.START_TOKEN]).nonzero().item()
    except:
        return None, None
    
    prog = [ex.TLang.I2T[t.item()] for t in tokens]
    
    deriv_prog = prog[ind+2:]
    
    pm = {}
    cur = []
    last = None
    
    for t in deriv_prog:
        if ex.STRUCT_LOC_TOKEN in t or ex.PARAM_LOC_TOKEN in t:
            if len(cur) > 0:
                pm[last] = cur
                cur = []
            last = t            
        else:
            cur.append(t)

    if len(cur) > 0:
        pm[last] = cur

    if ex.ex_name == 'shape':
        struct_tokens = ex.reformat_struct(prog[:ind+1]) + ['START']
    else:
        struct_tokens = prog[:ind+2]
        
    hole_map = pm
    
    param_inp = []
    COUNTS = {
        'params': 0
    }
    seen_shared = set()

    d2p_helper(
        ex, struct_tokens.pop(0), struct_tokens, COUNTS, param_inp, seen_shared, hole_map
    )

    assert len(struct_tokens) == 1
    
    param_inp += struct_tokens
    
    param_inp.append(f'{ex.PARAM_LOC_TOKEN}_0')
        
    param_inp = ex.TLang.tokens_to_tensor(param_inp)
    
    return param_inp, COUNTS['params']


def d2p_helper(
    ex, token, struct_tokens, COUNTS, param_inp, seen_shared, hole_map
):
    
    if ex.STRUCT_LOC_TOKEN in token or ex.PARAM_LOC_TOKEN in token:
        if token not in hole_map:
            return None

        while len(hole_map[token]) > 0:
            struct_tokens.insert(0, hole_map[token].pop(-1))
        
        nt = struct_tokens.pop(0)
        return d2p_helper(ex, nt, struct_tokens, COUNTS, param_inp, seen_shared, hole_map)


    assert ex.SHARED_TOKEN not in token and ex.TLang.get_out_type(token) not in ex.DEF_PARAM_TYPES, f'bad {token}'
    
    param_inp.append(token)

    for inp in ex.TLang.get_inp_types(token):
        
        if inp in ex.DEF_STRUCT_TYPES:
            # recurse
            nt = struct_tokens.pop(0)
            v = d2p_helper(ex, nt, struct_tokens, COUNTS, param_inp, seen_shared, hole_map)
            if v is None:
                return None
                        
        else:
            
            if inp in ex.DEF_PARAM_TYPES and ex.SHARED_TOKEN in struct_tokens[0]:
                nt = struct_tokens.pop(0)
                param_inp.append(nt)
                if not nt in seen_shared:
                    param_inp.append(f'{ex.PARAM_LOC_TOKEN}_{COUNTS["params"]}')
                    COUNTS["params"] += 1
                    seen_shared.add(nt)

            elif inp in ex.DEF_PARAM_TYPES and ex.TLang.get_out_type(struct_tokens[0]) in ex.DEF_PARAM_TYPES and\
                 ex.TLang.get_out_type(struct_tokens[0]) == inp:
                nt = struct_tokens.pop(0)
                param_inp.append(nt)

            else:
                param_inp.append(f'{ex.PARAM_LOC_TOKEN}_{COUNTS["params"]}')
                COUNTS["params"] += 1
                
    return True

def find_deriv(ex, expr, struct):

    if ex.ex_name == 'shape':
        return ex.find_deriv(expr, struct)
    
    assert ex.ex_name != 'shape'
    
    if ex.STRUCT_LOC_TOKEN not in ' '.join(struct):
        return []

    def _find_deriv(q):
        ipc = 1
        dv = []
        while(ipc > 0):
            ipc -= 1
            t = q.pop(0)
            if ex.TLang.get_out_type(t) in ex.DEF_STRUCT_TYPES:
                dv.append(t)
            ipc += ex.TLang.get_num_inp(t)

        return ' '.join(dv)
    
    derivs = []

    qs = deepcopy(struct)
    qe = expr.split()
    
    while len(qs) > 0:
        st = qs.pop(0)
        
        while len(qs) > 0 and ((ex.TLang.get_out_type(st) in ex.DEF_PARAM_TYPES) or ex.SHARED_TOKEN in st):

            st = qs.pop(0)
            if ex.TLang.get_out_type(st) in ex.DEF_PARAM_TYPES and \
               ex.TLang.get_out_type(qe[0]) in ex.DEF_PARAM_TYPES:
                qe.pop(0)
                
        if len(qs) == 0 and len(qe) == 0:
            break
                
        if ex.STRUCT_LOC_TOKEN in st:
            dv = _find_deriv(qe)            
            derivs.append(dv)
            while len(qe) > 0 and ex.TLang.get_out_type(qe[0]) in ex.DEF_PARAM_TYPES:
                et = qe.pop(0)
            
        else:
            et = qe.pop(0)
            if et != st:
                assert False
                
            while len(qe) > 0 and ex.TLang.get_out_type(qe[0]) in ex.DEF_PARAM_TYPES:
                et = qe.pop(0)
                
    return derivs
