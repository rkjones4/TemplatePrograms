import tutils as tu
from copy import deepcopy

class Dummy:
    def __init__(self):
        pass

class InferData:
    def __init__(self, ex, infos, args):
        
        self.infos = infos
        
        self.sd = tu.SampleData(None, None, dummy=True)
        self.vout = []

        if ex.ex_name == 'shape':
            self.init_shape_prob_data(ex, infos)
        else:
            self.init_prob_data(ex, infos)

        self.sd.make_prob_prog_batch(ex, args)            
                                    
    def init_shape_prob_data(self, ex, infos):

        assert ex.ex_name == 'shape'
        
        struct_info = infos[0]['struct']

        deriv_infos = [
            ex.find_deriv(_info['expr'], _info['struct'])
            for _info in infos if 'expr' in _info
        ]
            
        valid = []
        
        for _ in range(len(deriv_infos)):
            v = Dummy()
            v.struct_derivs = {}
            valid.append(v)
        
        for t in struct_info:
            if ex.STRUCT_LOC_TOKEN in t:
                for i,v in enumerate(valid):
                    v.struct_derivs[t] = deriv_infos[i].pop(0).split()
                    
        for di in deriv_infos:
            assert len(di) == 0
            

        first_struct = infos[0]['struct']
        first_expr = infos[0]['expr']

        nd_expr = ex.remove_deriv(first_expr, first_struct)        
        
        sampler_tokens, rels = parse_prob_sampler_tokens(
            ex,
            first_struct,
            nd_expr,
        )
        
        self.sd.rels = rels
        self.sd.sampler = Dummy()
        self.sd.sampler.tokens = sampler_tokens
        
        for i in range(len(infos)):
            v = valid[i]
            struct = infos[i]['struct']
            expr = infos[i]['expr']

            rfs = ex.reformat_struct(struct)            
            v.prog = expr

            img = ex.execute(expr)

            self.vout.append(img.cpu().detach())

            prob_make_valid_info(ex, v, rfs, expr)                
            
        self.sd.valid = valid        
        
    def init_prob_data(self, ex, infos):
        
        assert ex.ex_name != 'shape'
        
        struct_info = infos[0]['struct']
        
        deriv_infos = [
            tu.find_deriv(ex, _info['expr'], _info['struct'])
            for _info in infos if 'expr' in _info
        ]
        
        valid = []
        
        for _ in range(len(deriv_infos)):
            v = Dummy()
            v.struct_derivs = {}
            valid.append(v)
        
        for t in struct_info:
            if ex.STRUCT_LOC_TOKEN in t:
                for i,v in enumerate(valid):
                    v.struct_derivs[t] = deriv_infos[i].pop(0).split()

        for di in deriv_infos:
            assert len(di) == 0
            
        sampler_tokens, rels = parse_prob_sampler_tokens(
            ex,
            infos[0]['struct'],
            infos[0]['expr']
        )
        
        self.sd.rels = rels
        self.sd.sampler = Dummy()
        self.sd.sampler.tokens = sampler_tokens
        
        for i in range(len(infos)):
            v = valid[i]
            struct = infos[i]['struct']
            expr = infos[i]['expr']            
            v.prog = expr

            img = ex.execute(expr)

            self.vout.append(img.cpu().detach())
            
            prob_make_valid_info(ex, v, struct, expr)                

        self.sd.valid = valid        
                                        
    def get_images(self):
        return self.vout
        
    def make_prob_prog_batch(self, ex, args):
        return self.sd.make_prob_prog_batch(ex, args)



def parse_prob_sampler_tokens(ex, struct, expr):
        
    sampler_tokens = []
    rels = {}
    
    qs = deepcopy(struct)
    qe = expr.split()
    
    info =  {
        'nbt': 0,
        'shared': {},
        'sampler_tokens': sampler_tokens,
        'rels': rels
    }    
    
    pnst_helper(ex, qs, qe, info)
    
    return sampler_tokens, rels
    
def pnst_rm_deriv(ex, q):
    if ex.ex_name == 'shape':
        assert q[0] == 'hier'
        q.pop(0)
        return
        
    ipc = 1
    while(ipc > 0):
        ipc -= 1
        t = q.pop(0)
        ipc += ex.TLang.get_num_inp(t)

def pnst_helper(ex, qs, qe, info):
    
    if len(qs) == 0 and len(qe) == 0:
        return
    
    st = qs.pop(0)

    if ex.STRUCT_LOC_TOKEN in st:
        info['sampler_tokens'].append(st)
        pnst_rm_deriv(ex, qe)
        return

    assert ex.SHARED_TOKEN not in st \
        and ex.TLang.get_out_type(st) not in ex.DEF_PARAM_TYPES, f'bad {st}'

    info['sampler_tokens'].append(st)
    assert qe.pop(0) == st
    
    for ii, inp in enumerate(ex.TLang.get_inp_types(st)):

        if inp in ex.DEF_STRUCT_TYPES:
            pnst_helper(ex, qs, qe, info)
            
        else:

            if len(qs) > 0 and inp in ex.DEF_PARAM_TYPES and ex.SHARED_TOKEN in qs[0]\
               and inp == ex.TLang.get_out_type(qe[0]):
                
                nt = f'{ex.BLANK_TOKEN}_{inp}_{info["nbt"]}'
                info['nbt'] += 1

                if qs[0] not in info['shared']:
                    info['shared'][qs[0]] = nt

                info['rels'][nt] = ('reuse', info['shared'][qs[0]])

                qs.pop(0)
                qe.pop(0)

                info['sampler_tokens'].append(nt)

            elif len(qs) > 0 and inp in ex.DEF_PARAM_TYPES and ex.TLang.get_out_type(qs[0]) in ex.DEF_PARAM_TYPES\
                 and ex.TLang.get_out_type(qs[0]) == ex.TLang.get_out_type(qe[0]):
                                
                nt = f'{ex.BLANK_TOKEN}_{ex.TLang.get_out_type(qs[0])}_{info["nbt"]}'
                info['nbt'] += 1
                info['rels'][nt] = ('static', qs[0])
                assert qs.pop(0) == qe.pop(0)
                info['sampler_tokens'].append(nt)
                
            else:
                et = qe.pop(0)
                nt = f'{ex.BLANK_TOKEN}_{ex.TLang.get_out_type(et)}_{info["nbt"]}'
                info['nbt'] += 1        
                info['sampler_tokens'].append(nt)


def prob_make_valid_info(ex, v, struct, expr):

    v.bck_map = {}
    v.tokens = []
    v.param_vals = {}
            
    qs = deepcopy(struct)
    qe = expr.split()
    
    info =  {
        'nbt': 0,
        'nplt': 0,
        'nht': 0,
    }    

    nmvi_helper(ex, v, qs, qe, info)

def nmvi_rm_deriv(ex, v, q, info):
    assert ex.ex_name != 'shape'
    ipc = 1

    while(ipc > 0):
        ipc -= 1
        t = q.pop(0)

        if ex.TLang.get_out_type(t) in ex.DEF_STRUCT_TYPES:
            v.tokens.append(t)
        else:
            hole_nt = f'{ex.HOLE_PARAM_TOKEN}_{ex.TLang.get_out_type(t)}_{info["nht"]}'
            ploc_nt = f'{ex.PARAM_LOC_TOKEN}_{info["nplt"]}'
            info['nht'] += 1
            info['nplt'] += 1
            v.tokens.append(ploc_nt)
            v.bck_map[hole_nt] = ploc_nt
            v.param_vals[ploc_nt] = t
        
        ipc += ex.TLang.get_num_inp(t)

def shape_nmvi_rm_deriv(ex, v, q, info):
    assert ex.ex_name == 'shape'


    while True:

        t = q.pop(0)

        if ex.TLang.get_out_type(t) in ex.DEF_STRUCT_TYPES:
            v.tokens.append(t)
        else:
            hole_nt = f'{ex.HOLE_PARAM_TOKEN}_{ex.TLang.get_out_type(t)}_{info["nht"]}'
            ploc_nt = f'{ex.PARAM_LOC_TOKEN}_{info["nplt"]}'
            info['nht'] += 1
            info['nplt'] += 1
            v.tokens.append(ploc_nt)
            v.bck_map[hole_nt] = ploc_nt
            v.param_vals[ploc_nt] = t
            
        if t == 'end':
            break

    
def nmvi_helper(ex, v, qs, qe, info):
    if len(qs) == 0 and len(qe) == 0:
        return
    
    st = qs.pop(0)

    if ex.STRUCT_LOC_TOKEN in st:
        if ex.ex_name == 'shape':
            shape_nmvi_rm_deriv(ex, v, qe, info)
            return True
        else:
            nmvi_rm_deriv(ex, v, qe, info)
            return False
        
    assert ex.SHARED_TOKEN not in st \
        and ex.TLang.get_out_type(st) not in ex.DEF_PARAM_TYPES, f'bad {st}'
    
    v.tokens.append(st)
    assert qe.pop(0) == st
    
    for ii, inp in enumerate(ex.TLang.get_inp_types(st)):        
        if inp in ex.DEF_STRUCT_TYPES:
            cont = nmvi_helper(ex, v, qs, qe, info)
            while cont:
                cont = nmvi_helper(ex, v, qs, qe, info)
            
        else:
            if len(qs) > 0 and inp in ex.DEF_PARAM_TYPES and ex.SHARED_TOKEN in qs[0]\
               and inp == ex.TLang.get_out_type(qe[0]):
                
                blank_nt = f'{ex.BLANK_TOKEN}_{inp}_{info["nbt"]}'
                ploc_nt = f'{ex.PARAM_LOC_TOKEN}_{info["nplt"]}'
                
                info['nbt'] += 1
                info['nplt'] += 1
                
                v.tokens.append(ploc_nt)
                v.bck_map[blank_nt] = ploc_nt
                v.param_vals[ploc_nt] = qe[0]
                
                qs.pop(0)
                qe.pop(0)
                

            elif len(qs) > 0 and inp in ex.DEF_PARAM_TYPES and ex.TLang.get_out_type(qs[0]) in ex.DEF_PARAM_TYPES\
                 and ex.TLang.get_out_type(qs[0]) == ex.TLang.get_out_type(qe[0]):
                                
                blank_nt = f'{ex.BLANK_TOKEN}_{ex.TLang.get_out_type(qs[0])}_{info["nbt"]}'
                ploc_nt = f'{ex.PARAM_LOC_TOKEN}_{info["nplt"]}'
                
                info['nbt'] += 1
                info['nplt'] += 1

                v.tokens.append(ploc_nt)
                v.bck_map[blank_nt] = ploc_nt
                v.param_vals[ploc_nt] = qs[0]
                
                assert qs.pop(0) == qe.pop(0)                                
                
            else:
                et = qe.pop(0)
                blank_nt = f'{ex.BLANK_TOKEN}_{ex.TLang.get_out_type(et)}_{info["nbt"]}'
                ploc_nt = f'{ex.PARAM_LOC_TOKEN}_{info["nplt"]}'

                info['nbt'] += 1
                info['nplt'] += 1
                
                v.tokens.append(ploc_nt)
                v.bck_map[blank_nt] = ploc_nt
                v.param_vals[ploc_nt] = et            

    return False


def add_part_info(ex, expr, struct):
    
    out_expr = []
    
    qs = deepcopy(struct)
    qe = expr.split()
    
    info =  {
        'pc': 0,
        'out_expr': out_expr,
    }    
    
    api_helper(ex, qs, qe, info)

    assert expr == ' '.join([oe.split('!')[0] for oe in out_expr])
    
    return ' '.join(out_expr)

def shape_add_part_info(ex, expr, struct):
    
    out_expr = []
    
    qs = deepcopy(struct)
    qe = expr.split()
    
    info =  {
        'pc': 0,
        'out_expr': out_expr,
    }    

    rqs = ex.reformat_struct(qs)     
    
    shape_api_helper(ex, rqs, qe, info)

    assert expr == ' '.join([oe.split('!')[0] for oe in out_expr])
    
    return ' '.join(out_expr)


def api_deriv(ex, q, info):
    ipc = 1
    while(ipc > 0):
        ipc -= 1
        t = q.pop(0)
        ipc += ex.TLang.get_num_inp(t)

        if t in ex.PRT_FNS:
            info['out_expr'] += [f'{t}!{info["pc"]}']
        else:
            info['out_expr'] += [t]


def shape_api_deriv(ex, q, info):

    assert ex.ex_name == 'shape'

    while True:
        t = q.pop(0)

        if t == 'Cuboid':
            assert q[0] == 'leaf'
            info['out_expr'] += [f'{t}!{info["pc"]}']
        elif t == 'fill':
            info['out_expr'] += [f'{t}!{info["pc"]}']
        else:
            info['out_expr'] += [t]
            
        if t == 'end':
            break
        
def api_helper(ex, qs, qe, info):
    
    if len(qs) == 0 and len(qe) == 0:
        return
    
    st = qs.pop(0)

    if ex.STRUCT_LOC_TOKEN in st:        
        api_deriv(ex, qe, info)
        info['pc'] += 1
        return

    assert ex.SHARED_TOKEN not in st \
        and ex.TLang.get_out_type(st) not in ex.DEF_PARAM_TYPES, f'bad {st}'

    etf = qe.pop(0)
    assert etf == st

    if etf in ex.PRT_FNS:
        info['out_expr'] += [f'{etf}!{info["pc"]}']
        info['pc'] += 1
    else:
        info['out_expr'] += [etf]
    
    for ii, inp in enumerate(ex.TLang.get_inp_types(st)):

        if inp in ex.DEF_STRUCT_TYPES:
            api_helper(ex, qs, qe, info)
            
        else:

            if len(qs) > 0 and inp in ex.DEF_PARAM_TYPES and ex.SHARED_TOKEN in qs[0]\
               and inp == ex.TLang.get_out_type(qe[0]):
                
                qs.pop(0)

            elif len(qs) > 0 and inp in ex.DEF_PARAM_TYPES and ex.TLang.get_out_type(qs[0]) in ex.DEF_PARAM_TYPES\
                 and ex.TLang.get_out_type(qs[0]) == ex.TLang.get_out_type(qe[0]):
                qs.pop(0)
                
            et = qe.pop(0)

            if et in ex.PRT_FNS:
                info['out_expr'] += [f'{et}!{info["pc"]}']
                info['pc'] += 1
            else:
                info['out_expr'] += [et]
                
                
        


def shape_api_helper(ex, qs, qe, info):
    
    assert ex.ex_name == 'shape'
    
    if len(qs) == 0 and len(qe) == 0:
        return False
    
    st = qs.pop(0)

    if ex.STRUCT_LOC_TOKEN in st:        
        shape_api_deriv(ex, qe, info)
        info['pc'] += 1
        return True

    assert ex.SHARED_TOKEN not in st \
        and ex.TLang.get_out_type(st) not in ex.DEF_PARAM_TYPES, f'bad {st}'

    etf = qe.pop(0)
    assert etf == st

    if etf == 'Cuboid' and qe[0] == 'leaf':
        info['out_expr'] += [f'{etf}!{info["pc"]}']
        info['pc'] += 1
    else:
        info['out_expr'] += [etf]
    
    for ii, inp in enumerate(ex.TLang.get_inp_types(st)):

        if inp in ex.DEF_STRUCT_TYPES:
            cont = shape_api_helper(ex, qs, qe, info)
            while cont:
                cont = shape_api_helper(ex, qs, qe, info)
            
        else:

            if len(qs) > 0 and inp in ex.DEF_PARAM_TYPES and ex.SHARED_TOKEN in qs[0]\
               and inp == ex.TLang.get_out_type(qe[0]):
                
                qs.pop(0)

            elif len(qs) > 0 and inp in ex.DEF_PARAM_TYPES and ex.TLang.get_out_type(qs[0]) in ex.DEF_PARAM_TYPES\
                 and ex.TLang.get_out_type(qs[0]) == ex.TLang.get_out_type(qe[0]):
                qs.pop(0)
                
            et = qe.pop(0)

            if et == 'Cuboid' and qe[0] == 'leaf':
                info['out_expr'] += [f'{et}!{info["pc"]}']
                info['pc'] += 1
            else:
                info['out_expr'] += [et]
                
    return False
        
