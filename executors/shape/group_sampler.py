from copy import deepcopy
import numpy as np
import random
import torch
import math

MAX_SAMP_TRIES = 10
EPS = 0.025
MIN_HEIGHT = 0.05
MIN_REF_DIST = 0.1
MIN_TRANS_DIST = 0.05
PREC = 2

def norm_np(L):
    return np.array(L) / np.array(L).sum()

NCO = [0, 1, 2, 3, 4]
NCP = norm_np([.1,.2,.3,.3,.1])

FPT = ['width', 'height', 'depth', 'u', 'v', 'sym_dist']

CMIN = 0.01
CMAX = 1.0
AMIN = 0.0
AMAX = 1.0

LSTD = 0.025
HSTD = 0.2

def make_sampler(v, MIV, MAV):
    mu = v
    lr, up = LSTD,HSTD
    a = random.random()
    std = (lr * a) + (up * (1-a))

    def sampler():
        val = mu + (np.random.randn() * std)
        return max(min(val, MAV), MIV)
        
    return sampler

class ShapeGroupSampler:

    def __init__(self, ex, sample_params):

        self.gs = Sampler()
        self.gs.init_sampler(sample_params)
        
        if not self.gs.gs_valid:
            self.gs_valid = False
            return
        
        self.gs.sample(ex)

        self.gs_valid = self.gs.gs_valid
        
        if not self.gs_valid:
            return
        
        self.ex = ex
                
        self.tokens, self.params, self.struct_prog_tokens = self.gs.get_struct_token_info(
            self.ex
        )

        
    # return a ShapeGroup deriv
    def sample_deriv(self):
        sg = ShapeGroup(self.gs, self.ex)        
        return sg
        
    # return a relations 
    def get_relations(self):
        return {k:v for k,v in self.params.items() if v is not None}

    def get_struct_prog_tokens(self):
        return self.struct_prog_tokens

class ShapeGroup:
    def __init__(self, gs, ex):

        self.ex = ex
        prog_tokens = gs.sample(self.ex)
        if prog_tokens is None:
            self.gs_valid = False
            return
        else:
            self.gs_valid = True
            
        self.prog = ' '.join(prog_tokens)    
        
        self.tokens, self.param_vals, self.struct_derivs, self.bck_map = \
            gs.get_deriv_token_info(ex)

        prog = ' '.join([
            self.param_vals[t] if t in self.param_vals else t for t in self.tokens
        ])
        
        assert self.prog == prog
        
    def ex_prog(self):
        self.P = self.ex.ex_prog(self.prog.split())

    def get_sig(self):
        return self.P.get_state_sig()

    def is_valid(self):
        return self.ex.check_valid_prog(self.P)



##########################
##########################
##########################


def sample_bbox_dims():

    width = help_norm_sample(.65, .35, .3, 1.0)
    height = help_norm_sample(.65, .35, .3, 1.0)
    depth = help_norm_sample(.65, .35, .3, 1.0)

    r = random.randint(0,2)

    if r == 0:
        width = 1.0
    elif r == 1:
        height = 1.0
    elif r == 2:
        depth = 1.0
    else:
        assert False
    
    return (
        width,
        height,
        depth        
    )
    

def check_add_ref_sym(sym_opts, AXIS, bounds, bbdim, prm):
    
    if (bounds[0] >= 0.5 and bounds[1]) <= 0.5 or abs(bounds[0] - (1 - bounds[1])) > 0.1:
        return 

    if (bounds[1] - bounds[0]) * bbdim < MIN_REF_DIST:
        return
    
    sym_opts.append((prm, 'reflect', AXIS, 2, None))

def check_add_trans_sym(sym_opts, AXIS, bounds, bbdim, prm):
    max_cubes = math.floor((bounds[1] - bounds[0]) * bbdim / MIN_TRANS_DIST)

    if max_cubes <= 1:
        return

    num_cubes = random.randint(2, min(max_cubes,5))

    min_dist = num_cubes * MIN_TRANS_DIST
    max_dist = bounds[1] - bounds[0]

    dist = help_uni_sample(min_dist, max_dist)

    sym_opts.append((prm, 'translate', AXIS, num_cubes, dist))


def sample_app_params(apt, li):
    
    st, adir, level = apt.app_att_info
    
    l_r, d_u, b_f = li.get_bounds(level)
        
    assert l_r is not None and b_f is not None and d_u is not None

    bw, bh, bd = li.bb_dims

    apt.level = level

    loc_place = {}
    
    min_width = EPS/bw
    max_width = l_r[1] - l_r[0]

    min_height = EPS/bh
    max_height = (d_u[1] - d_u[0]) * li.heights[level]

    min_depth = EPS/bd
    max_depth = b_f[1] - b_f[0]

    SCL = {
        'width': (min_width, max_width),
        'height': (min_height, max_height),
        'depth': (min_depth, max_depth),        
    }

    BCL = {
        'width': l_r,
        'height': d_u,
        'depth': b_f,
    }

    DL = {
        ('width', 'u', 'low'): 'bot',
        ('width', 'u', 'high'): 'top',
        ('width', 'v', 'low'): 'back',
        ('width', 'v', 'high'): 'front',

        ('height', 'u', 'low'): 'left',
        ('height', 'u', 'high'): 'right',
        ('height', 'v', 'low'): 'back',
        ('height', 'v', 'high'): 'front',

        ('depth', 'u', 'low'): 'left',
        ('depth', 'u', 'high'): 'right',
        ('depth', 'v', 'low'): 'bot',
        ('depth', 'v', 'high'): 'top',
        
    }

    kl = {
        'width': 'l_r',
        'height': 'd_u',
        'depth': 'b_f',
    }

    for k in ('u', 'v'):
        r = random.random()
        if r < 0.33:
            loc_place[k] = 'low'        
        elif r < 0.66:
            loc_place[k] = 'high'
        else:
            loc_place[k] = 'center'        

    if adir in ('left', 'right'):
        d1,d2,d3 = ('width', 'height', 'depth')            
    elif adir in ('bot', 'top'):
        d1,d2,d3 = ('height', 'width', 'depth')  
    elif adir in ('back', 'front'):
        d1,d2,d3 = ( 'depth', 'width', 'height')        
    else:
        assert False

    if apt.app_sym_info is not None:
        s_un, s_type, s_axis, s_num, s_dist = apt.app_sym_info

        li.remove_att_bounds(level)
        apt.set_attr(d1, sample_size(SCL[d1][0], SCL[d1][1]))

        apt.sym_axis = s_axis
            
        if s_type == 'translate':
            apt.sym_num = s_num
            apt.set_sym_dist(s_dist)
            
        for un,dn  in [('u', d2), ('v', d3)]:
            if s_un == un:                
                apt.set_attr(dn, sample_size(SCL[dn][0], SCL[dn][1]/s_num))
                apt.set_attr(un, help_uni_sample(
                    BCL[dn][0] + apt.get_attr(dn)/2.,
                    (1.0/s_num) - apt.get_attr(dn)/2.
                ))

            else:
                apt.set_attr(dn, sample_size(SCL[dn][0], SCL[dn][1]))
                apt.set_attr(un, 0.5)
        
    else:
        
        apt.set_attr(d1, sample_size(SCL[d1][0], SCL[d1][1]))

        li.update_att_bounds(
            level,
            kl[d1],
            BCL[d1][0] + apt.get_attr(d1) / 2.,
            1.0
        )

        for un,dn  in [('u', d2), ('v', d3)]:

            if (SCL[dn][1]/2. - SCL[dn][0]) > EPS and loc_place[un] == 'low':
                apt.set_attr(dn, sample_size(SCL[dn][0], SCL[dn][1]/2.))
                apt.set_attr(un, help_uni_sample(
                    BCL[dn][0] + apt.get_attr(dn)/2.,
                    .5 - apt.get_attr(dn)/2.
                ))
                li.update_att_bounds(
                    level,
                    kl[dn],
                    BCL[dn][0] + apt.get_attr(dn)/2.,
                    1.0
                )
                apt.pos_atts += [(DL[(dn, un, 'low')], level)]

            elif (SCL[dn][1]/2. - SCL[dn][0]) > EPS and loc_place[un] == 'high':
                apt.set_attr(dn, sample_size(SCL[dn][0], SCL[dn][1]/2.))
                apt.set_attr(un, help_uni_sample(
                    .5 + apt.get_attr(dn)/2.,
                    BCL[dn][1] - apt.get_attr(dn)/2.,                    
                ))
                li.update_att_bounds(
                    level,
                    kl[dn],
                    0.0,
                    BCL[dn][1] - apt.get_attr(dn)/2.,
                )
                apt.pos_atts += [(DL[(dn, un, 'high')], level)]
            else:
                apt.set_attr(dn, sample_size(SCL[dn][0], SCL[dn][1]))
                apt.set_attr(un, 0.5)

    for fpt in FPT[:5]:
        assert apt.float_param_vals[fpt] is not None
                                
    if len(apt.pos_atts) > 0:
        li.add_att_opt(level, apt)                    

class GSupType:
    def __init__(self, name):

        self.name = name
        self.hier = None
        self.samp_sym_logic = None
        self.samp_att_logic = None
        self.share_info = {}
        self.param_names = []
        self.hole_name = None
        self.in_hole = False

        self.set_sym_logic = None

        self.float_param_samplers = {fpt:None for fpt in FPT}
        
        self.reset()

    def reset(self):
        self.pos_atts = []
        self.sym_axis = None
        self.sym_num = None                
        self.att_cube = None
        self.att_face = None
        self.level = None
        self.loc_place = {
            'side': None,
            'lat': None
        }
                    
        self.cub_line = None
        self.move_line = None
        self.sym_line = None

        self.app_sym_info = None
        self.app_att_info = None

        self.float_param_vals = {fpt:None for fpt in FPT}

        
    def width(self):
        key = 'width'
        assert self.float_param_vals[key] is not None
        return self.float_param_vals[key]

    def height(self):
        key = 'height'
        assert self.float_param_vals[key] is not None
        return self.float_param_vals[key]

    def depth (self):
        key = 'depth'
        assert self.float_param_vals[key] is not None
        return self.float_param_vals[key]

    def u(self):
        key = 'u'
        assert self.float_param_vals[key] is not None
        return self.float_param_vals[key]

    def v(self):
        key = 'v'
        assert self.float_param_vals[key] is not None
        return self.float_param_vals[key]

    def sym_dist(self):
        key = 'sym_dist'
        assert self.float_param_vals[key] is not None
        return self.float_param_vals[key]
    
    def set_width(self, v):
        key = 'width'
        
        val = v

        self.float_param_vals[key] = val
    
    def set_height(self, v):
        key = 'height'
        
        val = v

        self.float_param_vals[key] = val

    def set_depth(self, v):
        key = 'depth'
        
        val = v

        self.float_param_vals[key] = val

    def set_u(self, v):
        key = 'u'
        
        val = v

        self.float_param_vals[key] = val

    def set_v(self, v):
        key = 'v'
        
        val = v

        self.float_param_vals[key] = val

    def set_sym_dist(self, v):
        key = 'sym_dist'
        
        val = v

        self.float_param_vals[key] = val
        
        
    def add_deriv_token_info(self, ex, tokens, param_vals, bck_map, counts):
        self.pc = 0
        
        def ast(t):
            tokens.append(t)

        def apt(val):
            tkn = f'{ex.PARAM_LOC_TOKEN}_{counts["params"]}'
            counts["params"] += 1
            opn = self.param_names[self.pc]
            self.pc += 1

            tokens.append(tkn)
            param_vals[tkn] = val
            bck_map[opn] = tkn
    
        ast('Cuboid')

        if self.hier == ex.HOLE_TOKEN:
            assert self.hole_name is not None
            ast('hier')
        else:
            ast(self.hier)

        for cval in self.cub_line[2:]:
            apt(cval)

        ast('Attach')
        for aval in self.move_line[1:]:
            apt(aval)

        if len(self.sym_line) > 0:
            ast(self.sym_line[0])
            for spv in self.sym_line[1:]:
                apt(spv)
        
        assert self.pc == len(self.param_names), 'didnt cover param names'
        
    def add_struct_token_info(self, ex, tokens, params, struct_tokens, counts):
        assert len(self.param_names) == 0
        def ast(t):
            tokens.append(t)
            struct_tokens.append(t)

        def apt(inp,pt=None,pv=None):

            if self.in_hole:
                nt = f'{ex.HOLE_PARAM_TOKEN}_{inp}_{counts["hparams"]}'
                counts['hparams'] += 1
            else:
                nt = f'{ex.BLANK_TOKEN}_{inp}_{counts["params"]}'
                counts['params'] += 1
                
            tokens.append(nt)
            self.param_names.append(nt)
            
            if inp in ex.CAT_PARAM_TYPES:
                assert inp not in self.share_info
                self.share_info[inp] = nt
            
            if pt is None or pt == 'i':
                params[nt] = None
                return

            if pt == 'c':
                if inp == 'cind':
                    pv = f'cind_{pv}'
                if inp == 'cnum':
                    pv = f'cnum_{pv}'
                assert pv in ex.TLang.tokens
                params[nt] = ('static', pv)
                return
            
            elif pt == 's':
                assert callable(pv)
                params[nt] = ('reuse', pv(inp))
                return 

            assert False        

        ast('Cuboid')
            
        if self.hier == ex.HOLE_TOKEN:
            struct_tokens.append(self.hier)
            nt = f'{ex.STRUCT_LOC_TOKEN}_{counts["hole"]}'            
            tokens.append(nt)
            counts['hole'] += 1
            self.hole_name = nt
            
        else:
            ast(self.hier)

        for _ in range(3):
            apt('cflt')
            
        ast('Attach')

        apt('cind', self.samp_att_logic[0], self.samp_att_logic[1])
        apt('face', self.samp_att_logic[2], self.samp_att_logic[3])

        for _ in range(2):
            apt('pflt')

        sym_logic = self.samp_sym_logic

        if sym_logic is None:
            return

        if sym_logic[0] == 'ref':
            ast('Reflect')
            apt('axis', sym_logic[1], sym_logic[2])
        elif sym_logic[0] == 'trans':
            ast('Translate')
            apt('axis', sym_logic[1], sym_logic[2])
            apt('pflt')
            apt('cnum', sym_logic[3], sym_logic[4])
        
            
    def sample_cat_decisions(self):
        self.sample_att_dec()
        self.sample_sym_dec()
            
    def getAbsDims(self, bb_dims):
        return self.width() * bb_dims[0], self.height() * bb_dims[1], self.depth() * bb_dims[2]

    def print_info(self):
        print(f"~~ {self.name} ({self.level}) ~~ ")
        print(self.samp_att_logic)
        print(f"  {(self.att_cube, self.att_face, self.sym_axis)}")
        print(f"  loc_place: {self.loc_place}")


    def add_trans_sym(self, axis_rel, axis_val, cnum_rel, cnum_val):
        self.samp_sym_logic = ('trans', axis_rel, axis_val, cnum_rel, cnum_val)
        
    def add_ref_sym(self, axis_rel, axis_val):
        self.samp_sym_logic = ('ref', axis_rel, axis_val)
        
    def sample_sym_dec(self):
        if self.samp_sym_logic is None:
            return

        if self.samp_sym_logic[0] == 'ref':
            return self.sample_ref_dec()

        assert False

    def GA(self, att):
        def ga(r=None):
            if r is not None:
                return self.share_info[r]
            
            return getattr(self, att)
        return ga

    def disam_val(self, r, V):
        if callable(V):
            v = V()
        elif isinstance(V, list):
            assert r == 'i'
            v = random.choice(V)
        else:
            assert r == 'c'
            v = V

        return v
    
    def sample_ref_dec(self):
        _, axis_rel, U_axis_val = self.samp_sym_logic

        axis_val = self.disam_val(axis_rel, U_axis_val)

        self.sym_axis = axis_val

    def sample_att_dec(self):
        cube_rel, cube_val, face_rel, face_val = self.samp_att_logic

        self.att_cube = self.disam_val(cube_rel, cube_val)
        self.att_face = self.disam_val(face_rel, face_val)
        
    def add_attach(self, cube_rel, cube_val, face_rel, face_val):
        self.samp_att_logic = (cube_rel, cube_val, face_rel, face_val)        

    def make_cuboid_line(self):

        hier = self.hier

        if hier == 'HOLE':
            hier = 'hier'

        assert hier in ('leaf', 'hier')
        
        assert self.hier is not None
        return [
            'Cuboid',
            hier,
            self.width(),
            self.height(),
            self.depth()
        ]

    def make_att_line(self):
        c1 = self.att_cube
        face = self.att_face
        return ['Attach', f'cind_{c1}', face, self.u(), self.v()]

    def make_sym_line(self):
        if self.samp_sym_logic is None:
            return []
        elif self.samp_sym_logic[0] == 'ref':
            return ['Reflect', self.sym_axis]
        elif self.samp_sym_logic[0] == 'trans':
            return ['Translate', self.sym_axis, self.sym_dist(), f'cnum_{self.sym_num}']
    
    def make_lines(self, ex):

        self.cub_line = replace_floats_with_tokens(ex, self.make_cuboid_line())
        self.move_line = replace_floats_with_tokens(ex,self.make_att_line())
        self.sym_line = replace_floats_with_tokens(ex,self.make_sym_line())

        comb_lines = self.cub_line + self.move_line + self.sym_line
        
        return comb_lines

    def set_attr(self, name, value):
        assert name in FPT

        if name == 'width':
            return self.set_width(value)
        elif name == 'height':
            return self.set_height(value)
        elif name == 'depth':
            return self.set_depth(value)
        elif name == 'u':
            return self.set_u(value)
        elif name == 'v':
            return self.set_v(value)

        assert False

    def get_attr(self, name):
        assert name in FPT

        if name == 'width':
            return self.width()
        elif name == 'height':
            return self.height()
        elif name == 'depth':
            return self.depth()
        elif name == 'u':
            return self.u()
        elif name == 'v':
            return self.v()

        assert False
        
    def sample_app_att_logic(self, level_info):
        open_spots = level_info.sample_open_spots()

        if len(open_spots) == 0:
            return None, None
        
        choice = random.choice(open_spots)
        
        cl, name, face, st = choice

        I = [None, None, None, None]

        if random.random() < 0.5:
            I[0] = 'c'
            I[1] = name
        else:
            I[0] = 'i'

        if random.random() < 0.5:
            I[2] = 'c'
            I[3] = face
        else:
            I[2] = 'i'

        self.att_cube = name
        self.att_face = face
        
        return tuple(I), (st, face, cl)

    def check_app_att_logic(self, level_info):
        open_spots = level_info.sample_open_spots()

        choices = []

        cr, cv, fr, fv = self.samp_att_logic
        
        for (cl, name, face, st) in open_spots:
            if cr == 'c' and name != cv:
                continue
            if fr == 'c' and face != fv:
                continue
            choices.append((cl, name, face, st))

        if len(choices) == 0:
            return False, None
            
        choice = random.choice(choices)

        cl, name, face, st = choice
        
        self.att_cube = name
        self.att_face = face

        return True, (st, face, cl)

        
    def sample_app_sym_logic(self, li, asi):

        sym_opts = self.app_sym_pre_logic(li, asi)

        if sym_opts is False:
            return False, None

        if len(sym_opts) == 0 or random.random() < 0.5:
            return True, None

        s_un, s_type, s_axis, s_num, s_dist = random.choice(sym_opts)

        self.app_sym_info = (s_un, s_type, s_axis, s_num, s_dist)

        I = [None, None, None, None]

        if random.random() < 0.5:
            I[0] = 'c'
            I[1] = s_axis
        else:
            I[0] = 'i'

        if random.random() < 0.25:
            I[2] = 'c'
            I[3] = s_num
        else:
            I[2] = 'i'
        
        if s_type == 'reflect':
            sasl = ('ref', I[0], I[1])
        elif s_type == 'translate':
            sasl = ('trans', I[0], I[1], I[2], I[3])
        else:
            assert False

        return True, sasl

    def app_sym_pre_logic(self, li, asi):
        st, adir, level = asi
        l_r, d_u, b_f = li.get_bounds(level)

        if l_r is None or b_f is None or d_u is None:
            return False

        bw, bh, bd = li.bb_dims

        sym_opts = []

        if adir in ('left', 'right'):
            check_add_trans_sym(sym_opts, 'AY', d_u, bh, 'u')
            check_add_trans_sym(sym_opts, 'AZ', b_f, bd, 'v')
            check_add_ref_sym(sym_opts, 'AZ', b_f, bd, 'v')
            
        elif adir in ('bot', 'top'):
            check_add_trans_sym(sym_opts, 'AX', l_r, bw, 'u')
            check_add_ref_sym(sym_opts, 'AX', l_r, bw, 'u')
            check_add_trans_sym(sym_opts, 'AZ', b_f, bd, 'v')
            check_add_ref_sym(sym_opts, 'AZ', b_f, bd, 'v')

        elif adir in ('back', 'front'):        
            check_add_trans_sym(sym_opts, 'AX', l_r, bw, 'u')
            check_add_ref_sym(sym_opts, 'AX', l_r, bw, 'u')
            check_add_trans_sym(sym_opts, 'AY', d_u, bh, 'v')

        else:
            assert False

        return sym_opts
    
    def check_app_sym_logic(self, li, asi):
        sym_opts = self.app_sym_pre_logic(li, asi)

        if sym_opts is False:
            return False

        if self.samp_sym_logic is None:
            return True

        sym_logic = self.samp_sym_logic

        if len(sym_logic) == 3:
            sfn, ar, av = sym_logic
        else:
            sfn, ar, av, cnr, cnv = sym_logic

        choices = []

        for (s_un, s_type, s_axis, s_num, s_dist) in sym_opts:
            if s_type != sfn:
                continue

            if ar == 'c' and s_axis != av:
                continue

            if s_type == 'trans' and cnr == 'c' and cnv != s_num:
                continue

            choices.append((s_un, s_type, s_axis, s_num, s_dist))

        if len(choices) == 0:
            return False

        s_un, s_type, s_axis, s_num, s_dist = random.choice(sym_opts)

        self.app_sym_info = (s_un, s_type, s_axis, s_num, s_dist)

        return True
        
    def sample_app_decisions(self, level_info):
        
        if self.samp_att_logic is None:
            saal, asi = self.sample_app_att_logic(level_info)
            if saal is None:
                return False
            self.samp_att_logic = saal
        else:
            valid, asi = self.check_app_att_logic(level_info)
            if not valid:
                return False

        self.app_att_info = asi

        # This should populate self.samp_sym_info 
        if self.set_sym_logic is None:
            self.set_sym_logic = 'y'
            valid, sasl = self.sample_app_sym_logic(level_info, asi)
            if not valid:
                return False
            
            self.samp_sym_logic = sasl
        else:
            valid = self.check_app_sym_logic(level_info, asi)
            if not valid:
                return False
                    
        sample_app_params(self, level_info)

        return True
        
def sample_sup_struct_v1():

    st = GSupType(1)

    st.add_attach('c', 0, 'c', 'bot')

    def sample_cov_info(st_list):
        assert len(st_list) == 1

        st_list[0].level = 1
        
        cov_info = {
            1: ['width', 'depth']
        }
        
        return cov_info

    r = random.random()

    if r < 0.25:
        st.add_ref_sym('c','AX')

    elif r < 0.5:
        st.add_ref_sym('c','AZ')
        
    else:
        st.add_ref_sym('i',['AX','AZ'])
            
    return [st], sample_cov_info
    

def sample_sup_struct_v2():


    st1 = GSupType(1)
    st2 = GSupType(2)
    
    st1.add_attach('c', 0, 'c', 'bot')
    st2.add_attach('c', 0, 'c', 'bot')

    r = random.random()
    
    if r < .25:
        st1.add_ref_sym('c', 'AX')
        st2.add_ref_sym('c', 'AX')
        
    elif r < .5:

        st1.add_ref_sym('c', 'AZ')
        st2.add_ref_sym('c', 'AZ')

    else:
        st1.add_ref_sym('i', ['AX','AZ'])
        st2.add_ref_sym('s', st1.GA('sym_axis'))
                
    def sample_cov_info(st_list):
        assert len(st_list) == 2

        st1 = st_list[0]
        st2 = st_list[1]
        
        st1.loc_place['side'] = 'left'
        st1.loc_place['lat'] = 'back'
        
        st1.level = 1
        st2.level = 1
        
        if 'AX' == st1.sym_axis :
            
            st2.loc_place['side'] = 'left'
            st2.loc_place['lat'] = 'front'
            
        elif 'AZ' == st1.sym_axis:
    
            st2.loc_place['side'] = 'right'
            st2.loc_place['lat'] = 'back'
        else:
            
            assert False
            
        cov_info = {
            1: ['width', 'depth'],
            2: ['width', 'depth'],
        }

        return cov_info
    
    return [st1, st2], sample_cov_info

def sample_sup_struct_v3():
    st1 = GSupType(1)
    st2 = GSupType(2)
    
    if random.random() < 0.5:
        aface = 'top'
        oface = 'bot'
    else:
        aface = 'bot'
        oface = 'top'

    ct1 = 'c'
    co1 = 0
    ft1 = random.choice(['i','c'])

    if ft1 == 'i':
        fo1 = ['top','bot']
    elif ft1 == 'c':
        fo1 = aface

    ct2 = random.choice(['i', 'c'])

    if ct2 == 'i':
        co2 = [0,1]
    else:
        co2 = random.choice([0,1])

    def SV3FN(A, B):
        def q():

            if B.att_cube == 1:
                return A.att_face
            else:
                if A.att_face == 'top':
                    return 'bot'
                elif A.att_face == 'bot':
                    return 'top'

            assert False
                

        return q

    st1.add_attach(ct1,co1,ft1,fo1)
    
    fo2 = SV3FN(st1, st2)
    
    if ct2 == 'i':
        ft2 = 'i'
    else:
        if ft1 == 'i':
            if co2 == 0:
                ft2 = 'i'
            elif co2 == 1:
                ft2 = 's'
                fo2 = st1.GA('att_face')
        else:
            ft2 = 'c'
            assert ct1 == 'c' and ft1 == 'c'
            assert ct2 == 'c'
            if co2 == 1:
                fo2 = fo1
            else:
                if fo1 == 'bot':
                    fo2 = 'top'
                elif fo1 == 'top':
                    fo2 = 'bot'
                else:
                    assert False
            
    st2.add_attach(ct2,co2,ft2,fo2)

    rr = random.random()
    
    if rr < 0.25:
        st2.add_ref_sym('i', ['AX','AZ'])
    elif rr < 0.5:
        st2.add_ref_sym('c', random.choice(['AX','AZ']))
    
    
    def sample_cov_info(st_list):
        assert len(st_list) == 2

        st1 = st_list[0]
        st2 = st_list[1]
        
        if st1.att_face == 'top':        
            st1.level = 2
            st2.level = 1

        elif st1.att_face == 'bot':            
            st1.level = 1
            st2.level = 2
        else:
            assert False
                    
        cov_info = {
            1: [],
            2: []
        }

        r1 = random.random() < 0.5
    
        if random.random() < 0.5:
            cov_info[1].append('width')
            if r1: cov_info[2].append('width')
        else:
            cov_info[2].append('width')
            if r1: cov_info[1].append('width')
                
        r2 = random.random() < 0.5
        
        if random.random() < 0.5:
            cov_info[1].append('depth')
            if r2: cov_info[2].append('depth')
        else:
            cov_info[2].append('depth')
            if r2: cov_info[1].append('depth')

        return cov_info        
        
    return [st1, st2], sample_cov_info

def sample_sup_struct_v4():

    st1 = GSupType(1)
    st2 = GSupType(2)
    st3 = GSupType(3)
    
    if random.random() < 0.5:
        aface = 'top'
        oface = 'bot'
    else:
        oface = 'top'
        aface = 'bot'

    def SV4FN1(A):
        def q():
            if A.att_face == 'top':
                return 'bot'
            elif A.att_face == 'bot':
                return 'top'

            assert False
        return q

    def SV4FN2(A, B, C):
        def q():
            if A.att_cube == B.name:
                return B.att_face
            elif A.att_cube == C.name:
                return C.att_face
            else:
                print(A.att_cube)
                print(B.name)
                print(C.name)
                print(B.att_face)
                print(C.att_face)                
                assert False
                
        return q
    
    const = random.random() < 0.5
    
    if const:
        st1.add_attach('c', 0, 'c', aface)
        st2.add_attach('c', 0, 'c', oface)        
    else:
        st1.add_attach('c', 0, 'i', [aface, oface])
        st2.add_attach('c', 0, 'i', SV4FN1(st1))

    r = random.random()
    if const:
        if r < 0.4:
            st3.add_attach('c', 1, 'c', aface)
        elif r < 0.8:
            st3.add_attach('c', 2, 'c', oface)
        else:
            st3.add_attach('i', [1,2], 'i', SV4FN2(st3, st1, st2))
    else:
        if r < 0.4:
            st3.add_attach('c', 1, 's', st1.GA('att_face'))
        elif r < 0.8:
            st3.add_attach('c', 2, 's', st2.GA('att_face'))
        else:
            st3.add_attach('i', [1,2], 'i', SV4FN2(st3, st1, st2))
                    
    rr = random.random()
    
    if rr < 0.25:
        st3.add_ref_sym('i', ['AX','AZ'])
    elif rr < 0.5:
        st3.add_ref_sym('c', random.choice(['AX','AZ']))

    def sample_cov_info(st_list):
        assert len(st_list) == 3

        st1 = st_list[0]
        st2 = st_list[1]
        st3 = st_list[2]
        
        st3.level = 2

        if st1.att_face == 'top':        
            st1.level = 3
            st2.level = 1
        elif st1.att_face == 'bot':
            st1.level = 1
            st2.level = 3
        else:
            assert False
            
        cov_info = {
            1: [],
            2: [],
            3: [],
        }
        
        wc = [1,2,3]
        dc = [1,2,3]

        random.shuffle(wc)
        random.shuffle(dc)

        cov_info[wc[0]].append('width')
        cov_info[dc[0]].append('depth')

        for i in wc[1:]:
            if random.random() < 0.5:
                cov_info[i].append('width')

        for i in dc[1:]:
            if random.random() < 0.5:
                cov_info[i].append('depth')

        return cov_info
                
    return [st1, st2, st3], sample_cov_info
    
def sample_sup_struct_v5():

    st1 = GSupType(1)
    st2 = GSupType(2)
    st3 = GSupType(3)

        
    if random.random() < 0.5:
        aface = 'top'
        oface = 'bot'
    else:
        oface = 'top'
        aface = 'bot'

    const = random.random() < 0.5

    if const:
        st1.add_attach('c', 0, 'c', aface)
        st2.add_attach('c', 1, 'c', aface)
    else:
        st1.add_attach('c', 0, 'i', [aface,oface])
        st2.add_attach('c', 1, 's', st1.GA('att_face'))

    def SV5FN(A, B):
        def q():
            if A.att_cube == B.name:
                return B.att_face
            elif A.att_cube == 0:
                if B.att_face == 'top':
                    return 'bot'
                elif B.att_face == 'bot':
                    return 'top'

            assert False
        return q

    r = random.random()
    if const:
        if r < 0.4:
            st3.add_attach('c', 2, 'c', aface)
        elif r < 0.8:
            st3.add_attach('c', 0, 'c', oface)
        else:
            st3.add_attach('i',[0,2], 'i', SV5FN(st3, st2))
        
    else:
        if r < 0.4:
            st3.add_attach('c', 2, 's', st1.GA('att_face'))
        elif r < 0.8:
            st3.add_attach('c', 0, 'i', SV5FN(st3, st2))
        else:
            st3.add_attach('i',[0,2], 'i', SV5FN(st3, st2))
            

    rr = random.random()
    
    if rr < 0.25:
        st3.add_ref_sym('i', ['AX','AZ'])
    elif rr < 0.5:
        st3.add_ref_sym('c', random.choice(['AX','AZ']))


    def sample_cov_info(st_list):
        assert len(st_list) == 3

        st1 = st_list[0]
        st2 = st_list[1]
        st3 = st_list[2]

        if st1.att_face == 'top':                
            st1.level=3
            st2.level=2
            st3.level=1
            
        elif st1.att_face == 'bot':
            st1.level=1
            st2.level=2
            st3.level=3
        else:
            assert False
            

        cov_info = {
            1: [],
            2: [],
            3: [],
        }

        wc = [1,2,3]
        dc = [1,2,3]

        random.shuffle(wc)
        random.shuffle(dc)

        cov_info[wc[0]].append('width')
        cov_info[dc[0]].append('depth')

        for i in wc[1:]:
            if random.random() < 0.5:
                cov_info[i].append('width')

        for i in dc[1:]:
            if random.random() < 0.5:
                cov_info[i].append('depth')

        return cov_info
                
    return [st1, st2, st3], sample_cov_info
    


def sample_support_info(mode):
    T = []

    OPTS = ['SP1', 'SP21', 'SP2{2/3}', 'SP3{1/2}', 'SP3{3/4}']
    if mode == 'root':
        PROBS = [.1, .1, .2, .3, .3]
    elif mode == 'sub':
        PROBS = [.3, .25, .25, .1, .1]
    else:
        assert False

    stype = sample_help(OPTS, PROBS)
    
    sup_fn_map = {
        'SP1': sample_sup_struct_v1,
        'SP21': sample_sup_struct_v2,
        'SP2{2/3}': sample_sup_struct_v3,
        'SP3{1/2}': sample_sup_struct_v4,
        'SP3{3/4}': sample_sup_struct_v5,
    }

    return sup_fn_map[stype]()

    

def add_loc_place_info(st, cov_info):

    if st.sym_axis == 'AX':    
        st.loc_place['side'] = 'left'

    if st.sym_axis == 'AZ':
        st.loc_place['lat'] = 'back'
                        
    if st.loc_place['side'] is None:
        if 'width' in cov_info[st.name]:
            st.loc_place['side'] = 'center'
                
        else:
            r = random.random()
            if r < 0.33:
                st.loc_place['side'] = 'left'
            if r < 0.66:
                st.loc_place['side'] = 'right'
            else:
                st.loc_place['side'] = 'center'

    if st.loc_place['lat'] is None:
        if 'depth' in cov_info[st.name]:
            st.loc_place['lat'] = 'center'
        else:
            r = random.random()
            if r < 0.33:
                st.loc_place['lat'] = 'back'
            elif r < 0.66:
                st.loc_place['lat'] = 'front'
            else:
                st.loc_place['lat'] = 'center'



def sample_level_heights(sup_type, bb_dims):

    num_levels = len(set([st.level for st in sup_type]))
    heights = {}
    if num_levels == 1:
        heights[1] = 1.

    if num_levels == 2:
        r = max(min(random.random(),1.- MIN_HEIGHT), MIN_HEIGHT)
        heights[1] = r
        heights[2] = 1.-r

    if num_levels == 3:        
        r1 = random.random()
        r2 = random.random()
        r3 = random.random()
        
        s = r1+r2+r3
        
        r1 = max(r1, MIN_HEIGHT * s)
        r2 = max(r2, MIN_HEIGHT * s)
        r3 = max(r3, MIN_HEIGHT * s)
        
        s = r1+r2+r3
    
        heights[1] = r1/s
        heights[2] = r2/s
        heights[3] = r3/s

    for st in sup_type:
        st.set_height(heights[st.level])
        heights[st.level] = st.height()
        
    level_info = LevelInfo(bb_dims, heights, len(sup_type))

    return level_info

class LevelInfo:

    def __init__(self, bb_dims, heights, num_cuboids):

        self.num_cuboids = num_cuboids
        self.bb_dims = bb_dims
        self.heights = heights
        self.bounds = {}
        self.opts = {}
        
        for l in heights.keys():
            self.opts[l] = []
            
            self.bounds[l] = {
                'l_r': [0., 1.],
                'd_u': [0., 1.],
                'b_f': [0., 1.],
            }

        dstB = GSupType(0)
        dstT = GSupType(0)
            
        self.fill_levels = {
            0: dstB,
            l+1: dstT
        }

    def get_bounds(self, level):
        return self.bounds[level]['l_r'], \
            self.bounds[level]['d_u'],\
            self.bounds[level]['b_f']
    
        
    def finalize_pos_atts(self):

        for l, B in self.bounds.items():
            if B is None:
                continue

            if l not in self.fill_levels and l-1 in self.fill_levels:
                _st = self.fill_levels[l-1]
                _st.pos_atts += [('bot', l-1)]
                self.add_att_opt(l, _st)

            if l not in self.fill_levels and l+1 in self.fill_levels:
                _st = self.fill_levels[l+1]
                _st.pos_atts += [('top',l+1)]
                self.add_att_opt(l, _st)
                                        
    def record_fill_level(self, l, st):
        self.fill_levels[l] = st
        
    def get_abs_to_rel_size(self, l, k, e):
        v = 1.0
        if k == 'b_f':
            v *= self.heights[l]

        v *= self.bb_dims[['l_r','b_f','d_u'].index(k)]

        return e / v

    def remove_att_bounds(self, l):
        self.bounds[l] = None
    
    def update_att_bounds(self, l, k, mi, ma):
        if self.bounds[l] is None:
            return
        pmi, pma = self.bounds[l][k]
        self.bounds[l][k] = [max(pmi,mi), min(pma,ma)]

        size_thresh = self.get_abs_to_rel_size(l, k, EPS)
        
        if self.bounds[l][k][0] + size_thresh >= self.bounds[l][k][1]:            
            self.bounds[l] = None
            self.opts[l] = []
            
    def add_att_opt(self, level, st):
        if self.bounds[level] is not None:
            self.opts[level].append(st)
            
    def sample_open_spots(self):

        choices = []
        
        pos_levels = [k for k,v in self.opts.items() if len(v) > 0]
        
        for cl in pos_levels:            
            if self.bounds[cl] is None:
                self.opts[cl] = []
                continue

            for opt in self.opts[cl]:
                for pa in opt.pos_atts:
                    choices.append((cl, opt.name, pa[0], opt))
        
        return choices

def sample_size(l, h):
    return help_norm_sample((l+h)/2., (l+h)/2., l, h)
    
def first_sample_size(l, h):
    if l >= h:
        return l
    c = 0
    v = None
    while v is None or v < l or v > h:
        v = np.random.beta(1.75, 3)
        c += 1
        if c >= MAX_SAMP_TRIES:
            return l

    return v

def refl_cov_logic(size):
    return size/2.

def sample_help(O, UP):
    NP = np.array(UP)
    NP /= NP.sum()

    return np.random.choice(O, p=NP)

def help_uni_sample(a,b):
    r = random.random()
    return a * r + (1-r) * b

def help_norm_sample(m,s,l,h):
    if l >= h:
        return l
    c = 0
    v = None
    while v is None or v < l or v > h:
        v = m + (np.random.randn() * s)
        c += 1
        if c >= MAX_SAMP_TRIES:
            return l
        
    return v

def sample_support_params(
    sup_infos, fill, dims
):

    level_info = sample_level_heights(sup_infos, dims)
    
    for st in sup_infos:
        
        level = st.level
        
        if 'width' in fill[st.name]:
            if st.loc_place['side'] == 'left':
                st.set_width(sample_size(EPS, .5-EPS))
                st.set_u(refl_cov_logic(st.width()))
            elif st.loc_place['side'] == 'right':
                st.set_width(sample_size(EPS, .5-EPS))
                st.set_u(1.0-refl_cov_logic(st.width()))            
            else:
                st.set_width(1.0)
        else:
            st.set_width(sample_size(EPS, 1.-EPS))

        if 'depth' in fill[st.name]:
            if st.loc_place['lat'] == 'back':
                st.set_depth(sample_size(EPS, .5-EPS))
                st.set_v(refl_cov_logic(st.depth()))

            elif st.loc_place['lat'] == 'front':
                st.set_depth(sample_size(EPS, .5-EPS))
                st.set_v(1.0-refl_cov_logic(st.depth()))
            
            else:
                st.set_depth(1.0)
        else:
            st.set_depth(sample_size(EPS, 1.-EPS))

        if st.sym_axis is not None:
            if 'AX' == st.sym_axis:
                st.set_width(sample_size(EPS, .5-EPS))
                if 'width' in fill[st.name]:
                    st.set_u(refl_cov_logic(st.width()))
                else:
                    st.set_u(help_uni_sample(st.width()/2., .5 - st.width()/2.))

                ind = st.u() + (st.width()/2.)
                level_info.update_att_bounds(level, 'l_r', ind, 1-ind)
                st.pos_atts += [('left', level)]
                
            elif 'AZ' == st.sym_axis:
                st.set_depth(sample_size(EPS, .5-EPS))
                if 'depth' in fill[st.name]:
                    st.set_v(refl_cov_logic(st.depth()))
                else:
                    st.set_v(help_uni_sample(st.depth()/2., .5 - st.depth()/2.))

                ind = st.v() + (st.depth()/2.)
                level_info.update_att_bounds(level, 'b_f', ind, 1-ind)
                st.pos_atts += [('back', level)]

            else:
                assert False

        if st.loc_place is None:
            assert False

        side_place = st.loc_place['side']
        lat_place = st.loc_place['lat']    
        
        if st.float_param_vals['u'] is not None:
            pass
        elif side_place == 'center':
            st.set_u(0.5)            
        elif side_place == 'left':
            st.set_u(help_uni_sample(st.width()/2., .5-EPS))
            level_info.update_att_bounds(level, 'l_r',st.u+st.width()/2., 1.0)
            st.pos_atts += [('left', level)]
        elif side_place == 'right':
            st.set_u(help_uni_sample(.5+EPS, 1 - (st.width()/2)))
            level_info.update_att_bounds(level,'l_r',0.0, st.u() - st.width()/2.)
            st.pos_atts += [('right', level)]
        else:
            assert False

        if st.float_param_vals['v'] is not None:
            pass
        elif lat_place == 'center':
            st.set_v(0.5)
        elif lat_place == 'back':
            st.set_v(help_uni_sample(st.depth()/2., .5-EPS))
            level_info.update_att_bounds(level, 'b_f', st.v()+st.depth()/2., 1.0)
            st.pos_atts += [('back', level)]
        elif lat_place == 'front':
            st.set_v(help_uni_sample(.5+EPS, 1 - (st.depth()/2)))
            level_info.update_att_bounds(level, 'b_f', 0.0, st.v() - st.depth()/2.)
            st.pos_atts += [('front', level)]
        else:
            assert False

        if st.u() == 0.5 and st.v() == 0.5:
            level_info.record_fill_level(level, st)
                    
        if len(st.pos_atts) > 0:
            level_info.add_att_opt(level, st)

    level_info.finalize_pos_atts()
                
    return level_info
        
def lookup_float(ftype, T, val):
    ind = (T - val).abs().argmin()
    return f'{ftype}_{ind}'

def replace_floats_with_tokens(ex, full_prog):
    token_prog = []
    last_fn = None

    M = {
        'RootProg': 'cflt',
        'Cuboid': 'cflt',
        'Attach': 'pflt',
        'Squeeze': 'pflt',
        'Translate': 'pflt',
    }
    
    for t in full_prog:

        if '.' in str(t):
            ft = M[last_fn]
            ct = lookup_float(ft, ex.TMAP[ft][0], float(t))            
            token_prog.append(ct)

        else:
            if t in M:
                last_fn = t
            
            token_prog.append(t)
            
    return token_prog

class BBoxInfo:
    def __init__(self):
        self.param_names = []
        self.reset()
        self.float_param_samplers = {fpt:None for fpt in FPT}

    def reset(self):
        self.bbox_line = None
        self.float_param_vals = {fpt:None for fpt in FPT}
        
    def width(self):
        key = 'width'
        assert self.float_param_vals[key] is not None
        return self.float_param_vals[key]

    def height(self):
        key = 'height'
        assert self.float_param_vals[key] is not None
        return self.float_param_vals[key]

    def depth(self):
        key = 'depth'
        assert self.float_param_vals[key] is not None
        return self.float_param_vals[key]
        
    def sample_bbox_dims(self):
        
        width, height, depth = sample_bbox_dims()

        self.set_width(width)
        self.set_height(height)
        self.set_depth(depth)
        
    def set_width(self, v):
        key = 'width'
        
        val = v

        self.float_param_vals[key] = val
    
    def set_height(self, v):
        key = 'height'
        
        val = v

        self.float_param_vals[key] = val

    def set_depth(self, v):
        key = 'depth'
        
        val = v

        self.float_param_vals[key] = val
        
    def get_bbox_dims(self):
        return (self.width(), self.height(), self.depth())
    
    def make_line(self, ex):
        pline = ['START', 'RootProg'] + [b for b in self.get_bbox_dims()]

        self.bbox_line = replace_floats_with_tokens(ex, pline)

        return self.bbox_line

    def add_deriv_token_info(self, ex, tokens, param_vals, bck_map, counts):
        self.pc = 0
        def ast(t):
            tokens.append(t)

        def apt(val):
            tkn = f'{ex.PARAM_LOC_TOKEN}_{counts["params"]}'
            counts["params"] += 1
            opn = self.param_names[self.pc]
            self.pc += 1
            tokens.append(tkn)
            param_vals[tkn] = val
            bck_map[opn] = tkn

        ast(self.bbox_line[0])
        ast(self.bbox_line[1])
        for bbp in self.bbox_line[2:]:
            apt(bbp)

        assert self.pc == len(self.param_names), 'didnt cover param names'
        
    def add_struct_token_info(self, ex, tokens, params, struct_tokens, counts):
        assert len(self.param_names) == 0
        
        def ast(t):
            tokens.append(t)
            struct_tokens.append(t)

        def apt(inp):
            nt = f'{ex.BLANK_TOKEN}_{inp}_{counts["params"]}'
            counts['params'] += 1
            tokens.append(nt)            
            params[nt] = None
            self.param_names.append(nt)
            return

        ast('START')
        ast('RootProg')
        for _ in range(3):
            apt('cflt')

class HoleDummy:
    def __init__(self, hole_type):
        self.hole_type = hole_type

    def reset(self):
        pass

    def make_lines(self, ex):
        return [self.hole_type]

    def add_struct_token_info(self, ex, tokens, param_rels, struct_tokens, counts):
        tokens.append(self.hole_type)
        struct_tokens.append(self.hole_type)

    def add_deriv_token_info(self, ex, tokens, param_vals, bck_map, counts):
        tokens.append(self.hole_type)
            
class SubSampler:
    def __init__(self, ss_ind, parent, child_type):
        self.ss_ind = ss_ind
        self.parent = parent
        self.child_type = child_type

    def get_struct_deriv(self, ex):
        infos = ['SubProg'] + self.child_sup_struct + self.child_app_struct + ['end']
        struct_tokens = []
        
        _tokens = []
        _param_rels = {}
        _counts = {
            'params': 0,
            'hparams': 0,
            'hole': 0
        }

        for inf in infos:
            if isinstance(inf, str):
                _tokens.append(inf)
                struct_tokens.append(inf)
                continue

            inf.add_struct_token_info(ex, _tokens, _param_rels, struct_tokens, _counts)

        return struct_tokens
            
    def sample_sub_support(self, par_dims):
        
        for sp in self.child_sup_struct:
            sp.sample_cat_decisions()
            
        cov_info = self.child_sup_ci_sfn(self.child_sup_struct)

        for sp in self.child_sup_struct:
            add_loc_place_info(sp, cov_info)
            
        level_info = sample_support_params(self.child_sup_struct, cov_info, par_dims)

        return level_info
        
    def sample_sub_app(self, level_info):

        for ap in self.child_app_struct:
            valid = ap.sample_app_decisions(level_info)
            if not valid:
                self.gs_valid = False
                return
        
    def sample_sub_prog(self, bb_dims):

        if isinstance(self.child_sup_struct[0], HoleDummy):
            return
        
        par_dims = self.parent.getAbsDims(bb_dims)
        level_info = self.sample_sub_support(par_dims)
        self.sample_sub_app(level_info)

    def sample_hole_type(self):

        r = random.random()

        if r < .1:
            return 'empty'
        elif r < .2:
            return 'fill'
        else:
            return 'hole'
                
    def make_hole_choice(self):
        hole_type = self.sample_hole_type()
        if hole_type in ('fill', 'empty'):
            self.child_sup_struct = [HoleDummy(hole_type)]
            self.child_app_struct = []
        elif hole_type == 'hole':
            self.make_child_support_info()
            self.make_child_app_info()
        else:
            assert False
            
        for ss in self.child_sup_struct + self.child_app_struct:
            ss.in_hole = True
        
    def reset(self):

        if self.child_type == 'HOLE':
            self.child_sup_struct = None
            self.child_app_struct = None
            self.make_hole_choice()
        
        for ss in self.child_sup_struct + self.child_app_struct:
            ss.reset()
        
    def make_child_support_info(self):
        sup_struct, cov_info_sample_fn = sample_support_info('sub')
        self.child_sup_struct = sup_struct
        self.child_sup_ci_sfn = cov_info_sample_fn

        for st in self.child_sup_struct:
            st.hier='leaf'
        
    def make_child_app_info(self):
        num_appendix_cubes = random.randint(0,2)
        num_cuboids = len(self.child_sup_struct) + 1         
        self.child_app_struct = []
        
        for i in range(num_appendix_cubes):
            self.child_app_struct.append(GSupType(i+num_cuboids))

        for st in self.child_app_struct:
            st.hier = 'leaf'
            
class Sampler:
    def __init__(self):
        self.gs_valid = True
        self.bbox_info = BBoxInfo()
        self.sub_prog_infos = []
        
    def make_root_support_info(self):
        
        sup_struct, cov_info_sample_fn = sample_support_info('root')        
        
        self.root_sup_struct = sup_struct
        self.root_sup_ci_sfn = cov_info_sample_fn    
        
    def make_root_app_info(self):

        num_appendix_cubes = random.randint(0,2)

        num_cuboids = len(self.root_sup_struct) + 1
         
        self.root_app_struct = []
        
        for i in range(num_appendix_cubes):
            self.root_app_struct.append(GSupType(i+num_cuboids))

    def make_hierarchy(self, num_holes):
        
        OPTS = []
        PROBS = []

        count = 0
        c2s = {}
        
        for st in self.root_sup_struct:
            st.hier='leaf'
            OPTS.append(count)
            c2s[count] = st
            count +=1
            if st.samp_sym_logic is None:
                PROBS.append(3.)
            else:
                PROBS.append(2.)

        for st in self.root_app_struct:
            st.hier = 'leaf'
            OPTS.append(count)
            c2s[count] = st
            count +=1
            if st.samp_sym_logic is None:
                PROBS.append(2.)
            else:
                PROBS.append(1.)

        num_children = min(np.random.choice(NCO,p=NCP), len(OPTS))

        pcinds = np.random.choice(OPTS,p=norm_np(PROBS),size=num_children,replace=False)

        pcinds = pcinds.tolist()
        pcinds.sort()
        parents = [c2s[ci] for ci in pcinds]
        
        pinds = list(range(len(parents)))
        random.shuffle(pinds)
        
        hier_pinds = set(pinds[:num_holes])
        
        for i,par in enumerate(parents):

            if i in hier_pinds:
                ct = 'HOLE'
            else:
                ct = 'hier'

            par.hier = ct
            
            ss = SubSampler(i, par, ct)

            self.sub_prog_infos.append(ss)

        
    def make_sub_infos(self):
        for ss in self.sub_prog_infos:
            if ss.child_type == 'HOLE':
                continue
            ss.make_child_support_info()
            ss.make_child_app_info()
        
    def init_sampler(self, sample_params):
        
        self.make_root_support_info()        
        self.make_root_app_info()

        self.make_hierarchy(sample_params['num_holes'])

        self.make_sub_infos()

    def sample_root_support(self, bbox_dims):

        for sp in self.root_sup_struct:
            sp.sample_cat_decisions()
            
        cov_info = self.root_sup_ci_sfn(self.root_sup_struct)

        for sp in self.root_sup_struct:
            add_loc_place_info(sp, cov_info)
            
        level_info = sample_support_params(self.root_sup_struct, cov_info, bbox_dims)

        return level_info
        
    def sample_root_app(self, level_info):

        for ap in self.root_app_struct:
            valid = ap.sample_app_decisions(level_info)
            if not valid:
                self.gs_valid = False
                return
            
    def sample_root_prog(self):
        self.bbox_info.sample_bbox_dims()

        bbox_dims = self.bbox_info.get_bbox_dims()
        
        level_info = self.sample_root_support(bbox_dims)
        
        self.sample_root_app(level_info)
                
    def init_sample(self):
        self.gs_valid = True
        
        self.bbox_info.reset()

        for ss in self.root_sup_struct + self.root_app_struct:
            ss.reset()
            
        for ss in self.sub_prog_infos:
            ss.reset()
            ss.gs_valid = True
            
    def make_prog(self, ex):
        root_infos = self.root_sup_struct + self.root_app_struct

        lines = []
        
        lines += self.bbox_info.make_line(ex)
        
        for ri in root_infos:
            lines += ri.make_lines(ex)

        lines += ['end']

        for ss in self.sub_prog_infos:
            sub_infos = ss.child_sup_struct + ss.child_app_struct
            lines += ['SubProg']
            for si in sub_infos:
                lines += si.make_lines(ex)            
            lines += ['end']
                
        lines += ['end']
                
        return lines

    def get_struct_token_info(self, ex):
        tokens = []
        param_rels = {}
        struct_tokens = []

        counts = {
            'params': 0,
            'hparams': 0,
            'hole': 0
        }
        
        infos = [self.bbox_info] + self.root_sup_struct + self.root_app_struct + ['end']

        for v in self.sub_prog_infos:
            if v.child_type == 'HOLE':
                continue
            infos += ['SubProg'] + v.child_sup_struct + v.child_app_struct + ['end']

        infos += ['end']

        for inf in infos:
            if isinstance(inf, str):
                tokens.append(inf)
                struct_tokens.append(inf)
                continue

            inf.add_struct_token_info(ex, tokens, param_rels, struct_tokens, counts)
            
        return tokens, param_rels, struct_tokens

    def sample_sub_prog(self, ss):
        bbox_dims = self.bbox_info.get_bbox_dims()
        ss.sample_sub_prog(bbox_dims)

        if not ss.gs_valid:
            self.gs_valid = False        
    
    def sample(self, ex):

        self.init_sample()        

        self.sample_root_prog()

        if not self.gs_valid:
            return None
        
        for ss in self.sub_prog_infos:
            self.sample_sub_prog(ss)
            if not self.gs_valid:
                return None
            
        return self.make_prog(ex)

    def get_deriv_token_info(self, ex):

        tokens = []
        param_vals = {}
        struct_derivs = {}        
        bck_map = {}

        counts = {
            'params': 0,
        }
        
        infos = [self.bbox_info] + self.root_sup_struct + self.root_app_struct + ['end']

        for v in self.sub_prog_infos:
            infos += ['SubProg'] + v.child_sup_struct + v.child_app_struct + ['end']

            if v.child_type == 'HOLE':
                par_hole_name = v.parent.hole_name
                assert par_hole_name is not None

                sd = v.get_struct_deriv(ex)
                struct_derivs[par_hole_name] = sd
            
        infos += ['end']
        
        for inf in infos:
            if isinstance(inf, str):
                tokens.append(inf)
                continue

            inf.add_deriv_token_info(ex, tokens, param_vals, bck_map, counts)

        return tokens, param_vals, struct_derivs, bck_map
            
def sample_group(ex, sample_params):
            
    S = None
    while S is None or not S.gs_valid:
        S = ShapeGroupSampler(ex, sample_params)
        
    return S
