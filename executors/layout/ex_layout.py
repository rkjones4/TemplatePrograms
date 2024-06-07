import matplotlib.pyplot as plt
import sys
import random
from copy import deepcopy
import numpy as np
import torch
import math

sys.path.append('executors')
sys.path.append('executors/common')
sys.path.append('..')
sys.path.append('../common')
import base
import lutils as lu

device=torch.device('cuda')

LAY_CONFIG = {    
    'BASE_TYPE' : 'shape',
    'DEF_STRUCT_TYPES' : ['shape', 'hole'],

    'VDIM': 64,
    'DEF_PRIM_SIZE': 0.5,    
    'MIN_UNIQUE_PIXELS': 8,
    'MIN_VIS_CNT': 8,
    'OUT_THRESH': 0.1,
    'MAX_PARTS': 32,   
    'MAX_SEM_PARTS': 10,    
        
    'TYPED_CAT_REL_PROBS': {
        'swtype': lu.norm_np([50, 50, 0]),
        'shtype': lu.norm_np([50, 50, 0]),
        'mxtype': lu.norm_np([50, 50, 0]),
        'mytype': lu.norm_np([50, 50, 0]),
        'cval': lu.norm_np([40, 30, 30]),
        'pval': lu.norm_np([40, 30, 30]),
        'axis': lu.norm_np([25, 70, 5]),
        'int': lu.norm_np([60, 40, 0]),
    },

    'NUM_HOLE_DIST': (
        [0,1,2],
        lu.norm_np([.65,.25,.1])
    ),


    'NUM_SUB_PROG_DIST': (
        [1,2,3,4,5,6,7,8],
        lu.norm_np([.025,.075, .125, .175, .15, .1, .05, .025])
    ),

    'SPEC_GROUP_SAMPLE_TYPES': ['swtype', 'shtype', 'mxtype', 'mytype', 'sflt', 'mflt'],
    
    'GROUP_FLOAT_STD_RNG': {
        'scale': (0.01, 0.2),
        'move': (0.01, 0.2),
        'symTranslate': (0.01, 0.2),
    },
    
    'SKIP_HOLE_TOKENS': ('start')

}

EX_PTS = None
OUT_THRESH = None
DEF_PRIM_SIZE = None

# Cast coarse to float
def cast_c2f(ctype, cflt, op, ex, div):

    cv = (float(ctype.split('_')[1]) / div)

    if abs(cv) > 0:    
        rv = cv + ex.FLOAT_MAP[cflt]
    else:
        rv = cv
        
    return round(cv, 3), round(rv, 3)
    
# Start LANG Execution Logic

class Primitive:
    def __init__(self, ptype, sem_cls):

        self.ptype = ptype
        self.sem_cls = sem_cls
        self.color = 'grey'
        
        self.W = DEF_PRIM_SIZE
        self.H = DEF_PRIM_SIZE
        self.X = 0.0
        self.Y = 0.0

        self.prim_map = {
            'square': self.get_square_pixels,
            'circle': self.get_circle_pixels,
            'triangle': self.get_triangle_pixels,
        }

        self.lik_map = {
            'square': self.get_square_lik,
            'circle': self.get_circle_lik,
            'triangle': self.get_triangle_lik,
        }

            
    def has_soft_error(self):
        if self.W <= 0.0:
            return True

        if self.H <= 0.0:
            return True

        if self.X - self.W < -1.0 - OUT_THRESH:
            return True

        if self.X + self.W > 1.0 + OUT_THRESH:
            return True

        if self.Y + self.H > 1.0 + OUT_THRESH:
            return True

        if self.Y - self.H < -1.0 - OUT_THRESH:
            return True

        return False
                
    def copy(self):
        n = Primitive(self.ptype, self.sem_cls)
        n.color = self.color
        n.W = self.W
        n.H = self.H
        n.X = self.X
        n.Y = self.Y

        return n

    def get_pixels(self):
        return self.prim_map[self.ptype]()

    def get_lik(self):
        return self.lik_map[self.ptype]()
    
    def get_triangle_pixels(self):
        tpts = EX_PTS - torch.tensor(
            [self.X, self.Y - self.H],
            device = device
        )
        spts = tpts / torch.tensor([self.W, self.H], device=device)

        in_pixels = (spts[:,1] >= 0.0) & (((spts[:,0].abs() * 2) + spts[:,1]) <= 2.0)

        return in_pixels
        
    def get_circle_pixels(self):
        tpts = torch.abs(EX_PTS - torch.tensor([self.X, self.Y],device = device))
        spts = tpts / torch.tensor([self.W, self.H], device=device)
        in_pixels = spts.norm(dim=1) <= 1.0

        return in_pixels                
    
    def get_square_pixels(self):

        tpts = torch.abs(EX_PTS - torch.tensor([self.X, self.Y],device = device))
        spts = tpts / torch.tensor([self.W, self.H], device=device)
        in_pixels = (spts <= 1.0).all(dim=1)

        return in_pixels

    def get_triangle_lik(self):
        tpts = EX_PTS - torch.tensor(
            [self.X, self.Y - self.H],
            device = device
        )
        spts = tpts / torch.tensor([self.W, self.H], device=device)

        lik1 = torch.sigmoid((-1. * self.H) * spts[:,1])
        lik2 = torch.sigmoid((-1. * max(self.W,self.H))  * (((spts[:,0].abs() * 2) + spts[:,1]) - 2.0))

        lik = torch.stack((lik1,lik2),dim=1)

        return lik.min(dim=1).values    
        
    def get_circle_lik(self):
        tpts = torch.abs(EX_PTS - torch.tensor([self.X, self.Y],device = device))
        spts = tpts / torch.tensor([self.W, self.H], device=device)
        return torch.sigmoid((-1. * max(self.W, self.H)) * (spts.norm(dim=1) - 1.0))
    
    def get_square_lik(self):

        tpts = torch.abs(EX_PTS - torch.tensor([self.X, self.Y],device = device))
        spts = tpts / torch.tensor([self.W, self.H], device=device)

        return torch.sigmoid((-1. * max(self.W, self.H)) * (spts.max(dim=1).values - 1.0))
        
    def get_sig(self):
        return (self.ptype, self.color, self.W, self.H, self.X, self.Y)
    
class Shape:
    def __init__(self):
        self.parts = []

    def printInfo(self):
        for i, p in enumerate(self.parts):
            print(f'Prim {i} : {p.ptype} | {p.color} | {p.W} | {p.H} | {p.X} | {p.Y} ')

    def get_sig(self):
        sigs = []
        for p in self.parts:
            sigs.append(p.get_sig())

        sigs.sort()
        return tuple(sigs)


class Program:
    def __init__(self, ex):
        self.state = None
        self.ex = ex
        self.invis_parts = []
        
    def reset(self):
        self._expr = None
        self.state = None
        self.soft_error = False
        self.cmap = {
            'grey': torch.tensor([0.5, 0.5, 0.5], device=device),
            'red': torch.tensor([1.0, 0.0, 0.0], device=device),
            'green': torch.tensor([0.0, 1.0, 0.0], device=device),
            'blue': torch.tensor([0.0, 0.0, 1.0], device=device),            
        }

    def get_state_sig(self):
        return self.state.get_sig()

    def neg_reject(self):
        with torch.no_grad():

            _CMAP = {
                'grey': 1,
                'red': 2,
                'green': 3,
                'blue': 4
            }
            flat_canvas = torch.zeros(                
                self.ex.VDIM * self.ex.VDIM,
                device=device
            ).float()
            
            exp_canvas = torch.zeros(                
                self.ex.VDIM * self.ex.VDIM,
                len(self.state.parts),
                device=device
            ).float()
            
            for i, part in enumerate(self.state.parts):
                occ_pixels = part.get_pixels()
                
                j = _CMAP[part.color]

                diff_col_pixels = (flat_canvas != j)
                
                change_pixels = occ_pixels & diff_col_pixels
                
                exp_canvas[occ_pixels, :] = 0.
                exp_canvas[change_pixels, i] = 1.0
                flat_canvas[occ_pixels] = j

            self.invis_parts = (
                exp_canvas.sum(dim=0) < (self.ex.MIN_VIS_CNT / 2)
            ).nonzero().flatten().tolist()
                
            min_uniq_occ = exp_canvas.sum(dim=0).min().item()
                        
            rej = min_uniq_occ < self.ex.MIN_VIS_CNT            
                
            return rej
                
            
    def has_soft_error(self):    
        if self.soft_error:
            return self.soft_error

        if len(self.state.parts) == 0:
            self.soft_error = True
            return self.soft_error
        
        for part in self.state.parts:
            if part.has_soft_error():
                self.soft_error = True
                return self.soft_error

        return self.soft_error
        
    def get_pixel_value(self, color):
        return self.cmap[color]

    def ex_primitive(self, cmd, sem_cls=None):
        p = Primitive(cmd, sem_cls)
        s = Shape()
        s.parts.append(p)
        return s

    def ex_symReflect(self, S, A):
        new_parts = []

        for r in S.parts:
            n = r.copy()

            if A == 'AX':

                n.X = -1 * n.X

            elif A == 'AY':

                n.Y = -1 * n.Y

            new_parts.append(n)

        S.parts += new_parts

        return S
                    
        
    def ex_symTranslate(self, S, X, Y, K):
        new_parts = []

        for r in S.parts:
            for k in range(1, K+1):
                n = r.copy()

                perc = (k * 1.) / K

                n.X += perc * X
                n.Y += perc * Y

                new_parts.append(n)

        S.parts += new_parts
        return S

    def ex_symRotate(self, S, K):
        new_parts = []

        for r in S.parts:

            xv = np.array([1.0, 0.0])
            rv = np.array([r.X, r.Y])

            rv_norm = np.linalg.norm(rv)

            if rv_norm == 0:
                rv_norm += 1e-8
                self.soft_error = True        
            
            dp = np.arccos(np.dot(xv, rv / rv_norm))

            if r.Y < 0.:
                dp = -1 * dp
                
            for k in range(1, K+1):
                n = r.copy()

                perc = (k * 1.) / (K+1)
                
                incr = perc * (np.pi * 2.)
                
                nv = dp + incr
                
                n.X = np.cos(nv) * rv_norm
                n.Y = np.sin(nv) * rv_norm
                
                new_parts.append(n)
                
        S.parts += new_parts
        return S
        
    
    def ex_move(self, S, X, Y):        
        for p in S.parts:
            p.X += X
            p.Y += Y
            
        return S

    def ex_color(self, S, color):        
        for p in S.parts:
            p.color = color
            
        return S

    def ex_scale(self, S, W, H):
        
        for p in S.parts:
            p.W *= W
            p.H *= H
            
        return S

    def ex_union(self, A, B):
        s = Shape()
        s.parts += A.parts + B.parts
        return s
                            
    def _execute(self, fn, params):

        if '!' in fn:
            assert len(params) == 1
            sem_cls = int(fn.split('!')[1])
            fn = fn.split('!')[0]
            assert fn == 'prim'
            assert params[0] in self.ex.D_PRIMS
            return self.ex_primitive(params[0], sem_cls)
        
        elif fn == 'prim':
            assert len(params) == 1
            params = [self.execute(p) for p in params]
            assert params[0] in self.ex.D_PRIMS
            return self.ex_primitive(params[0])

        elif fn == 'move':            
            params = [self.execute(p) for p in params]

            
            assert len(params) == 5
            flt_params = [
                self.ex.UCF_MAP['mxtype'][(params[0], params[2])],
                self.ex.UCF_MAP['mytype'][(params[1], params[3])]
            ]
            shape = params[4]                
                
            assert isinstance(shape, Shape)
            return self.ex_move(shape, flt_params[0], flt_params[1])

        elif fn == 'scale':            
            params = [self.execute(p) for p in params]
            
            assert len(params) == 5
            flt_params = [
                self.ex.UCF_MAP['swtype'][(params[0], params[2])],
                self.ex.UCF_MAP['shtype'][(params[1], params[3])]
            ]
            shape = params[4]
                
            assert isinstance(shape, Shape)
            return self.ex_scale(shape, flt_params[0], flt_params[1])

        elif fn == 'color':
            params = [self.execute(p) for p in params]            
            assert len(params) == 2
            assert isinstance(params[1], Shape)
            assert params[0] in self.ex.D_COLORS
            
            return self.ex_color(params[1], params[0])
            
        elif fn == 'union':
            params = [self.execute(p) for p in params]
            assert isinstance(params[0], Shape)
            assert isinstance(params[1], Shape)
            assert len(params) == 2
            return self.ex_union(params[0], params[1])        
 
        elif fn == 'symReflect':
            params = [self.execute(p) for p in params]
            assert len(params) == 2
            assert isinstance(params[1], Shape)
            assert params[0] in ('AX', 'AY')
            return self.ex_symReflect(params[1], params[0])

        elif fn == 'symTranslate':
            params = [self.execute(p) for p in params]                        

            
            assert len(params) == 6
            flt_params = [
                self.ex.UCF_MAP['mxtype'][(params[0], params[2])],
                self.ex.UCF_MAP['mytype'][(params[1], params[3])]
            ]
            k = int(params[4])
            shape = params[5]
                
            assert isinstance(shape, Shape)
            return self.ex_symTranslate(
                shape, flt_params[0], flt_params[1], k
            )

        elif fn == 'symRotate':            
            params = [self.execute(p) for p in params]
            assert len(params) == 2
            assert isinstance(params[1], Shape)
            return self.ex_symRotate(
                params[1], int(params[0])
            )
                        
        elif fn in self.ex.TLang.params:
            return fn
        
        else:

            try:
                float(fn)
                return fn
            
            except:
                pass    
            
            assert False, f'bad function {fn}'
    
    def execute(self, expr):

        if not isinstance(expr, list):
            try:
                float(expr)
            except:
                assert expr.split('!')[0] in self.ex.TLang.params, f'bad token {expr}'
            return self._execute(expr, [])
            
        fn = expr[0]        
        
        ipc = self.ex.TLang.get_num_inp(fn.split('!')[0])
        
        params = []        

        cur = []

        pc = 0

        
        for c in expr[1:]:
            
            cur.append(c)

            if pc > 0:            
                pc -= 1
            
            cipc = self.ex.TLang.get_num_inp(c.split('!')[0])

            pc += cipc
            
            if pc == 0:
                if len(cur) == 1:
                    params.append(cur[0])
                else:
                    params.append(cur)
                    
                cur = []

        if len(cur) > 0:
            params.append(cur)
            
        assert len(params) == ipc
        assert pc == 0

        o = self._execute(fn, params)
        if isinstance(o, Shape):
            if len(o.parts) >= self.ex.MAX_PARTS:
                assert False, 'too many parts, likely bad prog'
            
        return o

    def make_image(self):

        canvas = torch.zeros(self.ex.VDIM * self.ex.VDIM, 3, device=device).float()
        for part in self.state.parts:
            pixels = part.get_pixels()
            p_val = self.get_pixel_value(part.color)
            canvas[pixels] = p_val

        return canvas.reshape(self.ex.VDIM, self.ex.VDIM, 3)

    def make_sem_image(self):
        canvas = torch.zeros(self.ex.VDIM * self.ex.VDIM, device=device).float() - 1
        for part in self.sem_state.parts:
            pixels = part.get_pixels()
            canvas[pixels] = part.sem_cls
            
        img = lu.color_votes(canvas.view(self.ex.VDIM, self.ex.VDIM), self.ex.MAX_SEM_PARTS)
        
        return img

    def make_sem_seg(self):
        canvas = torch.zeros(self.ex.VDIM * self.ex.VDIM, self.ex.MAX_SEM_PARTS, device=device).float()
        
        for part in self.sem_state.parts:
            pixels = part.get_pixels()
            canvas[pixels, :] = 0.0
            canvas[pixels, part.sem_cls] = 1.0
            
        return canvas.view(self.ex.VDIM, self.ex.VDIM, self.ex.MAX_SEM_PARTS)
            
    
    def make_new_part_pred(self, max_parts):

        canvas = torch.zeros(self.ex.VDIM * self.ex.VDIM, max_parts, device=device).float()

        for part in self.state.parts:
            liks = part.get_lik().float()

            pixels = part.get_pixels()
            
            sem_cls = part.sem_cls

            assert sem_cls is not None

            canvas[pixels.nonzero().flatten(),:] = 0.0
            canvas[pixels.nonzero().flatten(), sem_cls] = 2.0 

            out_pixels = (~pixels).nonzero().flatten()
            
            canvas[out_pixels,sem_cls] = torch.stack((liks[out_pixels], canvas[out_pixels, sem_cls]),dim=1).max(dim=1).values
            
        return canvas

    
    def render(self, name=None):

        with torch.no_grad():
        
            # 64 x 64 x 3 image
            img = self.make_image()

            plt.imshow(img.cpu().numpy(), origin='lower')
        
            if name is not None:
                plt.savefig(f'{name}.png')
            else:
                plt.show()
                            
    def run(self, expr):

        self.reset()
        self._expr = expr

        if expr[0] == self.ex.START_TOKEN:
            expr = expr[1:]
            
        self.state = self.execute(expr)        


    def add_sem_info(self, sem_info):
        self.sem_state = Shape()
        for si in sem_info:

            prim_type = si[0]

            color = si[5]
            top_tokens = si[6]

            sem_ind = si[7]

            scale_tokens = self.ex.make_scale_tokens([
                self.ex.find_closest_float_val(si[1] * 2., 'swtype'),
                self.ex.find_closest_float_val(si[2] * 2., 'shtype'),
            ])
            move_tokens = self.ex.make_move_tokens([
                self.ex.find_closest_float_val(si[3], 'mxtype'),
                self.ex.find_closest_float_val(si[4], 'mytype'),
            ])        

            if color != 'grey':
                tokens = ['color', color]
            else:
                tokens = []
            tokens += move_tokens + scale_tokens + ['prim', prim_type]

            if top_tokens is not None:
                on_top = []
                if 'symTranslate' in top_tokens:
                    v1 = top_tokens[1]
                    v2 = top_tokens[2]
                    K = top_tokens[3]
                    params = [
                        self.ex.find_closest_float_val(v1, 'mxtype'),
                        self.ex.find_closest_float_val(v2, 'mytype'),
                    ]
                    sym_tokens = self.ex.make_symtranslate_tokens(params + [K])
                    on_top = sym_tokens
                else:
                    on_top = list(top_tokens)
                    
                tokens = on_top + tokens

            _state = self.execute(tokens)

            assert sem_ind < self.ex.MAX_SEM_PARTS
            
            for prim in _state.parts:
                prim.sem_cls = sem_ind
                self.sem_state.parts.append(prim)

        for prim, sprim in zip(self.state.parts, self.sem_state.parts):

            if prim.W != sprim.W or prim.H != sprim.H:
                sprim.W = prim.W
                sprim.H = prim.H
                
                            
class LayExecutor(base.BaseExecutor):
    def __init__(self, config = None):
        self.ex_name = 'lay'        
        if config is not None:
            LAY_CONFIG.update(config)
            
        LAY_CONFIG['FLOAT_PARAM_TYPES'] = ['sflt', 'mflt']
        self.last_sgs_type_map = {}
                        
        self.prog_cls = Program
        self.base_init(LAY_CONFIG)

        self.make_lang()
        self.init_pts()
        self.set_globals()

    # returns a float
    def find_closest_float_val(self, v, op):
        ind = (self.UCF_TD_FLTS[op] - v).abs().argmin()
        rv =  self.UCF_D_FLTS[op][ind]                        
        return rv
            

    def make_scale_tokens(self, params):

        assert len(params) == 2
        if params[0] == 1.0 and params[1] == 1.0:
            return []
        
        p1n,p1f = self.REV_UCF_MAP['swtype'][params[0]]
        p2n,p2f = self.REV_UCF_MAP['shtype'][params[1]]
        return ['scale', p1n, p2n, p1f, p2f]

    def make_move_tokens(self, params):

        assert len(params) == 2
        if params[0] == 0.0 and params[1] == 0.0:
            return []
        
        p1n,p1f = self.REV_UCF_MAP['mxtype'][params[0]]
        p2n,p2f = self.REV_UCF_MAP['mytype'][params[1]]
        return ['move', p1n, p2n, p1f, p2f]

    def make_symtranslate_tokens(self, params):
        assert len(params) == 3
        if params[0] == 0.0 and params[1] == 0.0:
            return []

        
        p1n,p1f = self.REV_UCF_MAP['mxtype'][params[0]]
        p2n,p2f = self.REV_UCF_MAP['mytype'][params[1]]
        O =  ['symTranslate', p1n, p2n, p1f, p2f, str(params[2])]
        return O 
                        
    def vis_on_axes(self, ax, img):
        ax.imshow(img, origin='lower')
        
    def execute(self, expr, vis=False):
        tokens = expr.split()
        
        assert tokens[0] == self.START_TOKEN
    
        P = Program(self)
        P.run(tokens)
    
        with torch.no_grad():
            img = P.make_image()
                
        if vis:
            plt.imshow(img.cpu().numpy(), origin='lower')
            plt.show()
        
        else:
            return img

    def ex_prog(self, tokens):
        P = Program(self)
        P.run(tokens)
        return P

    def make_new_part_pred(self, expr):
        tokens = expr.split()
        
        assert tokens[0] == self.START_TOKEN

        P = Program(self)
        P.run(tokens)
        with torch.no_grad():
            return P.make_new_part_pred(expr.count('!')+1)
    
    def set_globals(self):
        global OUT_THRESH
        OUT_THRESH = self.OUT_THRESH
        global DEF_PRIM_SIZE
        DEF_PRIM_SIZE = self.DEF_PRIM_SIZE
        
        
    def init_pts(self):
        a = (torch.arange(self.VDIM).float() * 2 / self.VDIM) - 1.0 + (1./self.VDIM)
        c = a.unsqueeze(0).repeat(self.VDIM, 1)
        d = a.unsqueeze(1).repeat(1, self.VDIM)
        pts = torch.stack((c,d), dim=2).view(-1, 2).to(device)
        global EX_PTS
        EX_PTS = pts
        
    def make_lang(self):
        self.add_token_info()
        self.set_tlang()

    def set_tlang(self):
        TLang = base.TokenLang(self)
        
        TLang.add_token(self.START_TOKEN, 'shape', 'prog')
        TLang.add_token(self.HOLE_TOKEN, '', 'hole', 'out_only')
        TLang.add_token('union', 'shape,shape', 'shape')
                
        TLang.add_token('scale', 'swtype,shtype,sflt,sflt,shape', 'shape')
        TLang.add_token('move', 'mxtype,mytype,mflt,mflt,shape', 'shape')
            
        TLang.add_token('color', 'cval,shape', 'shape')
        TLang.add_token('prim', 'pval', 'shape')
        TLang.add_token('symReflect', 'axis,shape', 'shape')

        
        TLang.add_token('symTranslate', 'mxtype,mytype,mflt,mflt,int,shape', 'shape')
            
        TLang.add_token('symRotate', 'int,shape', 'shape')
                
        for t in self.D_PRIMS:
            TLang.add_token(t, '', 'pval')

        for t in self.D_COLORS:
            TLang.add_token(t, '', 'cval')

        for t in self.D_AXIS:
            TLang.add_token(t, '', 'axis')

        for t in self.D_INTS:
            TLang.add_token(str(t), '', 'int')

            
        self.TMAP = {}
        self.FLOAT_MAP = {}
        self.REV_FLOAT_MAP = {}

        # maintain indent
        if True:
            for vals, typ in [
                (self.D_SWTYPES, 'swtype'),
                (self.D_SHTYPES, 'shtype'),
                (self.D_MXTYPES, 'mxtype'),
                (self.D_MYTYPES, 'mytype'),
            ]:
                for v in vals:
                    TLang.add_token(v, '', typ)
        
            for vals, name in [
                (self.D_SFLT, 'sflt'),
                (self.D_MFLT, 'mflt'),
            ]:
            
                self.TMAP[name] = (torch.tensor(vals), vals)
    
                for i,val in enumerate(vals):
                    TLang.add_token(f'{name}_{i}', '', name)
                    self.FLOAT_MAP[f'{name}_{i}'] = val
                    self.REV_FLOAT_MAP[(name, val)] = f'{name}_{i}'

            self.UCF_MAP = {}
            self.REV_UCF_MAP = {}
            self.CTM = {}
            self.CT_D_FLTS = {}
            self.CT_TD_FLTS = {}

            for op, op_type_names, op_flt_name, op_flt_vals, div in [
                ('swtype', self.D_SWTYPES, 'sflt', self.D_SFLT, 100.),
                ('shtype', self.D_SHTYPES, 'sflt', self.D_SFLT, 100.),
                ('mxtype', self.D_MXTYPES, 'mflt', self.D_MFLT, 1000.),
                ('mytype', self.D_MYTYPES, 'mflt', self.D_MFLT, 1000.),                    
            ]:

                self.CT_D_FLTS[op] = [
                    self.REV_FLOAT_MAP[(op_flt_name, fv)] for fv in op_flt_vals
                ]
                self.CT_TD_FLTS[op] = torch.tensor(op_flt_vals)
                
                self.UCF_MAP[op] = {}
                self.REV_UCF_MAP[op] = {}
                
                for type_name in op_type_names:
                                                                               
                    for flt_val in op_flt_vals:
                        flt_name = self.REV_FLOAT_MAP[(op_flt_name, flt_val)]

                        cval, val = cast_c2f(type_name, flt_name, op_flt_name, self, div)

                        if type_name not in self.CTM:
                            self.CTM[type_name] = cval
                            
                        k = (type_name, flt_name)

                        self.UCF_MAP[op][k] = val
                        if val not in self.REV_UCF_MAP[op]:
                            self.REV_UCF_MAP[op][val] = k

            self.UCF_D_FLTS = {op: list(M.keys()) for op, M in self.REV_UCF_MAP.items()}
            self.UCF_TD_FLTS = {op: torch.tensor(M) for op,M in self.UCF_D_FLTS.items()}

            self.G_MIN_SCALE = min(self.UCF_D_FLTS['swtype'])
            
        for t in self.D_PARAM_LOCS:
            TLang.add_token(t, '', 'param_loc', 'inp_only')
 
        for t in self.D_STRUCT_LOCS:
            TLang.add_token(t, '', 'struct_loc', 'inp_only')

        TLang.init()
        
        self.TLang = TLang        

    def make_sgs_type_sampler(self, lt_PC, PC, ltpt, mu, stddev):
        def sample_fn():

            assert PC.next_val is None

            tar_val = mu + (np.random.randn() * stddev)
                                    
            if lt_PC.relation is None:
                
                ret_val, next_flt_val = self.REV_UCF_MAP[ltpt][
                    self.find_closest_float_val(tar_val, ltpt)
                ]

                PC.next_val = next_flt_val
                
            else:
                if lt_PC.relation[0] == 'static':
                    ret_val =  lt_PC.relation[1]
            
                elif lt_PC.relation[0] == 'reuse':
                    ret_val = lt_PC.context_vals[lt_PC.relation[1]]

                else:
                    assert False
                                    
                resid_tar_val = tar_val - self.CTM[ret_val]
                
                PC.next_val = self.CT_D_FLTS[ltpt][
                    (self.CT_TD_FLTS[ltpt] - resid_tar_val).abs().argmin().item()
                ]
                            
            return ret_val
                
        return sample_fn
        
    def make_spec_group_sampler(self, PC, val, pt, pfn, pp, prev_params, skip_params):        
        if pt in self.FLOAT_PARAM_TYPES:

            if pfn in ('move', 'symTranslate'):
                if pp == 2:
                    ltpt = 'mxtype'
                elif pp == 3:
                    ltpt = 'mytype'
                else:
                    assert False
        
            elif pfn == 'scale':
                if pp == 2:
                    ltpt = 'swtype'
                elif pp == 3:
                    ltpt = 'shtype'
                else:
                    assert Fales
            else:
                assert False
            
            lt_PC, lt_val =  self.last_sgs_type_map.pop(ltpt)

            ok = (lt_val, val)

            mu = self.UCF_MAP[ltpt][ok]

            lr, up = self.GROUP_FLOAT_STD_RNG[pfn]
            a = random.random()
            stddev = (lr * a) + (up * (1-a))

            PC.next_val = None
            
            lt_PC.sample_fn = self.make_sgs_type_sampler(lt_PC, PC, ltpt, mu, stddev)

            def pc_sample_fn():
                assert PC.next_val is not None
                ret_val = PC.next_val
                PC.next_val = None
                return ret_val

            return pc_sample_fn
            
        else:
            assert pt not in self.last_sgs_type_map
            self.last_sgs_type_map[pt] = (PC, val)
            PC.sample_cat_relation(val, pt, prev_params, skip_params)            
            return 'temp'
        
        
    def add_token_info(self):
        
        self.D_PRIMS = ('circle', 'square', 'triangle')
        self.D_COLORS = ('red', 'green', 'blue')
        self.D_AXIS = ('AX', 'AY')
        self.D_INTS = tuple([i for i in range(1, 7)])

        self.D_PARAM_LOCS = tuple(
            [f'{self.PARAM_LOC_TOKEN}_{i}' for i in range(self.MAX_PARAM_TOKENS)]
        )
        self.D_STRUCT_LOCS = tuple(
            [f'{self.STRUCT_LOC_TOKEN}_{i}' for i in range(self.MAX_HOLES)]
        )
                
        self.D_SFLT = [-.15, -.1, -.05, 0., .05, 0.1, .15]
        self.D_MFLT = [-.125, -.075, -0.025,  0.025, 0.075, .125]                        
        self.D_SWTYPES = [f'swtype_{i}' for i in ('20', '55', '90','125','160','195')] # div by 100
        self.D_SHTYPES = [f'shtype_{i}' for i in ('20', '55', '90','125','160','195')] # div by 100
        self.D_MXTYPES = [f'mxtype_{i}' for i in ('-775','-475','-175','0','175','475','775')] # div by 1000
        self.D_MYTYPES = [f'mytype_{i}' for i in ('-775','-475','-175','0','175','475','775')] # div by 1000
        self.DEF_PARAM_TYPES = ['sflt', 'mflt', 'swtype', 'shtype', 'mxtype', 'mytype']                                        

        self.DEF_PARAM_TYPES += ['cval','pval','axis','int']
        self.PRT_FNS = ['prim']

        
    def get_input_shape(self):
        return [self.VDIM, self.VDIM, 3]

    def get_group_sample_params(self):
        return {
            'num_holes': lu.sample_dist(self.NUM_HOLE_DIST),
            'num_sub_progs': lu.sample_dist(self.NUM_SUB_PROG_DIST)
        }
    
    def check_valid_tokens(self, tokens, ret_vdata=False, ret_prog=False):
        
        P = Program(self)

        try:
            P.run(tokens)
        except Exception as e:            
            if ret_prog:
                return None, None
                
            return None

        if ret_prog:        
            return self.check_valid_prog(P, ret_vdata), P

        return self.check_valid_prog(P, ret_vdata)
    
    def check_valid_prog(self, P, ret_vdata=False):

        if P.has_soft_error():
            return None

        if P.neg_reject():
            return None

        if not ret_vdata:
            return True
        
        try:
            img = P.make_image()
        except Exception as e:
            img = None

        return img
    
    def sample_det_prog(self, sample_params):
        num_sub_progs = sample_params['num_sub_progs']

        sampled_prims = [sample_prim(self) for _ in range(num_sub_progs)]
        
        orig_prog = comb_sub_progs([prim.get_prog() for prim in sampled_prims])
                        
        ordered_prims = reorder_prims(sampled_prims, self)

        if len(ordered_prims) < len(sampled_prims):        
            valid = self.check_valid_tokens(orig_prog)
            if valid:
                ordered_prims = sampled_prims
            
        sub_progs = []
        for oprim in ordered_prims:
            sub_progs.append(oprim.get_prog())
                
        comb_prog = comb_sub_progs(sub_progs)
                    
        return comb_prog
        
    def render_group(self, images, name=None, rows=1):
        if rows == 1:
            f, axarr = plt.subplots(rows,len(images),figsize=(30,3))
            for i in range(len(images)):
                axarr[i].axis("off")
                
                if images[i] is not None:
                    axarr[i].imshow(images[i].cpu().numpy(), origin='lower')
                
        else:
            num_per_row = math.ceil(len(images) / rows)
            f, axarr = plt.subplots(rows, num_per_row, figsize=(30,3 * rows))
            j = 0
            k = 0
            
            for i in range(len(images)):
                axarr[k][j].axis("off")

                if images[i] is not None:
                    axarr[k][j].imshow(images[i].cpu().numpy(), origin='lower')
                            
                j += 1

                if j == num_per_row:
                    k += 1
                    j = 0
            
        if name is None:
            plt.show()
        else:
            plt.savefig(f'{name}.png')
            
        plt.close()


    def build_struct_scope(self, context, token):

        context.name = token

        local = parse_local_scope(self, context.tokens)
        
        context.struct_scope = {
            'local': local
        }    
        
    def group_sample_sub_prog(self, scope, counts):
        return group_sample_sub_prog(self, scope, counts)


#################
#################
### Random Program Sampling Logic
#################
#################

def comb_sub_progs(sub_shapes):
    return ['START'] + _comb_sub_progs(sub_shapes)

def _comb_sub_progs(ss):
    if len(ss) == 1:
        return ss[0]
    
    return ['union'] + ss[0]  + _comb_sub_progs(ss[1:])    


def reorder_prims(sampled_prims, ex):

    # list of prims
    canvas = []

    q = [(sp, False) for sp in sampled_prims]
    
    while len(q) > 0:
        sp, pa = q.pop(0)
        if len(canvas) == 0:
            canvas.append(sp)
            continue
        
        bad_colors = set()
        covered = []

        for i, cp in enumerate(canvas):
            # sI is percent of newly added that is covered
            # cI is precent of previous added that is covered

            R = calc_overlap(sp, cp)

            sI, cI, sA, cA = R
            if sI > .65:
                bad_colors.add(cp.color_params)

            if cA < sA and cI > .75:
                covered.append(i)

        if sp.color_params not in bad_colors and len(covered) == 0:
            canvas.append(sp)
            continue

        good_colors = [
            c for c in ('red', 'green', 'blue', 'grey') if c not in bad_colors
        ]

        if sp.color_params in bad_colors:
            if len(good_colors) > 0:                
                sp.color_params = random.choice(good_colors)
            else:
                continue

        if len(covered) == 0:
            canvas.append(sp)
            continue

        if pa:
            continue

        rmvd = [(c,True) for i,c in enumerate(canvas) if i in covered]
        canvas = [c for i,c in enumerate(canvas) if i not in covered]
        canvas.append(sp)
        q += rmvd

    return canvas


class PrimGroup:
    def __init__(self, ex, local=None):

        self.top_info = []                
        self.ex = ex
        if local is not None:
            return self.in_context_init(ex, local)
            
        if random.random() < 0.75:
            self.color_params = random.choice(ex.D_COLORS)
        else:
            self.color_params = 'grey'
        
        r = random.random()
        if r < 0.85:
            self.num_prims = 1
        elif r < 0.95:
            self.num_prims = 2
        else:
            self.num_prims = 3
            
        self.prims = []
        # left, right, bot, top extents
        self.bboxes = torch.ones(self.num_prims, 4).float() * 1.1

        for _ in range(self.num_prims):

            # pbbox = (4)
            prim, pbbox = sample_move_scale_prim(ex)

            cbboxes = torch.stack((pbbox.unsqueeze(0).repeat(self.num_prims, 1), self.bboxes),dim=1)
            # num_prims x 2 x 4

            lhs = torch.relu(cbboxes[:,:,1].min(dim=1).values - cbboxes[:,:,0].max(dim=1).values)
            rhs = torch.relu(cbboxes[:,:,3].min(dim=1).values - cbboxes[:,:,2].max(dim=1).values)
            intersect = lhs * rhs

            min_area = ((cbboxes[:,:,1] - cbboxes[:,:,0]) * (cbboxes[:,:,3] - cbboxes[:,:,2])).min(dim=1).values
            
            max_perc_intersect = (intersect / (min_area+1e-8)).max()

            if max_perc_intersect > 0.5:
                continue
                        
            self.bboxes[len(self.prims)] = pbbox
            self.prims.append(prim)


    def in_context_init(self, ex, local):

        self.color_params = 'grey'
        if 'color' not in local:        
            if random.random() < 0.75:
                self.color_params = random.choice(ex.D_COLORS)

        r = random.random()
        if r < 0.85:
            self.num_prims = 1
        elif r < 0.95:
            self.num_prims = 2
        else:
            self.num_prims = 3
            
        self.prims = []
        # left, right, bot, top extents
        self.bboxes = torch.ones(self.num_prims, 4).float() * 1.1

        up_inds = [i for i,t in enumerate(local) if t == 'union' or 'sym' in t]
        if len(up_inds) > 0:
            last_up_ind = max(up_inds)        
            after_up_ind = local[last_up_ind:]
        else:
            after_up_ind = local

        skip_move = False
        skip_scale = False
        
        if 'move' in after_up_ind:
            skip_move = True

        if 'scale' in after_up_ind:
            skip_scale = True
            
        for _ in range(self.num_prims):
            # pbbox = (4)
            prim, pbbox = sample_move_scale_prim(ex)

            if skip_scale or skip_move:
                if skip_scale:
                    prim.scale_params = (1., 1.)
                if skip_move:
                    prim.move_params = (0., 0.)

                pbbox = prim.get_bbox()
            
            cbboxes = torch.stack((pbbox.unsqueeze(0).repeat(self.num_prims, 1), self.bboxes),dim=1)
            # num_prims x 2 x 4

            lhs = torch.relu(cbboxes[:,:,1].min(dim=1).values - cbboxes[:,:,0].max(dim=1).values)
            rhs = torch.relu(cbboxes[:,:,3].min(dim=1).values - cbboxes[:,:,2].max(dim=1).values)
            intersect = lhs * rhs

            min_area = ((cbboxes[:,:,1] - cbboxes[:,:,0]) * (cbboxes[:,:,3] - cbboxes[:,:,2])).min(dim=1).values
            
            max_perc_intersect = (intersect / (min_area+1e-8)).max()

            if max_perc_intersect > 0.5:
                continue
                        
            self.bboxes[len(self.prims)] = pbbox
            self.prims.append(prim)
            
            
    def sample_sym(self, ex):
                        
        sym_opts = self.check_sym_opts()

        if len(sym_opts) == 0:            
            return
        
        sym_choice = random.choice(sym_opts)

        if 'ref' in sym_choice:
            self.add_ref_sym(sym_choice)
            if len(self.top_info) > 0:
                self.sample_extra_move(ex)
        elif 'rot' in sym_choice:
            self.add_rot_sym(sym_choice)
            if len(self.top_info) > 0:
                self.sample_extra_move(ex)
        elif 'trans' in sym_choice:
            self.add_trans_sym(sym_choice, ex)


    def check_sym_opts(self):
        left_extent = self.bboxes[:,0].min().item()
        right_extent = self.bboxes[:,1].max().item()
        bot_extent = self.bboxes[:,2].min().item()
        top_extent = self.bboxes[:,3].max().item()

        so = []

        if left_extent > 0. or right_extent < 0.:
            so.append('ref_AX')

        if bot_extent > 0. or top_extent < 0.:
            so.append('ref_AY')

        if len(so) == 2:
            so.append('rot')
        elif 'ref_AX' in so and\
             max(abs(right_extent), (left_extent)) + ((top_extent-bot_extent)/2.) < 1.0:
            so.append('rot')
        elif 'ref_AY' in so and\
             max(abs(bot_extent), (top_extent)) + ((right_extent-left_extent)/2.) < 1.0:
            so.append('rot')

        to = set()
        if left_extent > 0:
            to.add('left')

        if right_extent < 0:
            to.add('right')

        if bot_extent > 0:
            to.add('bot')

        if top_extent < 0:
            to.add('top')

        if len(to) > 0:
            so.append(f'trans_{"+".join(list(to))}')

        if 'symReflect' in self.top_info:
            so = [s for s in so if 'ref' not in s]

        if 'symRotate' in self.top_info:
            so = [s for s in so if 'rot' not in s]

        if 'symTranslate' in self.top_info:
            so = [s for s in so if 'trans' not in s]

        self.last_sym_check = so
        return so
        

    def add_ref_sym(self, sc):
        if 'AX' in sc:
            self.top_info = ['symReflect', 'AX'] + self.top_info

            rbboxes = self.bboxes.clone()

            rbboxes[:,0] = -1 * self.bboxes[:,1]
            rbboxes[:,1] = -1 * self.bboxes[:,0]

            self.bboxes = torch.cat((
                self.bboxes,
                rbboxes
            ),dim=0)
            
        elif 'AY' in sc:
            self.top_info = ['symReflect', 'AY'] + self.top_info

            rbboxes = self.bboxes.clone()

            rbboxes[:,2] = -1 * self.bboxes[:,3]
            rbboxes[:,3] = -1 * self.bboxes[:,2]

            self.bboxes = torch.cat((
                self.bboxes,
                rbboxes
            ),dim=0)
            
        else:
            assert False, f' bad {sc}'
        

    def add_rot_sym(self, sc):    

        K = random.randint(2,6)
        new_boxes = []
        for bbox in self.bboxes:

            X = (bbox[1] + bbox[0]) / 2.
            Y = (bbox[3] + bbox[2]) / 2.
            hW = (bbox[1] - bbox[0]) / 2.
            hH = (bbox[3] - bbox[2]) / 2.

            xv = np.array([1.0, 0.0])
            rv = np.array([X, Y])

            rv_norm = np.linalg.norm(rv)

            if rv_norm == 0:
                return

            dp = np.arccos(np.dot(xv, rv / rv_norm))

            if Y < 0.:
                dp = -1 * dp
                
            for k in range(1, K+1):
                
                perc = (k * 1.) / (K+1)
                
                incr = perc * (np.pi * 2.)
                
                nv = dp + incr
                
                nX = np.cos(nv) * rv_norm
                nY = np.sin(nv) * rv_norm

                nbbox = torch.tensor([
                    nX - hW,
                    nX + hW,
                    nY - hH,
                    nY + hH
                ])

                if nbbox.abs().max() > 1.0:
                    return

                new_boxes.append(nbbox)

        self.bboxes = torch.cat((
            self.bboxes,
            torch.stack(new_boxes,dim=0)
        ),dim=0)
        
        self.top_info = ['symRotate', str(K)] + self.top_info

    def parse_trans_dir(self, dr):
        left_extent = self.bboxes[:,0].min()
        right_extent = self.bboxes[:,1].max()
        bot_extent = self.bboxes[:,2].min()
        top_extent = self.bboxes[:,3].max()
        
        if dr == 'right':
            ind = 0
            min_incr = right_extent - left_extent
            max_ext = 1 - right_extent
        elif dr == 'left':
            ind = 0
            min_incr = left_extent - right_extent
            max_ext = -1 - left_extent
        elif dr == 'top':
            ind = 1
            min_incr = top_extent - bot_extent
            max_ext = 1 - top_extent
        elif dr == 'bot':
            ind = 1
            min_incr = bot_extent - top_extent
            max_ext = -1 - bot_extent
        else:
            assert False, f'bad dr {dr}'

        return ind, min_incr.item(), max_ext.item()
            
    def add_trans_sym(self, sc, ex):        


        dos = sc.split('_')[1].split('+')

        random.shuffle(dos)

        main_dir = dos.pop(0)

        if len(dos) > 0:
            sec_dir = dos.pop(0)
        else:
            sec_dir = None

        main_ind, main_min_incr, main_max = self.parse_trans_dir(main_dir)
        if sec_dir is not None:
            sec_ind, _, sec_max = self.parse_trans_dir(sec_dir)
        else:
            sec_ind = int(not bool(main_ind))
            sec_max = 0.
            
        if abs(main_min_incr) > abs(main_max):
            return

        max_K = min(abs(main_max) // abs(main_min_incr), 3)

        K = random.randint(1, max_K)
        main_max_incr = main_max / K

        r = random.random()
        
        main_incr = (main_min_incr * r) + ((1-r) * main_max_incr)

        if random.random() < 0.5:
            sec_incr = 0.
        else:
            q = random.random()
            sec_incr = q * (sec_max / K)

        params = [None, None]
        params[main_ind] = main_incr * K
        params[sec_ind] = sec_incr * K

        assert None not in params

        params = [
            ex.find_closest_float_val(params[0], 'mxtype'),
            ex.find_closest_float_val(params[1], 'mytype'),
        ]
        
        ## update bboxes

        new_boxes = []

        dX = float(params[0])
        dY = float(params[1])
        
        for bbox in self.bboxes:

            X = (bbox[1] + bbox[0]) / 2.
            Y = (bbox[3] + bbox[2]) / 2.
            hW = (bbox[1] - bbox[0]) / 2.
            hH = (bbox[3] - bbox[2]) / 2.

            for k in range(1, K+1):
                perc = (k * 1.) / K

                nX = X + (perc * dX)
                nY = Y + (perc * dY)

                nbbox = torch.tensor([
                    nX - hW,
                    nX + hW,
                    nY - hH,
                    nY + hH
                ])

                if nbbox.abs().max() > 1.0:
                    return

                new_boxes.append(nbbox)
                
        self.bboxes = torch.cat((
            self.bboxes,
            torch.stack(new_boxes,dim=0)
        ),dim=0)

        sym_tokens = self.ex.make_symtranslate_tokens(params + [K])
        
        self.top_info = sym_tokens + self.top_info
        
    def sample_extra_move(self, ex):

        if random.random() > 0.35:
            return
        
        left_extent = self.bboxes[:,0].min()
        right_extent = self.bboxes[:,1].max()
        bot_extent = self.bboxes[:,2].min()
        top_extent = self.bboxes[:,3].max()

        max_left = -1 - left_extent
        max_right = 1 - right_extent
        max_bot = -1 - bot_extent
        max_top = 1 - top_extent

        if max([
            abs(max_left),
            abs(max_right),
            abs(max_bot),
            abs(max_top)
        ]) < ex.G_MIN_SCALE:
            return

        move_tokens = []

        for _ in range(5):
            xr = random.random()
            yr = random.random()
            
            x_move = max_left * xr + (max_right * (1-xr))
            y_move = max_bot * yr + (max_top * (1-yr))
            
            params = [
                ex.find_closest_float_val(x_move, 'mxtype'),
                ex.find_closest_float_val(y_move, 'mytype'),
            ]

            move_tokens = self.ex.make_move_tokens(params)
            
            if len(move_tokens) > 0:
                break
                        
        self.top_info = move_tokens + self.top_info

            
    def get_prog(self):
            
        tokens = deepcopy(self.top_info)

        if self.color_params != 'grey':
            tokens += ['color', self.color_params]
            
        sub_progs = []
        for prim in self.prims:
            sub_progs.append(prim.get_tokens())

        while len(sub_progs) > 0:
            sp = sub_progs.pop(0)

            if len(sub_progs) == 0:
                tokens += sp
            else:
                tokens += ['union'] + sp

        return tokens
                        

SYM_CHANCE = 0.35
def sample_prim(ex):
    prim = PrimGroup(ex)

    if random.random() < SYM_CHANCE:
        prim.sample_sym(ex)

    if len(prim.top_info) > 0:
        if random.random() < SYM_CHANCE:
            prim.sample_sym(ex)

    return prim

def sample_prim_in_context(ex, local):

    prim = PrimGroup(ex, local)

    skip_sym = False

    for t in local:
        if 'sym' in t:
            skip_sym = True
            break
        
    if not skip_sym and random.random() < SYM_CHANCE:
        prim.sample_sym(ex)

    return prim


def calc_overlap(AB, BB):

    A = AB.bboxes
    B = BB.bboxes

    EA = A.repeat(B.shape[0],1)
    EB = B.unsqueeze(1).repeat(1, A.shape[0],1).view(-1,4)

    cbboxes = torch.stack((EA, EB),dim=1)
    
    lhs = torch.relu(cbboxes[:,:,1].min(dim=1).values - cbboxes[:,:,0].max(dim=1).values)
    rhs = torch.relu(cbboxes[:,:,3].min(dim=1).values - cbboxes[:,:,2].max(dim=1).values)
    intersect = lhs * rhs

    areas = ((cbboxes[:,:,1] - cbboxes[:,:,0]) * (cbboxes[:,:,3] - cbboxes[:,:,2]))

    AA = areas[:,0]
    BA = areas[:,1]

    AAS = AA.sum() / B.shape[0]
    BAS = BA.sum() / A.shape[0]
    
    return (intersect / AA).max().item(), (intersect /BA).max().item(), AAS, BAS


def sample_move_scale_prim(ex):

    width_val = multi_norm_sample(
        ex,
        (
            (0.4, (1.2, 0.6)),
            (0.6, (0.25, 0.2))
        ),
        ex.G_MIN_SCALE,
        1.9,
        'swtype',
    )

    height_val = multi_norm_sample(
        ex,
        (
            (0.4, (1.2, 0.6)),
            (0.6, (0.25, 0.2))
        ),
        ex.G_MIN_SCALE,
        1.9,
        'shtype',
    )

    xpos_val = multi_norm_sample(
        ex,
        (
            (0.3, (0.0, 0.0)),
            (0.4, (0.4, 0.4)),
            (0.4, (-0.4, 0.4))
        ),
        -1. + (width_val * 0.5),
        1. - (width_val * 0.5),
        'mxtype'
    )

    ypos_val = multi_norm_sample(
        ex,
        (
            (0.3, (0.0, 0.0)),
            (0.4, (0.4, 0.4)),
            (0.4, (-0.4, 0.4))
        ),
        -1. + (height_val * 0.5),
        1. - (height_val * 0.5),
        'mytype'
    )

    prim = Prim(ex, width_val, height_val, xpos_val, ypos_val)

    return prim, prim.get_bbox()

class Prim:
    def __init__(self, ex, width, height, xpos, ypos):

        self.ex = ex
        self.scale_params = (width, height)
        self.move_params = (xpos, ypos)
        self.prim_params = random.choice(ex.D_PRIMS)
        
    def get_bbox(self):
        xpos, ypos = self.move_params
        width, height = self.scale_params
        return torch.tensor([
            xpos - (width*0.5),
            xpos + (width*0.5),
            ypos - (height*0.5),
            ypos + (height*0.5)
        ]).float()

    def get_tokens(self):
        tokens = []
        
        tokens += self.ex.make_move_tokens(self.move_params)
        
        tokens += self.ex.make_scale_tokens(self.scale_params)
                
        tokens += ['prim', self.prim_params]

        return tokens


def multi_norm_sample(ex, dists, mi, ma, op):
    v = None

    if mi == ma:
        return mi
    
    for _ in range(10):
        
        r = random.random()
        if len(dists) == 2:
            p1, (m1, s1) = dists[0]
            _, (m2, s2) = dists[1]

            if r < p1:
                mean = m1
                std = s1
            else:
                mean = m2
                std = s2

        elif len(dists) == 3:
            p1, (m1, s1) = dists[0]
            p2, (m2, s2) = dists[1]
            _, (m3, s3) = dists[2]

            if r < p1:
                mean = m1
                std = s1
            elif r < p1 + p2:
                mean = m2
                std = s2
            else:
                mean = m3
                std = s3
                
        else:
            assert False    
                
                
        v = mean + (np.random.randn() * std)
        if v >= mi and v <= ma:
            val = ex.find_closest_float_val(v, op)
            if val >= mi and v <= ma:
                return val

    return ex.find_closest_float_val(mi, op)

## GROUP SAMPLING HELPER FUNCTIONS

def group_sample_sub_prog(ex, scope, counts):

    expr = sample_prim_in_context(ex, scope['local']).get_prog()    

    struct = []    
    tokens = []
    params = {}

    for t in expr:
        pt = ex.TLang.get_out_type(t)
        if pt in ex.DEF_PARAM_TYPES:
            nm = f'{ex.HOLE_PARAM_TOKEN}_{pt}_{counts["hpcnt"]}'
            counts["hpcnt"] += 1
            params[nm] = t
            tokens.append(nm)
        else:
            struct.append(t)
            tokens.append(t)
            
    return tokens, params, struct

def parse_local_scope(ex, tokens):
    ct = deepcopy(tokens)

    top_scope = []

    return _parse_local_scope(ex, ct, top_scope)

def _parse_local_scope(ex, tokens, scope):
    t = tokens.pop(0)
    scope.append(t)
    last_scope = scope
    for inp in ex.TLang.get_inp_types(t):

        if len(tokens) == 0:
            return last_scope
        
        if inp in ex.DEF_STRUCT_TYPES:
            nscope = deepcopy(scope)
            last_scope = _parse_local_scope(ex, tokens, nscope)
        else:
            scope.append(tokens.pop(0))

    return last_scope
