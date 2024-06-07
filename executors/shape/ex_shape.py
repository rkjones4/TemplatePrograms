import utils
from utils import device
import time
import numpy as np
import random
from copy import deepcopy
import sys
import torch

import matplotlib.pyplot as plt
import math
from tqdm import tqdm

sys.path.append('executors/shape')
import ShapeAssembly
import group_sampler as gs
import shape_utils as shu

sys.path.append('executors')
sys.path.append('executors/common')
sys.path.append('..')
sys.path.append('../common')

import tutils as tu
import eutils as eu
import lutils as lu
import base

VERBOSE = False
MAX_PRIMS = 20

SHAPE_CONFIG = {
    'DEF_STRUCT_TYPES': ['hole', 'shape'],
    
    'MAX_SEM_PARTS': 16,
    'MAX_PRIMS': MAX_PRIMS,
    'MAX_PARTS': MAX_PRIMS,

    'TYPED_CAT_REL_PROBS': None,
    'GROUP_FLOAT_STD_RNG': None,
    'SKIP_HOLE_TOKENS': None,

    'VIN_TYPE': 'prim',
    'VOXEL_DIM': None
}

DIM = None
pts = None

def set_voxel_pts(vdim):
    global DIM
    global pts
    print(f'Setting voxel dim to {vdim}')
    DIM = vdim
    a = (torch.arange(DIM).float() / (DIM-1.)) - .5
    b = a.unsqueeze(0).unsqueeze(0).repeat(DIM, DIM, 1)
    c = a.unsqueeze(0).unsqueeze(2).repeat(DIM, 1, DIM)
    d = a.unsqueeze(1).unsqueeze(2).repeat(1, DIM, DIM)
    pts = torch.stack((b,c,d), dim=3).view(-1, 3).to(device)

def tensorize_prim(prim):
    tp = torch.cat((
        prim.dims,
        prim.pos
    ))
    return tp

            
def rejection_check(state):

    assert len(state.shape) == 2
    
    if state.abs().sum(dim=1).min().item() < 0.01:
        mi = (state.abs().sum(dim=1) < 0.01).nonzero().flatten()[0]
        if mi == 0:
            return True
        assert state[mi:].abs().sum() < 0.01
        state = state[:mi]
            
    if state.shape[0] > MAX_PRIMS:
        return True

    if state[:,:3].min() < 0.001:
        return True
    
    return False    
        
def make_att_params(face,u,v,flip):
    
    att = [None,None,None,None,None,None]

    if face == 'left':
        I = 3
        J = 0
        A = 1
        B = 2
        C = 4
        D = 5
        
    elif face == 'right':
        I = 0
        J = 3
        A = 1
        B = 2
        C = 4
        D = 5

    elif face == 'bot':
        I = 4
        J = 1
        A = 0
        B = 2
        C = 3
        D = 5

    elif face == 'top':
        I = 1
        J = 4
        A = 0
        B = 2
        C = 3
        D = 5

    elif face == 'back':
        I = 5
        J = 2
        A = 0
        B = 1
        C = 3
        D = 4

    elif face == 'front':
        I = 2
        J = 5
        A = 0
        B = 1
        C = 3
        D = 4

    if flip and face == 'bot':
        att[I] = 0.
        att[J] = 0.
    elif flip and face == 'top':
        att[I] = 1.
        att[J] = 1.
    else:
        att[I] = 1.0
        att[J] = 0.0
        
    att[A] = 0.5
    att[B] = 0.5
    att[C] = u
    att[D] = v
        
    return [str(a) for a in att]

class Program:
    def __init__(self, ex):
        self.ex = ex

    def get_state_sig(self):
        sig = [tuple(tensorize_prim(s).tolist()) for s in self.state]
        sig.sort()
        return tuple(sig)
    
    def reset(self):
        self.lines = []                
        self.bbox_dims = None
        self.hier_progs = []

        self.sub_prog_reset()

    def sub_prog_reset(self):
        self.cur_name = None        
        self.cind_to_names = ['bbox']
        self.cur_prog_count = 0
        
    def _execute(self, fn, params):

        if '!' in fn and 'fill' in fn:

            part_ind = fn.split('!')[1]
            assert len(params) == 1
            bw, bh, bd = self.bbox_dims            
            self.lines += [f'cube0#{part_ind}# = Cuboid({bw}, {bh}, {bd})']
            assert params[0] == 'end'
            self.execute(params[0])
            return                
            
        elif '!' in fn:
            assert 'Cuboid' in fn

            part_ind = fn.split('!')[1]
            assert len(params) == 5 and params[0] == 'leaf'

            eparams = [self.execute(p) for p in params[1:4]]

            cw,ch,cd = eparams

            bw, bh, bd = self.bbox_dims

            name = f'cube{self.cur_prog_count}#{part_ind}#'
            self.cind_to_names.append(name)            
            self.cur_prog_count += 1

            self.lines += [
                f'{name} = Cuboid({cw * bw}, {ch * bh}, {cd * bd})'
            ]

            self.cur_name = name
            self.execute(params[4])
            return
            
            
        elif fn == 'end':
            self.lines += ['}']
            return

        # RootProg        
        elif fn == 'RootProg':
            assert len(params) == 5

            eparams = [self.execute(p) for p in params[:3]]
            
            bw,bh,bd = eparams
            
            self.lines += [
                'Assembly Program_0 {',
                f'bbox = Cuboid({bw},{bh},{bd})'
            ]
            self.bbox_dims = (bw, bh, bd)
            
            self.execute(params[3])
            self.execute(params[4])
            return
            
        # SubProg
        elif fn == 'SubProg':
            assert len(params) == 2
            self.sub_prog_reset()
            
            sp_name, bbw, bbh, bbd = self.hier_progs.pop(0)
            
            self.bbox_dims = (bbw, bbh, bbd)

            self.lines += [
                f'Assembly {sp_name}' + ' {',
                f'bbox = Cuboid({bbw},{bbh},{bbd})'
            ]

            self.execute(params[0])
            self.execute(params[1])
            return

        elif fn == 'empty':
            assert len(params) == 1
            assert params[0] == 'end'
            self.execute(params[0])
            return                

        elif fn == 'fill':
            assert len(params) == 1
            bw, bh, bd = self.bbox_dims            
            self.lines += [f'cube0 = Cuboid({bw}, {bh}, {bd})']
            assert params[0] == 'end'
            self.execute(params[0])
            return                
            
        # Cuboid
        elif fn == 'Cuboid':
            assert len(params) == 5

            ctype = params[0]
            assert ctype in ('leaf','hier','HOLE')

            eparams = [self.execute(p) for p in params[1:4]]
            
            cw,ch,cd = eparams

            bw, bh, bd = self.bbox_dims
            
            if ctype == 'leaf':
                name = f'cube{self.cur_prog_count}'
            else:
                hpcount = len(self.hier_progs) + 1
                name = f'Program_{hpcount}'                
                
                self.hier_progs.append((name, cw * bw, ch * bh, cd * bd))

            self.cind_to_names.append(name)            
            self.cur_prog_count += 1

            self.lines += [
                f'{name} = Cuboid({cw * bw}, {ch * bh}, {cd * bd})'
            ]

            self.cur_name = name
            self.execute(params[4])
            return
            
        # Attach
        elif fn == 'Attach':
            assert len(params) == 5
            
            c1, face = params[:2]

            eparams = [self.execute(p) for p in params[2:4]]

            u,v = eparams
            
            assert face in self.ex.D_FACES

            assert c1 in self.ex.D_CIND
            
            n1 = self.cind_to_names[int(c1.split('_')[1])]

            att_params = make_att_params(face, u, v, n1 == 'bbox')
            
            self.lines += [
                f'Attach({self.cur_name},{n1},{",".join(att_params)})'
            ]

            self.execute(params[4])
            return
            
        # Squeeze
        elif fn == 'Squeeze':
            assert len(params) == 6

            c1, c2, face = params[:3]

            eparams = [self.execute(p) for p in params[3:5]]

            u,v = eparams                       
            
            assert face in self.ex.D_FACES
            assert c1 in self.ex.D_CIND
            assert c2 in self.ex.D_CIND
            
            n1 = self.cind_to_names[int(c1.split('_')[1])]
            n2 = self.cind_to_names[int(c2.split('_')[1])]

            self.lines += [
                f'Squeeze({self.cur_name}, {n1}, {n2}, {face}, {u}, {v})'
            ]
            self.execute(params[5])
            return
            
        # Reflect
        elif fn == 'Reflect':
            assert len(params) == 2
            axis = params[0]
            assert axis in self.ex.D_AXIS
            self.lines += [
                f'Reflect({self.cur_name}, {axis})'
            ]
            self.execute(params[1])
            return
            
        # Translate
        elif fn == 'Translate':
            assert len(params) == 4

            axis = params[0]
            dist = self.execute(params[1])
            num = params[2]
            
            assert axis in self.ex.D_AXIS
            assert num in self.ex.D_CNUM

            n = int(num.split('_')[1])
            
            self.lines += [
                f'Translate({self.cur_name}, {axis}, {n}, {dist})'
            ]
            self.execute(params[3])
            return
            
        elif fn in self.ex.FLOAT_MAP:
            return self.ex.FLOAT_MAP[fn]

        elif fn in self.ex.TLang.tokens:
            return fn
        
        else:
            try:
                return float(fn)            
            except:
                pass    
            
            assert False, f'bad function {fn}'
            
    def execute(self, expr):

        if not isinstance(expr, list):            
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
            
        assert len(params) == ipc, 'bad expr'        
        assert pc == 0, 'bad expr'

        return self._execute(fn, params)

    def run(self, expr):
        self.reset()
        self._expr = expr
        
        if expr[0] == self.ex.START_TOKEN:
            expr = expr[1:]

        self.execute(expr)

        assert self.lines[-1] == '}' and self.lines[-2] == '}', 'not enough ends'
        self.lines.pop(-1)
        
        self.state = ShapeAssembly.run_sa_prog(self.lines)

    def make_image(self):        
        return tensorize(MAX_PRIMS, self.state)
        
def draw_box(ax, cube, color = 'black'):

    if 'color' in cube.__dict__.keys():
        color = cube.color
    
    center = cube.pos.numpy()
    lengths = cube.dims.numpy()

    dir_1 = cube.rfnorm.numpy()
    dir_2 = cube.tfnorm.numpy()
    dir_3 = cube.ffnorm.numpy()
    
    rot = np.matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    center = (rot * center.reshape(-1, 1)).reshape(-1)
    dir_1 = (rot * dir_1.reshape(-1, 1)).reshape(-1)
    dir_2 = (rot * dir_2.reshape(-1, 1)).reshape(-1)
    dir_3 = (rot * dir_3.reshape(-1, 1)).reshape(-1)
    
    dir_1 = dir_1/np.linalg.norm(dir_1)
    dir_2 = dir_2/np.linalg.norm(dir_2)
    dir_3 = dir_3/np.linalg.norm(dir_3)
    
    cornerpoints = np.zeros([8, 3])

    d1 = 0.5*lengths[0]*dir_1
    d2 = 0.5*lengths[1]*dir_2
    d3 = 0.5*lengths[2]*dir_3

    cornerpoints[0][:] = center - d1 - d2 - d3
    cornerpoints[1][:] = center - d1 + d2 - d3
    cornerpoints[2][:] = center + d1 - d2 - d3
    cornerpoints[3][:] = center + d1 + d2 - d3
    cornerpoints[4][:] = center - d1 - d2 + d3
    cornerpoints[5][:] = center - d1 + d2 + d3
    cornerpoints[6][:] = center + d1 - d2 + d3
    cornerpoints[7][:] = center + d1 + d2 + d3
    
    ax.plot([cornerpoints[0][0], cornerpoints[1][0]], [cornerpoints[0][1], cornerpoints[1][1]],
            [cornerpoints[0][2], cornerpoints[1][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[2][0]], [cornerpoints[0][1], cornerpoints[2][1]],
            [cornerpoints[0][2], cornerpoints[2][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[3][0]], [cornerpoints[1][1], cornerpoints[3][1]],
            [cornerpoints[1][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[3][0]], [cornerpoints[2][1], cornerpoints[3][1]],
            [cornerpoints[2][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[5][0]], [cornerpoints[4][1], cornerpoints[5][1]],
            [cornerpoints[4][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[6][0]], [cornerpoints[4][1], cornerpoints[6][1]],
            [cornerpoints[4][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[5][0], cornerpoints[7][0]], [cornerpoints[5][1], cornerpoints[7][1]],
            [cornerpoints[5][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[6][0], cornerpoints[7][0]], [cornerpoints[6][1], cornerpoints[7][1]],
            [cornerpoints[6][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[4][0]], [cornerpoints[0][1], cornerpoints[4][1]],
            [cornerpoints[0][2], cornerpoints[4][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[5][0]], [cornerpoints[1][1], cornerpoints[5][1]],
             [cornerpoints[1][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[6][0]], [cornerpoints[2][1], cornerpoints[6][1]],
            [cornerpoints[2][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[3][0], cornerpoints[7][0]], [cornerpoints[3][1], cornerpoints[7][1]],
            [cornerpoints[3][2], cornerpoints[7][2]], c=color)
        
def make_image(cubes, name):
    fig = plt.figure()
    
    extent = 0.5

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_proj_type('persp')
    ax.set_box_aspect(aspect = (1,1,1))
    
    ax.set_xlim(-extent, extent)
    ax.set_ylim(extent, -extent)
    ax.set_zlim(-extent, extent)

    for cube in cubes:
        draw_box(ax, cube)

    plt.tight_layout()

    if name is None:
        plt.show()
        plt.close('all')
        plt.clf()
    else:
        plt.savefig(f'{name}.png')        
        plt.close('all')
        plt.clf()

def get_color(i):
    ri = i % len(colors)
    num_over = (i // len(colors))
    over = ((num_over + 1) // 2) * 55
    
    sign = 2 * (((num_over+1) % 2 == 0) - .5)    
    delta = over * sign
    raw_color = colors[ri]    
    return tuple([
        min(max(c+delta,0),255)/255.
        for c in raw_color
    ])

colors = [
    (31, 119, 180),
    (174, 199, 232),
    (255,127,14),
    (255, 187, 120),
    (44,160,44),
    (152,223,138),
    (214,39,40),
    (255,152,150),
    (148, 103, 189),
    (192,176,213),
    (140,86,75),
    (196,156,148),
    (227,119,194),
    (247,182,210),
    (127,127,127),
    (199,199,199),
    (188,188,34),
    (219,219,141),
    (23,190,207),
    (158,218,229)
]
        
def tensor_to_state(tnsr, labels=None):

    state = []

    for i,T in enumerate(tnsr):
        assert T.shape[0] == 6 and len(T.shape) == 1

        if T.abs().sum() < 0.01:
            continue

        dims = T[:3]
        pos = T[3:]

        c = ShapeAssembly.Cuboid('dummy')
        c.dims = dims
        c.pos = pos

        if labels is not None:
            label = labels[i]
            color = get_color(label)
            c.color = color

        state.append(c)

    return state

        
def make_multi_image(domain, scenes, rows, name):

    
    if rows == 1:
        fig = plt.figure(figsize=(16,2))
    else:
        fig = plt.figure(figsize=(16,8))    

    extent = 0.5
    for i, scene in enumerate(scenes):            
        ax = fig.add_subplot(rows, math.ceil(len(scenes)/rows), i+1, projection='3d')
        ax.axis('off')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_proj_type('persp')
        ax.set_box_aspect(aspect = (1,1,1))
        
        ax.set_xlim(-extent, extent)
        ax.set_ylim(extent, -extent)
        ax.set_zlim(-extent, extent)
        
        domain.vis_on_axes(ax, scene)

    plt.tight_layout()

    if name is None:
        plt.show()        
    else:
        plt.savefig(f'{name}.png')
        
    plt.close('all')
    plt.clf()

def tensorize(num_prims, state):
    T = torch.zeros(
        num_prims,
        6
    )
    for i,s in enumerate(state):
        T[i] = tensorize_prim(s)
            
    return T
    
class ShapeExecutor(base.BaseExecutor):
    def __init__(self, config = None):
        self.ex_name = 'shape'
        if config is not None:
            SHAPE_CONFIG.update(config)

        self.prog_cls = Program
        self.base_init(SHAPE_CONFIG)
        self.make_lang()

        if self.VIN_TYPE == 'voxel':
            assert self.VOXEL_DIM is not None
            set_voxel_pts(self.VOXEL_DIM)
            
    def tensor_to_state(self, A, B=None):
        return tensor_to_state(A, B)
        
    def reformat_struct(self, struct):
        return shu.reformat_struct(self, struct)
        
    def remove_deriv(self, expr, struct):
        return shu.remove_deriv(self, expr, struct)
        
    def find_deriv(self, expr, struct):
        return shu.find_deriv(self, expr, struct)
                   
    def make_plot_render(self, num, gs, fs):
        extent = 0.5
        fig = plt.figure(figsize=fs)
        c = 0
        axes = {}
        for i in range(num):
            for j in range(gs):
                c += 1
                ax = fig.add_subplot(num, gs, c, projection='3d')
                ax.axis('off')
                ax.set_xlabel('x')
                ax.set_ylabel('z')
                ax.set_zlabel('y')
                ax.set_proj_type('persp')
                ax.set_box_aspect(aspect = (1,1,1))
                
                ax.set_xlim(-extent, extent)
                ax.set_ylim(extent, -extent)
                ax.set_zlim(-extent, extent)

                axes[(i,j)] = ax
                                
        return fig, axes

    def vis_voxel_on_axes(self, ax, scene):
        ax.set_xlim(0, self.VOXEL_DIM)
        ax.set_ylim(0, self.VOXEL_DIM)
        ax.set_zlim(0, self.VOXEL_DIM)

        if isinstance(scene, np.ndarray):
            scene = torch.from_numpy(scene)
        
        ax.voxels(
            filled=(scene >= 0.5).permute(2,0,1).flip(dims=[1]),
            shade=True
        )
        
    def vis_on_axes(self, ax, scene):

        if self.VIN_TYPE == 'voxel' \
           and len(scene.shape) == 3 and scene.shape[0] == self.VOXEL_DIM:
            return self.vis_voxel_on_axes(ax, scene)

        if isinstance(scene, np.ndarray):
            scene = torch.from_numpy(scene)
            
        state = tensor_to_state(scene)
        
        for cube in state:
            draw_box(ax, cube)

    def draw_box(self, a, b, c=None):
        return draw_box(a, b, c)
                 
    def get_input_shape(self):
        if self.VIN_TYPE == 'prim':
            return [self.MAX_PRIMS, 6]
        elif self.VIN_TYPE == 'voxel':
            return [self.VOXEL_DIM, self.VOXEL_DIM, self.VOXEL_DIM]
        else:
            assert False, f'bad in type {self.VIN_TYPE}'

    def get_group_sample_params(self):
        HOLE_OPTS = [0,1,2,3,4]
        HOLE_PROB = lu.norm_np([1,.5,.25,.125,.0625])
        num_holes = np.random.choice(HOLE_OPTS, p = HOLE_PROB) 
        return {
            'num_holes': num_holes,
        }
    
    def get_det_sample_params(self):
        return {
            'num_holes_to_add': 0.
        }

    def make_new_part_pred(self, expr):
        tokens = expr.split()
        assert tokens[0] == self.START_TOKEN
        P = Program(self)
        P.reset()
        P.execute(tokens[1:])
        assert P.lines[-1] == '}' and P.lines[-2] == '}', 'not enough ends'
        P.lines.pop(-1)

        P.state, part_pred = ShapeAssembly.run_sa_prog_ppred(P.lines)
        
        return torch.tensor(part_pred).long()

    def add_part_info(self, expr, struct):
        eapi = eu.shape_add_part_info(self, expr, struct)
        return eapi
        
    def ex_prog(self, tokens):
        P = Program(self)
        P.run(tokens)
        return P
            
    def check_valid_prog(self, P, ret_vdata=False):
                
        if len(P.state) > self.MAX_PRIMS:
            return None

        T = tensorize(self.MAX_PRIMS, P.state)

        if torch.isnan(T).any():
            return None
        
        if rejection_check(T):
            return None

        if not ret_vdata:
            return True

        return P.state
            
    def render_group(self, images, name=None, rows=1):
        make_multi_image(self, images,rows=rows,name=name)
        
    def execute(self, expr, vis=False, ret_state=False):

        tokens = expr.split()
        
        assert tokens[0] == self.START_TOKEN
    
        P = Program(self)
        P.run(tokens)
                    
        if vis:
            make_image(P.state, None)

        if ret_state:
            return P.state
            
        return tensorize(self.MAX_PRIMS, P.state)
                    
    def make_lang(self):
        self.add_token_info()
        self.set_tlang()

    def add_token_info(self):
        
        self.D_PFLT = [round(i/20.,2) for i in range(21)]
        self.D_CFLT = [round(i/40.,2) for i in range(41)]

        self.D_CFLT[0] = 0.01
        
        self.D_AXIS = ['AX', 'AY', 'AZ']
        self.D_CNUM = [f'cnum_{i}' for i in (1,2,3,4,5)]
        self.D_CIND = [f'cind_{i}' for i in (0,1,2,3,4)]
        self.D_FACES = ['left','right','bot','top','back','front']
        
        self.D_PARAM_LOCS = tuple(
            [f'{self.PARAM_LOC_TOKEN}_{i}' for i in range(self.MAX_PARAM_TOKENS)]
        )
        self.D_STRUCT_LOCS = tuple(
            [f'{self.STRUCT_LOC_TOKEN}_{i}' for i in range(self.MAX_HOLES)]
        )
                
        self.CAT_PARAM_TYPES = ['axis', 'cnum', 'cind','face']
        self.FLOAT_PARAM_TYPES = ['pflt', 'cflt']
        
        self.DEF_PARAM_TYPES = self.CAT_PARAM_TYPES + self.FLOAT_PARAM_TYPES
        
        self.PRT_FNS = ['Cuboid']

    def extra_tlang_logic(self):
        self.TLang.T2SIPC['SubProg'] = None
        
    def set_tlang(self):

        TLang = base.TokenLang(self)

        TLang.add_token(self.START_TOKEN, 'shape', 'prog')
        TLang.add_token(self.HOLE_TOKEN, '', 'shape')
        TLang.add_token('end', '', 'shape')
        
        TLang.add_token('leaf', '', 'shape')
        TLang.add_token('hier', '', 'shape')
        TLang.add_token('empty', 'shape', 'shape')
        TLang.add_token('fill', 'shape', 'shape')

        TLang.add_token('RootProg', 'cflt,cflt,cflt,shape,shape', 'shape')
        TLang.add_token('SubProg', 'shape,shape', 'shape'), 
                
        TLang.add_token('Cuboid', 'shape,cflt,cflt,cflt,shape', 'shape')
        TLang.add_token('Attach', 'cind,face,pflt,pflt,shape', 'shape')
        TLang.add_token('Reflect', 'axis,shape','shape')
        TLang.add_token('Translate', 'axis,pflt,cnum,shape', 'shape')

        for face in self.D_FACES:
            TLang.add_token(face, '', 'face')
                       
        for cind in self.D_CIND:
            TLang.add_token(cind, '', 'cind')

        for ax in self.D_AXIS:
            TLang.add_token(ax, '', 'axis')

        for cnum in self.D_CNUM:
            TLang.add_token(cnum, '', 'cnum')            
            
        self.TMAP = {}
        self.FLOAT_MAP = {}
        self.REV_FLOAT_MAP = {}
        
        for vals, name in [
            (self.D_CFLT, 'cflt'),
            (self.D_PFLT, 'pflt'),
        ]:
            
            self.TMAP[name] = (torch.tensor(vals), vals)
    
            for i,val in enumerate(vals):
                TLang.add_token(f'{name}_{i}', '', name)
                self.FLOAT_MAP[f'{name}_{i}'] = val
                self.REV_FLOAT_MAP[(name, val)] = f'{name}_{i}'

        for t in self.D_PARAM_LOCS:
            TLang.add_token(t, '', 'param_loc', 'inp_only')

        for t in self.D_STRUCT_LOCS:
            TLang.add_token(t, '', 'struct_loc', 'inp_only')

        self.TLang = TLang
        TLang.init()
        
        
        

    def _group_prog_random_sample(
        self, num_progs, num_derivs_per_prog, vis_progs, use_pbar,
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

            group_sampler = gs.sample_group(self, sample_params)

            if not group_sampler.gs_valid:
                continue
            
            tokens = group_sampler.tokens
            
            if tokens is None or len(tokens) >= max_tokens:
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
                if self.VERBOSE:
                    print(f"too many shared tokens {rcount}")
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
            
                deriv = group_sampler.sample_deriv()

                if not deriv.gs_valid:
                    continue
                
                if len(deriv.tokens) >= max_tokens:
                    continue

                if len(deriv.param_vals.keys()) >= max_param_tokens:
                    continue

                try:
                    deriv.ex_prog()
                except Exception as e:
                    if VERBOSE:
                        print(f"Failed deriv ex with {e}")
                        print(deriv.prog)
                    continue

                # Get the signature of all primitives
                sig = deriv.get_sig()

                if sig is not None and sig in seen:
                    continue

                seen.add(sig)
            
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

    def conv_scene_to_vinput(self, scene):
        if self.VIN_TYPE == 'prim':
            return scene
        
        elif self.VIN_TYPE == 'voxel':
            if len(scene.shape) == 2:
                nc = (scene.abs().sum(dim=1) > 0).sum().item()                
                voxels = make_voxels(scene[:nc,:], self.VOXEL_DIM)
                return voxels
            else:
                assert scene.shape[0] == self.VOXEL_DIM and scene.shape[2] == self.VOXEL_DIM
                return scene
        else:
            assert False, f'bad vin type {self.VIN_TYPE}'
    

# Visualize helper function
def vis_voxels(voxels, fn):
    with open(fn,'w') as f:
        for i in range(0, DIM):
            for j in range(0, DIM):
                for k in range(0, DIM):
                    x,y,z = (i-DIM//2.) / (DIM//2.), (j-DIM//2.) / (DIM//2.), (k-DIM//2.) / (DIM//2.)
                    if voxels[i,j,k]:
                        f.write(f'v {x} {y} {z}\n')
        
def make_voxels(cubes, vdim):
    with torch.no_grad():
        return _make_voxels(cubes, vdim)
                        
def _make_voxels(cubes, vdim):
        
    cubes = cubes.to(device)
    ucubes = cubes.unsqueeze(0)
        
    cent_pts = pts.unsqueeze(1) - ucubes[:,:,3:6]    

    cube_sdfs = (
        cent_pts.abs() - ( ucubes[:,:,:3] / 2.)
    ).max(dim=2).values

    vthresh = (1.0 / vdim) / 1.41
        
    exp_voxels = (cube_sdfs <= vthresh)
    
    flat_voxels = exp_voxels.float().sum(dim=1)

    num = ((flat_voxels == 1.).view(-1, 1) & exp_voxels).sum(dim=0)

    vox = flat_voxels > 0.
    
    return vox.view(DIM,DIM,DIM)

