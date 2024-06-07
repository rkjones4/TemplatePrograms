import matplotlib.pyplot as plt
import sys
import random
from copy import deepcopy
import numpy as np
import torch
import math
import os
sys.path.append('executors')
sys.path.append('executors/common')
sys.path.append('..')
sys.path.append('../common')
import base
import lutils as lu
from tqdm import tqdm

device=torch.device('cuda')

CHECK_OVERLAP = 30
EX_PTS = None

OMN_CONFIG = {
    'DEF_STRUCT_TYPES' : ['hole', 'shape'],
    'VDIM': 28,
    'NUM_SMP_PTS': 250,
    'MAX_SEM_PARTS': 16,    

    'TYPED_CAT_REL_PROBS': {
        'mind': lu.norm_np([5, 95,0]),
        'dtype': lu.norm_np([0, 90, 10]),
        'stype': lu.norm_np([0, 90, 10]),
        'btype': lu.norm_np([0, 90, 10]),
        'mtype': lu.norm_np([0, 90, 10]),
    },
    
    'GROUP_FLOAT_STD_RNG': {
        'dflt': (1, 9),
        'sflt': (0.01, 0.25),
        'mflt': (1, 4),
        'bflt': (0.01, 15),
    },

    'SKIP_HOLE_TOKENS': ('end', 'start', 'START', 'on', 'off', 'empty')
}

def convert_to_small_image(d):    
    
    SVDIM = 28
    
    pimg = torch.zeros(SVDIM, SVDIM).float()

    myink = 1.0
    
    for stroke in d.strokes:

        stk = ((stroke.traj + 1.) / 2.) * SVDIM
                        
        x = stk[:,1]
        y = stk[:,0]

        xfloor = torch.clamp(torch.round(x).long(), 0, SVDIM-1)
        yfloor = torch.clamp(torch.round(y).long(), 0, SVDIM-1)
                    
        pimg[xfloor, yfloor] = 1.0
            
    return pimg

    
def make_ftype_norm_sampler(ftype, vals, mu, std):

    def sample_float(A=None, B=None):
        T,_ = vals

        val = mu + (np.random.randn() * std)
        
        ind = (T - val).abs().argmin()

        return f'{ftype}_{ind}'

    return sample_float
        
# Start LANG Execution Logic

def center_strokes(strokes):
    all_traj = []

    for stroke in strokes:
        all_traj.append(stroke.traj)

    all_traj = torch.cat(all_traj,dim=0)
    mapos = all_traj.max(dim=0).values
    mipos = all_traj.min(dim=0).values

    offset = ((mapos + mipos) / 2.).view(1,2)
    
    for stroke in strokes:
        stroke.traj -= offset
    

class Stroke:
    def __init__(self, start_pos, sem_cls=None):
        self.traj = None
        self.start_pos = start_pos
        self.end_pos = None
        self.sem_cls = sem_cls

    def get_length(self):
        length =(self.traj[:-1] - self.traj[1:]).norm(dim=-1).sum().item()
        return length
        
    def inst_linear(self, delta):
        
        self.end_pos = self.start_pos.clone() + delta
        a = (torch.arange(100).float() / 100.).view(-1,1)
        self.traj = (self.start_pos * (1-a)) + (self.end_pos * a)         
        
def calculate_pixel_overlap(ex, strokes, new_ink_thresh = 0.5, old_ink_thresh=0.5):
    VDIM = 64

    pimg = torch.zeros(VDIM, VDIM).float()

    myink = 1.0

    total_pc = 0
    
    for traj in strokes:

        stk = ((traj + 1.) / 2.) * VDIM
                        
        x = stk[:,1]
        y = stk[:,0]
        
        xfloor = torch.clamp(torch.floor(x).long(), 0, VDIM-1)
        yfloor = torch.clamp(torch.floor(y).long(), 0, VDIM-1)            
        xceil = torch.clamp(torch.ceil(x).long(), 0, VDIM-1)
        yceil = torch.clamp(torch.ceil(y).long(), 0, VDIM-1)
                
        x_c_ratio = x - xfloor
        y_c_ratio = y - yfloor
        x_f_ratio = 1 - x_c_ratio
        y_f_ratio = 1 - y_c_ratio

        pc1 = ((pimg[xfloor, yfloor] > old_ink_thresh) & (x_f_ratio*y_f_ratio > new_ink_thresh)).float().sum().item()
        pc2 = ((pimg[xceil, yfloor] > old_ink_thresh) & (x_c_ratio*y_f_ratio > new_ink_thresh)).float().sum().item()
        pc3 = ((pimg[xfloor,yceil] > old_ink_thresh) & (x_f_ratio*y_c_ratio > new_ink_thresh)).float().sum().item()
        pc4 = ((pimg[xceil,yceil] > old_ink_thresh) & (x_c_ratio*y_c_ratio > new_ink_thresh)).float().sum().item()

        total_pc += pc1 + pc2 + pc3 + pc4
        
        pimg = pimg.index_put((xfloor, yfloor), myink*x_f_ratio*y_f_ratio, accumulate=True)
        pimg = pimg.index_put((xceil, yfloor), myink*x_c_ratio*y_f_ratio, accumulate=True)
        pimg = pimg.index_put((xfloor, yceil), myink*x_f_ratio*y_c_ratio, accumulate=True)
        pimg = pimg.index_put((xceil, yceil), myink*x_c_ratio*y_c_ratio, accumulate=True)

    return total_pc

def sample_traj(traj, npts):
    
    length = (traj[:-1] - traj[1:]).norm(dim=-1)

    area = length.sum()

    if area == 0.:
        return None
    
    dist = torch.distributions.categorical.Categorical(probs=length / area)
    seg_index = dist.sample((npts,))

    a = torch.rand(npts,1)

    pts = (traj[seg_index] * a) + (traj[1:][seg_index] * (1-a))

    return pts


class Program:
    def __init__(self, ex):
        
        self.ex = ex
        self.soft_error = False

        # (start pos, end pos, bow_deg, on/off)
        self.strokes = [] 
        self.cursor_pos = torch.tensor([0., 0.]).float()
        self.cursor_state = 'on'
        self.bow_params = None

    def get_state_sig(self):
        sig = tuple(torch.cat([s.traj for s in self.strokes]).flatten().tolist())
        
        return sig
        
    def has_soft_error(self):        
        if self.soft_error:
            return True

        if CHECK_OVERLAP is not None:
            pixel_overlap = calculate_pixel_overlap(self.ex, [s.traj for s in self.strokes])
            if pixel_overlap > CHECK_OVERLAP:
                return True
                    
        return False

    def get_stroke_data(self):
        return [s.traj for s in self.strokes]
    
    def point_sample(self, num_points):
        lengths = []
        total = 0

        for stroke in self.strokes:
            sl = stroke.get_length()
            lengths.append(sl)
            total += sl

        if total == 0.:
            return []
            
        split_num_points = [math.ceil(l*num_points/total) for l in lengths]

        points = []

        for stroke,np in zip(self.strokes, split_num_points):
            pts=sample_traj(stroke.traj, np)
            if pts is None:
                continue
            points.append(pts)

        points = torch.cat(points,dim=0).view(-1,2)

        return points[:num_points]
    
    def reset(self):
        self._expr = None
        self.strokes = []
        self.cursor_pos = torch.tensor([0., 0.]).float()
        self.cursor_state = 'on'
                
    def ex_move_cursor(self, mind, mtype, mflt):
        stroke_ind = int(mind.split('_')[1])

        if len(self.strokes) == 0:
            self.soft_error=True
            return
        
        if stroke_ind >= len(self.strokes):           
            self.soft_error=True
            stroke_ind = len(self.strokes) - 1

        stroke = self.strokes[stroke_ind]

        mtype = int(mtype.split('_')[1])

        if mflt in self.ex.FLOAT_MAP:
            mflt = self.ex.FLOAT_MAP[mflt]
        else:
            mflt = float(mflt)

        assert stroke.traj.shape[0] == 100, 'bad num stroke traj'
        
        tind = round(mtype +  mflt)
            
        self.cursor_pos = stroke.traj[tind].clone()
            
    def ex_bow_last_stroke(self, btype, bflt):
        if len(self.strokes) == 0:
            self.soft_error = True
            return
        
        lstroke = self.strokes.pop(-1)

        A = lstroke.start_pos
        B = lstroke.end_pos

        if bflt in self.ex.FLOAT_MAP:
            bow_deg = int(btype.split('_')[1]) + self.ex.FLOAT_MAP[bflt]
        else:
            bow_deg = int(btype.split('_')[1]) + float(bflt)

        assert bow_deg != 0.
        
        if bow_deg < 0.0:
            bow_traj = self.calc_bow_traj(B, A, -1 * bow_deg)
            bow_traj = torch.flip(bow_traj,dims=[0])
        else:
            bow_traj = self.calc_bow_traj(A, B, bow_deg)
                        
        lstroke.traj = bow_traj

        self.strokes.append(lstroke)
        
    def calc_bow_traj(self, A, B, bow_deg):
        
        midpoint = (A + B) / 2.

        dlta = B - A
        odir = torch.tensor([dlta[1].item(), dlta[0].item() * -1])
        odir /= odir.norm() + 1e-8

        L = dlta.norm().item()
        
        assert bow_deg > 0 and bow_deg <= 180, 'bad deg num'

        if bow_deg == 180:
            X = 0.0
        else:
            bow_rad = (bow_deg * math.pi) / 180.
            X = L / (2 * math.tan(bow_rad / 2.))
            
        C = midpoint + (odir * X)
        
        AP = A - C
        BP = B - C
        
        rad = AP.norm()        

        APD = math.atan2(AP[1].item(), AP[0].item())
        BPD = math.atan2(BP[1].item(), BP[0].item())
        
        if (APD + math.pi) < 1e-4:
            APD = math.pi

        if (BPD + math.pi) < 1e-4:
            BPD = math.pi

        if abs(abs(APD - BPD) - math.pi) < 1e-4:
            eq_height = abs(A[1] - B[1]) < 1e-4
            if (not eq_height and A[1] < B[1]) or (eq_height and B[0] < A[0]):
                if APD < BPD:
                    APD += 2*math.pi
                else:
                    BPD += 2*math.pi
                        
        elif abs(APD - BPD) > math.pi + 1e-4:
            if APD < 0.0:
                APD += 2*math.pi
            elif BPD < 0.0:
                BPD += 2*math.pi
            else:
                assert False
                                                                    
        rng = torch.arange(100).float() / 100.

        DEGS = (APD * (1-rng)) + (BPD * rng)

        DEGS = torch.remainder(DEGS, 2*math.pi)
        
        XCDS = torch.cos(DEGS) * rad
        YCDS = torch.sin(DEGS) * rad
        
        return torch.stack((XCDS, YCDS),dim=1) + C.view(1,2)
        
        
    def ex_draw(self, dtype, stype, dflt, sflt,sem_cls):
        if dflt in self.ex.FLOAT_MAP:
            degree = float(dtype.split('_')[1]) + self.ex.FLOAT_MAP[dflt]
        else:
            degree = float(dtype.split('_')[1]) + float(dflt)
            
        radians = (degree * math.pi) / 180.

        if sflt in self.ex.FLOAT_MAP:
            scale = float(stype.split('_')[1]) / 1000. + self.ex.FLOAT_MAP[sflt]
        else:
            scale = float(stype.split('_')[1]) / 1000. + float(sflt)

        scale = max(scale, 0.0) + 1e-8
            
        delta = torch.tensor([scale * math.cos(radians), scale * math.sin(radians)]).float()

        if self.cursor_state == 'on':
            stroke = Stroke(self.cursor_pos, sem_cls)
            stroke.inst_linear(delta)
            self.strokes.append(stroke)
            
        self.cursor_pos = self.cursor_pos + delta
    
    def _execute(self, fn, params):

        if '!' in fn:
            assert 'draw' in fn            
            sem_cls = int(fn.split('!')[1])
            fn = 'draw'
        else:
            sem_cls = None
            
        if fn == 'end':
            return
            
        elif fn == 'move':
            assert len(params) == 3
            self.ex_move_cursor(params[0], params[1], params[2])            
            return

        elif fn == 'empty':
            assert len(params) == 0
            return
        
        elif fn == 'on':
            assert len(params) == 2
            self.cursor_state = 'on'
            self.execute(params[0])
            self.execute(params[1])
            return

        elif fn == 'off':
            assert len(params) == 2
            self.cursor_state = 'off'
            self.execute(params[0])
            self.execute(params[1])
            return

        elif fn == 'bow':
            assert len(params) == 3

            if self.cursor_state == 'off':
                self.soft_error = True
                
            elif self.cursor_state == 'on':
                self.bow_params = (params[0], params[1])
                
            self.execute(params[2])
            
            return

        elif fn == 'scale':
            assert len(params) == 2
            return params[0], params[1]
        
        elif fn == 'draw':
            assert len(params) == 3
                
            assert 'scale' in params[2]

            prm1,prm2 = params[:2]

            prm3,prm4 = self.execute(params[2])

            self.ex_draw(prm1,prm3,prm2,prm4,sem_cls)

            if self.bow_params is not None:
                self.ex_bow_last_stroke(self.bow_params[0], self.bow_params[1])
                self.bow_params = None
                            
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
            
        assert len(params) == ipc
        assert pc == 0

        return self._execute(fn, params)

    def make_image(self, break_on_error=False):

        for stroke in self.strokes:

            stk = ((stroke.traj + 1.) / 2.) * self.ex.VDIM
            
            if break_on_error and ((stk < 0.).any() or (stk > self.ex.VDIM).any()):
                return None

        return convert_to_small_image(self)            

    def make_new_part_pred(self):

        query_points = []

        for xm,ym in [(0., 0.), (1.,1.),(1.,-1.),(-1.,1.),(-1.,-1.)]:
            OEX_PTS = EX_PTS + torch.tensor(
                [[.035 * xm, .035 * ym]],device=EX_PTS.device).float()
            query_points.append(OEX_PTS)

        query_points = torch.stack(query_points,dim=0)

        traj_points = []
        traj_sem_cls = []
        
        for stroke in self.strokes:            
            stp = sample_traj(stroke.traj, 200)
            if stp is None:
                continue
            
            stp = stp.to(EX_PTS.device)
            traj_points.append(stp)
            traj_sem_cls.append(stroke.sem_cls)
            
        traj_points = torch.stack(traj_points, dim=0)

        D = query_points.view(query_points.shape[0],query_points.shape[1],1,1,2) -\
            traj_points.view(1,1,traj_points.shape[0],traj_points.shape[1],2)
        D = D.norm(dim=-1)
        
        canvas = torch.zeros(
            self.ex.VDIM * self.ex.VDIM, self.ex.MAX_SEM_PARTS, device=EX_PTS.device
        ).float()

        traj_sem_cls = torch.tensor(traj_sem_cls,device=EX_PTS.device).long()

        darange = torch.arange(self.ex.VDIM * self.ex.VDIM, device=EX_PTS.device).long()

        for i in range(5):
            d = D[i].view(D.shape[1],-1)

            cinds = torch.topk(-d,3).indices            
            finds = torch.div(cinds, 200, rounding_mode='floor')
            linds = traj_sem_cls[finds]

            canvas[darange, linds[:,0]] += 1.
            canvas[darange, linds[:,1]] += 1.
            canvas[darange, linds[:,2]] += 1.

        return canvas / canvas.sum(dim=-1).unsqueeze(-1)
                        
                            
    def make_sem_img(self, sem, vis=False):
        img = torch.zeros(self.ex.VDIM * self.ex.VDIM, 3)

        sem_exp = sem.view(-1, sem.shape[-1])

        seg = sem_exp.argmax(dim=1)
        occ = (sem_exp.sum(dim=1) > 0.0).nonzero().flatten()

        for uv in seg[occ].unique():
            img[((seg == uv) & (sem_exp.sum(dim=1) >0.0)).nonzero().flatten(),:] = lu.CMAP[uv.item()]

        img = img.view(self.ex.VDIM, self.ex.VDIM, 3)

        if vis:
            plt.imshow(img.cpu().numpy(), origin='lower',vmin=0.0,vmax=1.0)
            plt.show()
        
        return img

    
    def make_sem_seg(self):
        pimg = torch.zeros(self.ex.VDIM, self.ex.VDIM, self.ex.MAX_SEM_PARTS).float()

        myink = 1.0
        SVDIM = self.ex.VDIM
        
        for sind, stroke in enumerate(self.strokes):

            sind = min(sind, self.ex.MAX_SEM_PARTS-1)
            
            stk = ((stroke.traj + 1.) / 2.) * self.ex.VDIM
            
            x = stk[:,1]
            y = stk[:,0]

            xfloor = torch.clamp(torch.round(x).long(), 0, SVDIM-1)
            yfloor = torch.clamp(torch.round(y).long(), 0, SVDIM-1)
            
            sinds = torch.ones(xfloor.shape).long() * sind

            pimg[xfloor,yfloor,:] = 0.0
            pimg[xfloor,yfloor,sinds] = 1.0 

        return pimg
        
        
    def render(self, name=None, num_points=None):

        with torch.no_grad():
            
            # 64 x 64 x 3 image
            for stroke in self.strokes:
                stk = stroke.traj
                plt.plot(stk[:,0], stk[:,1])
                
            plt.xlim(-1,1)
            plt.ylim(-1,1)

            if num_points:
                points = self.point_sample(num_points)                
                plt.scatter(points[:,0], points[:,1], c='r')
                
            if name is not None:
                plt.savefig(f'{name}.png')
            else:
                plt.show()
                            
    def run(self, expr):
        self.reset()
        self._expr = expr

        if expr[0] == self.ex.START_TOKEN:
            expr = expr[1:]
            
        self.execute(expr)        
        center_strokes(self.strokes)

        
class OmnExecutor(base.BaseExecutor):
    def __init__(self, config = None):
        self.ex_name = 'omni'
        if config is not None:
            OMN_CONFIG.update(config)

        self.prog_cls = Program
        self.base_init(OMN_CONFIG)
        self.make_lang()
        self.init_pts()
        self.CDA_TOKENS = ('move', 'on', 'off', 'draw', 'bow')
        
    def vis_on_axes(self, ax, img):
        if len(img.shape) == 3:
            img = img[:,:,0]
        ax.imshow(img, origin='lower', cmap='gray', vmin=0.0, vmax=1.0)
        
    def execute(self, expr, vis=False, ret_pt_smps=False, ret_strokes=False):

        tokens = expr.split()
        
        assert tokens[0] == self.START_TOKEN
    
        P = Program(self)
        P.run(tokens)
    
        with torch.no_grad():
            img = P.make_image()
                
        if vis:
            plt.imshow(
                img.cpu().numpy(), origin='lower',
                cmap='gray', vmin=0.0, vmax=1.0
            )
            plt.show()

        elif ret_strokes:
            return img, P.get_stroke_data()
            
        elif ret_pt_smps:
            pt_samps = P.point_sample(self.NUM_SMP_PTS)
            return img, pt_samps.detach().cpu()
            
        else:
            return img

    def make_sem_seg(self, expr):
        tokens = expr.split()
        assert tokens[0] == self.START_TOKEN

        P = Program(self)
        P.run(tokens)

        return P.make_sem_seg()
        
    def make_new_part_pred(self, expr):
        tokens = expr.split()
        
        assert tokens[0] == self.START_TOKEN

        P = Program(self)
        P.run(tokens)
        with torch.no_grad():
            return P.make_new_part_pred()

    def ex_prog(self, tokens):
        P = Program(self)
        P.run(tokens)
        return P
        
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
        self.add_sample_dist()

    def add_sample_dist(self):
        
        self.D_DTYPE_PROBS = [3,2,3,2,3,2,3,2]
        self.D_DFLT_PROBS = [1.,1.,3.,1.,1.]

        self.D_STYPE_PROBS = [0.5,1.25,1.75,2.5,2.0,1.5,1.0,.75]
        self.D_SFLT_PROBS = [1.,1.,1.,1.,1.]
        
        self.smp_dtype_opts = [int(i.split('_')[1]) for i in self.D_DTYPES]
        self.smp_dflt_opts = self.D_DFLT

        self.smp_stype_opts = [float(i.split('_')[1]) / 1000. for i in self.D_STYPES]
        self.smp_sflt_opts = self.D_SFLT

        self.smp_dtype_prob = lu.norm_np(self.D_DTYPE_PROBS)
        self.smp_dflt_prob = lu.norm_np(self.D_DFLT_PROBS)
        self.smp_stype_prob = lu.norm_np(self.D_STYPE_PROBS)
        self.smp_sflt_prob = lu.norm_np(self.D_SFLT_PROBS)

        assert len(self.smp_dtype_prob) == len(self.smp_dtype_opts)
        assert len(self.smp_dflt_prob) == len(self.smp_dflt_opts)
        assert len(self.smp_stype_prob) == len(self.smp_stype_opts)
        assert len(self.smp_sflt_prob) == len(self.smp_sflt_opts)

        self.D_BTYPE_PROBS = [.3,.1,.1,.3]
        self.D_BFLT_PROBS = [1,1,1]

        self.D_MTYPE_PROBS = [3,1,3,1,3]
        self.D_MFLT_PROBS = [1,1,1]

        self.smp_btype_opts = self.D_BTYPES
        self.smp_bflt_opts = self.D_BFLT

        self.smp_mtype_opts = self.D_MTYPES
        self.smp_mflt_opts = self.D_MFLT

        self.smp_btype_prob = lu.norm_np(self.D_BTYPE_PROBS)
        self.smp_bflt_prob = lu.norm_np(self.D_BFLT_PROBS)
        self.smp_mtype_prob = lu.norm_np(self.D_MTYPE_PROBS)
        self.smp_mflt_prob = lu.norm_np(self.D_MFLT_PROBS)
        
        assert len(self.smp_btype_prob) == len(self.smp_btype_opts)
        assert len(self.smp_bflt_prob) == len(self.smp_bflt_opts)
        assert len(self.smp_mtype_prob) == len(self.smp_mtype_opts)
        assert len(self.smp_mflt_prob) == len(self.smp_mflt_opts)
        
    def set_tlang(self):
        TLang = base.TokenLang(self)

        TLang.add_token(self.START_TOKEN, 'shape', 'prog')
        TLang.add_token(self.HOLE_TOKEN, '', 'hole', 'out_only')

        TLang.add_token('end', '', 'shape')
        TLang.add_token('empty', '', 'shape')
        
        TLang.add_token('move', 'mind,mtype,mflt', 'shape')
        TLang.add_token('on', 'shape,shape', 'shape')
        TLang.add_token('off', 'shape,shape', 'shape')
                
        TLang.add_token('draw', 'dtype,dflt,shape', 'shape')
        TLang.add_token('scale', 'stype,sflt', 'shape')
        TLang.add_token('bow', 'btype,bflt,shape', 'shape')
                
        for di in self.D_MINDS:
            TLang.add_token(di, '', 'mind')
            
        self.TMAP = {}
        self.FLOAT_MAP = {}
        self.REV_FLOAT_MAP = {}
        
        for vals, typ in [
            (self.D_DTYPES, 'dtype'),
            (self.D_STYPES, 'stype'),
            (self.D_BTYPES, 'btype'),
            (self.D_MTYPES, 'mtype'),
        ]:
            for v in vals:
                TLang.add_token(v, '', typ)
        
        for vals, name in [
            (self.D_DFLT, 'dflt'),
            (self.D_SFLT, 'sflt'),
            (self.D_MFLT, 'mflt'),
            (self.D_BFLT, 'bflt')
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
            
        TLang.init()
        
        self.TLang = TLang

    def make_group_float_sampler(
        self, val, pt, pfn, pp
    ):
        
        lr, up = self.GROUP_FLOAT_STD_RNG[pt]
        a = random.random()
        stddev = (lr * a) + (up * (1-a))
        
        samp_fn = make_ftype_norm_sampler(pt, self.TMAP[pt], self.FLOAT_MAP[val], stddev)
        
        return samp_fn
        
    def add_token_info(self):
        
        self.D_DFLT = [-18,-9,0.,9,18]
        self.D_SFLT = [-.05,-.025,0.,.025,0.05]
        self.D_MFLT = [-8, 0, 8]
        self.D_BFLT = [-30.,  0., 30.]
        
        self.D_MINDS = [f'mind_{i}' for i in range(12)]

        self.D_DTYPES = [f'dtype_{i}' for i in
                         ('0','45','90','135','180','225','270','315')]
                
        self.D_STYPES = [f'stype_{i}' for i in ('0', '125', '250','375','500','625','750','875')]
        self.D_BTYPES = [f'btype_{i}' for i in ('-180','-90','90','180')]
        self.D_MTYPES = [f'mtype_{i}' for i in ('0','25','50','75','99')]
                
        self.D_PARAM_LOCS = tuple(
            [f'{self.PARAM_LOC_TOKEN}_{i}' for i in range(self.MAX_PARAM_TOKENS)]
        )
        self.D_STRUCT_LOCS = tuple(
            [f'{self.STRUCT_LOC_TOKEN}_{i}' for i in range(self.MAX_HOLES)]
        )
                
        self.CAT_PARAM_TYPES = ['mind', 'dtype', 'stype','btype','mtype']
        self.FLOAT_PARAM_TYPES = ['dflt', 'sflt', 'mflt', 'bflt']
        
        self.DEF_PARAM_TYPES = self.CAT_PARAM_TYPES + self.FLOAT_PARAM_TYPES
        
        self.PRT_FNS = ['draw']

                
    def get_input_shape(self):
        return [self.VDIM, self.VDIM, 1]

    def group_sample_sub_prog(self, scope, counts):
        return group_sample_sub_prog(self, scope, counts)
    
    def get_group_sample_params(self):
        STROKE_OPTS = [1,   2,  3, 4,  5,  6,  7,   8, 9,  10,  11,     12]
        STROKE_PROB = [0.125,0.4,1.,1.5,1.5,1.5,1.25,1.,0.75,0.5,0.25,0.125]

        STROKE_PROB = lu.norm_np(STROKE_PROB)

        num_strokes = np.random.choice(STROKE_OPTS, p=STROKE_PROB)
        
        num_groups = random.randint(0, min(4, num_strokes-1))

        HOLE_OPTS = [0,1,2,3,4]
        HOLE_PROB = lu.norm_np([1,.5,.25,.125,.0625])
        num_holes = np.random.choice(HOLE_OPTS, p =HOLE_PROB)
        
        return {
            'num_strokes': num_strokes,
            'num_groups': num_groups,
            'num_holes': num_holes
        }


    def make_global_scope(self, tokens):
        return {
            'max_sind': tokens.count('sep') - 1
        }
    
    def get_def_local_scope(self):
        return {'tokens': []}

    def get_def_global_scope(self):
        return {'max_sind': -1}
        
    def make_next_deriv_scopes(self, tkn, ipc, tar_ipc, global_scope, local_scope):
          
        if tkn == 'sep':

            global_scope['max_sind'] += 1
            
            assert ipc != tar_ipc
            nq = [
                (deepcopy(local_scope), ipc, tar_ipc+1),
                (deepcopy(local_scope), ipc-1, tar_ipc)
            ]
                
        else:
            if ipc != tar_ipc:
                nq = [(
                    local_scope, ipc, tar_ipc
                )]
            else:
                nq = []

        return nq
        
    def build_struct_scope(self, context, token):
        context.name = token

        if 'sep' not in context.tokens:
            local = context.tokens
            top = []
        else:            
            ui = len(context.tokens) - context.tokens[::-1].index('sep') - 1
            top = context.tokens[:ui+1]
            local = context.tokens[ui+1:]

        context.struct_scope = {
            'top': top,
            'local': local
        }        
                                
    def check_valid_prog(self, P, ret_vdata=False):
        
        if P.has_soft_error():
            return None

        if not ret_vdata:
            return True
        
        try:
            img = P.make_image(break_on_error=True)
        except Exception as e:
            print(e)
            img = None
            
        return img
    
    def sample_det_prog(self, sample_params):
        return sample_det_tokens(
            self, sample_params['num_strokes'], sample_params['num_groups']
        )        
        
    def render_group(self, images, name=None, rows=1):
        if rows == 0:
            print("Bad behavior on omni render group")
            return
        
        elif rows == 1:
            f, axarr = plt.subplots(rows,len(images),figsize=(30,3))
            for i in range(len(images)):
                axarr[i].imshow(
                    images[i].cpu().numpy(), origin='lower',
                    cmap='gray', vmin=0.0, vmax=1.0
                )
                axarr[i].axis("off")
        else:
            num_per_row = math.ceil(len(images) / rows)
            f, axarr = plt.subplots(rows, num_per_row, figsize=(30,3 * rows))
            j = 0
            k = 0
            
            for i in range(len(images)):
                axarr[k][j].imshow(
                    images[i].cpu().numpy(), origin='lower',
                    cmap='gray', vmin=0.0, vmax=1.0
                )
                axarr[k][j].axis("off")
            
                j += 1

                if j == num_per_row:
                    k += 1
                    j = 0
            
        if name is None:
            plt.show()
        else:
            plt.savefig(f'{name}.png')
            
        plt.close()
        
        
#################
#################
### Random Program Sampling Logic
#################
#################


class CursorPos:
    def __init__(self, ex):
        self.ex = ex
        self.tr_corner = [0.,0.]
        self.bl_corner = [0., 0.]
        
        self.pos = torch.tensor([0.0,0.0]).float()
        self.strokes = []        
        self.state = 'off'
        
    def sample_dtype(self):
        dtype = lu.sample_dist((self.ex.smp_dtype_opts, self.ex.smp_dtype_prob))
        return dtype
        
    def sample_dflt(self):
        dflt = lu.sample_dist((self.ex.smp_dflt_opts, self.ex.smp_dflt_prob))
        return dflt
    
    def sample_stype(self):
        stype = lu.sample_dist((self.ex.smp_stype_opts, self.ex.smp_stype_prob))
        return stype
    
    def sample_sflt(self):
        sflt = lu.sample_dist((self.ex.smp_sflt_opts, self.ex.smp_sflt_prob))
        return sflt

    def check_valid(self, degree, scale):
        radians = (degree * math.pi) / 180.

        delta = torch.tensor([scale * math.cos(radians), scale * math.sin(radians)]).float()

        new_pos = self.pos + delta

        new_tr_corner = [
            max(self.tr_corner[0], new_pos[0].item()),
            max(self.tr_corner[1], new_pos[1].item()), 
        ]

        new_bl_corner = [
            min(self.bl_corner[0], new_pos[0].item()),
            min(self.bl_corner[1], new_pos[1].item()), 
        ]

        if (new_tr_corner[0] - new_bl_corner[0]) >= 2.0 or \
           (new_tr_corner[1] - new_bl_corner[1]) >= 2.0:

            return False

        self.tr_corner = new_tr_corner
        self.bl_corner = new_bl_corner
        
        if self.state == 'on':
            self.strokes.append((self.pos, new_pos))
            self.pos = new_pos
            
        return True
        

    def sample_draw_params(self):

        while True:

            dtype = self.sample_dtype()
            dflt = self.sample_dflt()

            stype = self.sample_stype()
            sflt = self.sample_sflt()

            if sflt < 0.0:
                sflt = 0.

            if sflt > 1.0:
                sflt = 1.0
                
            if self.check_valid(dtype + dflt, stype + sflt):
                                    
                return (dtype, dflt, stype, sflt)

def sample_move(cp):

    mind = random.randint(0, len(cp.strokes)-1)
    mtype = lu.sample_dist((cp.ex.smp_mtype_opts, cp.ex.smp_mtype_prob))        
    mflt = lu.sample_dist((cp.ex.smp_mflt_opts, cp.ex.smp_mflt_prob))

    if '_0' in mtype and mflt < 0:
        mflt = 0

    if '_99' in mtype and mflt > 0:
        mflt = 0
    
    return [
        'move', f'mind_{mind}', mtype,
        cp.ex.REV_FLOAT_MAP[('mflt', mflt)]
    ]

def sample_draw(cp):
    dtype,dflt,stype,sflt = cp.sample_draw_params()
        
    return [
        'draw',
        f'dtype_{dtype}',
        cp.ex.REV_FLOAT_MAP[('dflt', dflt)],
        'scale',
        f'stype_{int(stype*1000)}',        
        cp.ex.REV_FLOAT_MAP[('sflt', sflt)]
    ]

def sample_bow(cp):
    
    btype = lu.sample_dist((cp.ex.smp_btype_opts, cp.ex.smp_btype_prob))        
    bflt = lu.sample_dist((cp.ex.smp_bflt_opts, cp.ex.smp_bflt_prob))

    if '_-180' in btype and bflt < 0.0:
        bflt = 0.

    if '_180' in btype and bflt > 0.0:
        bflt = 0.
    
    return ['bow', btype, cp.ex.REV_FLOAT_MAP[('bflt', bflt)]]
    
def sample_stroke(cp):
    tokens = []
    
    if random.random() > 0.5:
        tokens += sample_bow(cp)

    tokens += sample_draw(cp)

    return tokens
    
def sample_det_tokens(ex, num_strokes, num_groups):

    tokens = [ex.START_TOKEN]

    cp = CursorPos(ex)
    
    cmds = ['stroke'] * num_strokes

    insert_inds = random.sample(range(1,len(cmds)), num_groups)
    insert_inds.sort(reverse=True)

    for ii in insert_inds:
        cmds.insert(ii, 'grp_break')
        
    cp.state = 'on'

    for cmd in cmds:
        if cmd == 'stroke':            
            tokens += ['on'] + sample_stroke(cp)

        elif cmd == 'grp_break':
            cp.state = 'off'
            
            if random.random() > 0.5 and len(cp.strokes) > 0: 
                tokens += ['off'] + sample_move(cp) 
            else:
                if random.random()  > 0.8:
                    tokens += ['off'] + sample_draw(cp) + ['off'] + sample_draw(cp)
                else:
                    tokens += ['off'] + sample_draw(cp)
                
            cp.state = 'on'
            
        else:
            assert False
                
    tokens.append('end')
    
    return tokens


## GROUP SAMPLING HELPER FUNCTIONS

def group_sample_sub_prog(ex, scope, counts):

    cp = CursorPos(ex)
    
    ltokens = scope['local']
    
    struct = []
    tokens = []
    params = {}

    def add_param(v, pt):
        tn = f'{ex.HOLE_PARAM_TOKEN}_{pt}_{counts["hpcnt"]}'
        counts["hpcnt"] += 1
        tokens.append(tn)
        params[tn] = v

    def add_struct(t):
        struct.append(t)
        tokens.append(t)

    cmds = []

    extra = True
    
    if len(ltokens) > 2 and ltokens[-3] == 'bow':
        cmds.append('draw')
        
    elif len(ltokens) > 1 and ltokens[-1] == 'off':

        if random.random() < 0.7:
            cmds += ['draw']
        elif random.random() < 0.8:
            cmds += ['off', 'draw', 'draw']
        else:
            cmds += ['move']
            
        extra = False

    else:
        if random.random() < 0.25:
            cmds.append('empty')
            extra = False

    if extra:
                
        if random.random() < .2:
            cmds.append('off')
            cmds.append('draw')
            if random.random() < 0.2:
                cmds.append('off')
                cmds.append('draw')                
            
        ns = np.random.choice([1,2],p=lu.norm_np([.8,.2]))

        for _ns in range(ns):
            if _ns != ns - 1:
                cmds.append('on')
                
            if random.random() > .5:
                cmds.append('bow')
            cmds.append('draw')

    for cmd in cmds:
        if cmd in ('off', 'on', 'empty'):
            add_struct(cmd)

        elif cmd == 'draw':
            dparams = sample_draw(cp)
    
            add_struct('draw')
            add_param(dparams[1], 'dtype')
            add_param(dparams[2], 'dflt')
            add_struct('scale')
            add_param(dparams[4], 'stype')            
            add_param(dparams[5], 'sflt')

        elif cmd == 'bow':
            
            bparams = sample_bow(cp)
            add_struct('bow')
            add_param(bparams[1], 'btype')
            add_param(bparams[2], 'bflt')

        elif cmd == 'move':

            cp.strokes = [None] * ltokens.count('on')
            mparams = sample_move(cp)

            add_struct('move')
            add_param(mparams[0], 'mind')
            add_param(mparams[1], 'mtype')
            add_param(mparams[2], 'mflt')
            
        else:
            assert False, f'bad cmd {cmd}'

    return tokens, params, struct

    
################
################
#### LOADING OMNIGLOT DATA LOGIC
################
################

def load_dynamic_gt_targets(loc_ex, path, num, _A=None,_B=None):
    d = torch.load(path)
    data = d['data']

    def make_dummy_prog(gi):
        prog = Program(loc_ex)
        
        gt_trajs = data[gi][1]
        
        for gtj in gt_trajs:            
            stroke = Stroke(torch.zeros(5,2))
            stroke.traj = gtj
            prog.strokes.append(stroke)

        return prog

    groups = d['groups']

    M = {}

    for gname, ginds, _ in groups:
        if gname not in M:
            if len(M) < num:
                M[gname] = []

            elif len(M) >= num:
                continue
                
        M[gname] += [gi for gi in ginds]
    
    R = []
    for gname, ginds in tqdm(list(M.items())):
        G = []
        for gi in ginds:
            G.append(make_dummy_prog(gi))
        R.append((gname, G))
            
    return R
    
def load_static_gt_targets(
    loc_ex, path, num, group_num, mode
):

    d = torch.load(path)
    data = d['data']

    if group_num is None:
        return load_dynamic_gt_targets(loc_ex, path, num)
    
    def make_dummy_prog(gi):
        prog = Program(loc_ex)
        
        gt_trajs = data[gi][1]
        
        for gtj in gt_trajs:            
            stroke = Stroke(torch.zeros(5,2))
            stroke.traj = gtj
            prog.strokes.append(stroke)

        return prog

    groups = d['groups']
    R = []
    for gname, ginds, gvalid in tqdm(groups[:num]):        
        if (len(ginds) == group_num):
            if mode == 'coseg' and not gvalid:
                continue
            
            G = []
            for gi in ginds:
                G.append(make_dummy_prog(gi))

            R.append((gname, G))
            
        else:
            assert group_num == 1
            for gi in ginds:
                R.append((gname, make_dummy_prog(gi)))

    return R


def load_fsg_tasks(
    loc_ex, path, num_tasks, num_chars, group_num = 5
):

    d = torch.load(path)
    data = d['data']
    
    def make_dummy_prog(gi):
        prog = Program(loc_ex)
        prog.gind = gi
        gt_trajs = data[gi][1]
        
        for gtj in gt_trajs:            
            stroke = Stroke(torch.zeros(5,2))
            stroke.traj = gtj
            prog.strokes.append(stroke)

        return prog

    groups = d['groups']
    R = {}
    
    for gname, ginds, gvalid in groups:
        assert (len(ginds) == group_num)

        if gname not in R:

            if len(R) >= num_tasks:
                continue
            
            R[gname] = []
        
        for gi in ginds:
            R[gname].append(make_dummy_prog(gi))

    for name, progs in R.items():
        if len(progs) < num_chars:
            assert False

    return R
