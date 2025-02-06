import sys
sys.path.append('domains')
sys.path.append('data/shape')
import dom_common as dc
from tar_data import TargetDataset
import executors.shape.ex_shape as ex
from utils import device
import torch
from tqdm import tqdm
import random
import scipy.optimize as sopt
from copy import deepcopy

# Default worst metric value
WMV = 10.

SHAPE_CMN_ARGS = [

    # mst -> maximum tokens used in a template program
    # mdt -> maximum tokens used in a structural expansion
    # mpt -> maximum number of parameter tokens
    
    # msl -> sets bound for program length during synthetic sampling
    # mp - > number of visual codes for each input

    # est -> threshold for early stopping logic
    # esm -> metric used in early stopping

    # emn -> reconstruction metric in objective
    # ddp -> penalty for differences in tokens used between instantiated programs and template programs

        
    ('-mst', '--max_struct_tokens', 64, int),
    ('-mdt', '--max_deriv_tokens', 24, int),
    ('-mpt', '--max_param_tokens', 80, int),

    ('-msl', '--max_seq_len', 128, int),
    ('-mp', '--max_prim_enc', 4, int),

    ('-est', '--es_threshold', 0.00001,  float),
    ('-esm', '--es_metric', 'Obj',  str),                
    
    ('-emn', '--eval_metric_name', 'Cdist', str),    
    ('-ddp', '--ddof_pen', 0.001, float),

    # visual input type (prim or voxel)
    ('-vt', '--vin_type', 'prim', str),

    # if using voxels, what dimension should be
    ('-vd', '--voxel_dim', 64, int),
]

SHAPE_PT_ARGS = [
    # bs -> pretraining batch size, note: set this lower when using 'voxel' mode
    # beams -> pretraining beam size

    ('-bs', '--batch_size', 32, int),
    ('-beams', '--beams', 5, int),
]

SHAPE_FT_ARGS = [

    # thr -> threshold beyond which record new best objective value
    # bs -> batch size for finetuning

    # ts -> training size for finetuning
    # evs -> val size for finetuning
    # ets -> test size for finetuning

    # beams -> beam size used during inference phase
    # esb -> beam size used during early stopping logic for training phase
    # eps -> maximum epochs used during each training phase
    # evp -> how often to do eval early stopping logic during training phase

    # fsg_net -> few-shot gen num tasks
    # fsg_ppt -> number of prompts for each task 
    # fsg_gpp -> how many generations to produce for each prompt
    # fsg_pen -> how many images are in the prompt set
    # fsg_ten -> how many images are in the held out target set

    ('-thr', '--threshold', 0.00001, float),
    ('-bs', '--batch_size', 20, int), # if using voxel, potentially reduce this

    ('-ts', '--train_size', 1000, int),    
    ('-evs', '--eval_size', 100,  int),
    ('-ets', '--etest_size', 100,  int),
    
    ('-beams', '--beams', 5, int),
    ('-esb', '--es_beams', 5, int),
    ('-eps', '--epochs', 50, int),
    ('-evp', '--eval_per', 5, int),

    # for shape use slightly less generations for time
    ('-wts', '--ws_train_size', 10000, int),

    ('-fsg_nt', '--fsg_num_tasks', 100, int),
    ('-fsg_ppt', '--fsg_prompts_per_task', 1, int),
    ('-fsg_gpp', '--fsg_gens_per_prompt', 5, int),
    ('-fsg_pen', '--fsg_prompt_ex_num', 5, int),
    ('-fsg_ten', '--fsg_target_ex_num', 5, int),
]

        
# Class that defines the domain
class SHAPE_DOMAIN(dc.DOMAIN_BASE):
    def __init__(self):
        self.base_init()
        self.name='shape'
        self.device = device        
        self.ex_class = ex

    def make_executor(self, args):
        config = {
            'MAX_TOKENS': args.max_seq_len,
            'MAX_STRUCT_TOKENS': args.max_struct_tokens,
            'MAX_PARAM_TOKENS': args.max_param_tokens,
            'VIN_TYPE': args.vin_type,
            'VOXEL_DIM': args.voxel_dim,
        }        
        self.executor = ex.ShapeExecutor(config)        
        if self.executor.VIN_TYPE == 'voxel':
            args.eval_metric_name = 'iou'
        
    def cornerize(self, P):

        if 'OFFT' not in self.__dict__:
            self.OFFT = torch.tensor([
                [-1,-1,-1],
                [-1,-1,1],
                [-1,1,-1],
                [-1,1,1],
                [1,-1,-1],
                [1,-1,1],
                [1,1,-1],
                [1,1,1],            
            ], device=P.device).unsqueeze(0).float()
        
        cpts = P[:,3:].unsqueeze(1).repeat(1, 8, 1)        
        
        offs = (P[:,:3].unsqueeze(1).repeat(1, 8, 1) / 2.) * self.OFFT
        
        corners = cpts + offs

        return corners
        
    def calc_recon_metric(self, A, B):

        nA = (A.abs().sum(dim=-1) > 0).sum().item()
        nB = (B.abs().sum(dim=-1) > 0).sum().item()
        
        # 20 x 8 x 3
        corA = self.cornerize(A)
        corB = self.cornerize(B)

        ACD = (
            corA.view(corA.shape[0], 1, 8, 1, 3) - corB.view(1, corB.shape[0], 1, 8, 3)
        ).norm(dim=-1)

        corner_D = (
            ACD.min(dim=3).values.mean(dim=-1) + ACD.min(dim=2).values.mean(dim=-1)
        )

        cd = self.calc_min_ad(corner_D, nA, nB)

        return {
            'Cdist': cd,
        }
        

    def calc_min_ad(self, D, nA, nB):
        
        D[nA:,:nB] = WMV
        D[:nA,nB:] = WMV
                
        if torch.isnan(D).any():
            print("Failed distance check with nan")
            return WMV
        
        try:
            assignment = sopt.linear_sum_assignment(D.cpu().numpy())
            d = D[assignment].mean().item()
            
        except Exception as e:            
            print(f"Failed distance check with {e} : {D}")
            return WMV
        
        return d

    def calc_voxel_recon_metric(self, A, B):
        Av = A > 0
        Bv = B > 0
        inter = (Av & Bv).sum().item() + 1e-8
        union = (Av | Bv).sum().item() + 1e-8
        _iou = inter * 1. / union
        return {
            'iou': _iou
        }
                
    
    def get_vis_metric(self, prog, gt, extra=None, prog_info=None):
        try:
            p_scene = self.executor.execute(prog)            
        except Exception as e:
            return None, None, None

        p_vdata = self.executor.conv_scene_to_vinput(p_scene)

        with torch.no_grad():
            if self.executor.VIN_TYPE == 'prim':
                recon_metrics = self.calc_recon_metric(p_vdata.to(gt.device), gt)            

            elif self.executor.VIN_TYPE == 'voxel':
                recon_metrics = self.calc_voxel_recon_metric(p_vdata.to(gt.device), gt)
                        
            else:
                assert False
            
        mval = recon_metrics[self.args.eval_metric_name]

        prog_len = len(prog.split())
        
        recon_metrics.update({
            'prog_len': prog_len,
            'mval_cnt': 1
        })

        if prog_info is not None:
            ddof = self.get_deriv_dof(prog_info)
            recon_metrics['ddof'] = ddof
            
            if self.args.ddof_pen != 0.0:
                if self.get_obj_dir() == 'high':        
                    metric = mval - (self.args.ddof_pen * ddof)
                elif self.get_obj_dir() == 'low':
                    metric = mval + (self.args.ddof_pen * ddof)
                else:
                    assert False
            else:
                metric = mval                
        else:
            metric = mval
                        
        return p_scene, metric, recon_metrics
                                
    def load_real_data(self, mode='train'):
        TD = TargetDataset(self.args, self.device, self.executor, mode)
        TD.name = 'shape'
        TD.ex = self.executor
        return TD
    
    def init_metric_val(self):
        if self.get_obj_dir() == 'high':
            return 0.
        else:
            return WMV

    def get_obj_dir(self):
        if self.executor.VIN_TYPE == 'voxel':
            return 'high'
        else:
            return 'low'        
            
    def make_blank_visual_batch(self, batch_size, group_size, device):

        if self.executor.VIN_TYPE == 'voxel':
            assert group_size is not None
            VOXELD = self.executor.VOXEL_DIM
            return torch.zeros(
                batch_size,
                group_size,
                VOXELD,
                VOXELD,
                VOXELD,
                device=device
            ).float()
        
        if group_size is not None:
            return torch.zeros(
                batch_size,
                group_size,
                self.executor.MAX_PARTS,
                6,
                device=device
            ).float()
        else:
            return torch.zeros(
                batch_size,
                self.executor.MAX_PARTS,
                6,
                device=device
            ).float()

    def get_synth_data_cls(self):
        return dc.SynthDataset
        
    def get_cmn_args(self):
        return SHAPE_CMN_ARGS

    def get_pt_arg_list(self):
        return SHAPE_PT_ARGS

    def get_ft_arg_list(self):
        return SHAPE_FT_ARGS

    def extra_prob_eval_log_info(self):
        return [
            ('Cdist', 'info_Cdist', 'mval_cnt'),
            ('IoU', 'iou', 'mval_cnt'),
            ('DDOF', 'info_ddof', 'mval_cnt'),
        ]
