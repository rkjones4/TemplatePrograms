import sys
sys.path.append('domains')
sys.path.append('data/layout')
import dom_common as dc
import tar_data as TD 
import executors.layout.ex_layout as ex
from utils import device
import torch

LYT_CMN_ARGS = [

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
    ('-mdt', '--max_deriv_tokens', 16, int),
    ('-mpt', '--max_param_tokens', 72, int),

    ('-msl', '--max_seq_len', 100, int),
    ('-mp', '--max_prim_enc', 16, int),
    
    ('-est', '--es_threshold', 0.0001,  float),
    ('-esm', '--es_metric', 'Obj',  str),
        
    ('-emn', '--eval_metric_name', 'color_iou', str),    
    ('-ddp', '--ddof_pen', 0.001, float),
]

LYT_PT_ARGS = [
    # bs -> pretraining batch size
    # beams -> pretraining beam size
    
    ('-bs', '--batch_size', 32, int),
    ('-beams', '--beams', 5, int),
]

LYT_FT_ARGS = [

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
    # fsg_ppt -> number of prompts for each task (can be set higher for layout)
    # fsg_gpp -> how many generations to produce for each prompt
    # fsg_pen -> how many images are in the prompt set
    # fsg_ten -> how many images are in the held out target set
    
    ('-thr', '--threshold', 0.0001, float),
    ('-bs', '--batch_size', 20, int),

    ('-ts', '--train_size', 216, int),    
    ('-evs', '--eval_size', 72,  int),
    ('-ets', '--etest_size', 168,  int),
    
    ('-beams', '--beams', 5, int),
    ('-esb', '--es_beams', 5, int),
    ('-eps', '--epochs', 50, int),
    ('-evp', '--eval_per', 5, int),

    ('-fsg_nt', '--fsg_num_tasks', 168, int),
    ('-fsg_ppt', '--fsg_prompts_per_task', 1, int),
    ('-fsg_gpp', '--fsg_gens_per_prompt', 5, int),
    ('-fsg_pen', '--fsg_prompt_ex_num', 5, int),
    ('-fsg_ten', '--fsg_target_ex_num', 5, int),

]

                    
# Class that defines the domain
class LAYOUT_DOMAIN(dc.DOMAIN_BASE):
    def __init__(self):
        self.domain_name = 'layout'
        self.base_init()
        self.name='layout'
        self.device = device        
        self.ex_class = ex
        
    def make_executor(self, args):
        config = {
            'MAX_TOKENS': args.max_seq_len,
            'MAX_STRUCT_TOKENS': args.max_struct_tokens,
            'MAX_PARAM_TOKENS': args.max_param_tokens,
        }

        self.executor = ex.LayExecutor(config)
                
    def get_vis_metric(self, prog, gt, extra=None, prog_info=None):

        try:
            P = self.executor.prog_cls(self.executor)
            P.run(prog.split())
            vdata = P.make_image()
        except:
            return None, None, None
        
        recon_metrics = self.pixel_recon_metrics(vdata, gt)            
        mval = recon_metrics[self.args.eval_metric_name]

        prog_len = len(prog.split())
        
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

        recon_metrics.update({
            'prog_len': prog_len,
            'mval_cnt': 1.
        })
                
        return vdata, metric, recon_metrics

            
    def pixel_recon_metrics(self, pixels, gt):
                    
        pixels = pixels.to(gt.device) 
        
        def get_occ(inp):
            assert len(inp.shape) == 3
            finp = inp.view(-1, 3)

            o_a = (finp > 0.).any(dim=1)
            o_r = (finp[:,0] > 0.5)
            o_gn = (finp[:,1] > 0.5)
            o_b = (finp[:,2] > 0.5)
            o_gr = (finp.sum(dim=1) > 1.0)

            return o_a, o_r, o_gn, o_b, o_gr
            
        p_a, p_r, p_gn, p_b, p_gr = get_occ(pixels)
        g_a, g_r, g_gn, g_b, g_gr = get_occ(gt)

        CI = (p_r & g_r).sum() +\
            (p_gn & g_gn).sum() +\
            (p_b & g_b).sum() +\
            (p_gr & g_gr).sum()

        CU = (p_a | g_a).sum()

        cIoU = (CI / (CU + 1e-8)).item()
        
        return {
            'color_iou': cIoU
        }
            
    def load_real_data(self, mode='train'):
        return TD.TargetDataset(self.args, self.device, self.executor, mode)
            
    def init_metric_val(self):
        return 0.

    def get_obj_dir(self):
        return 'high'        
            
    def make_blank_visual_batch(self, batch_size, group_size, device):
        if group_size is not None:
            return torch.zeros(
                batch_size,
                group_size,
                self.executor.VDIM,
                self.executor.VDIM,
                3,
                device=device
            ).float()
        else:
            return torch.zeros(
                batch_size,
                self.executor.VDIM,
                self.executor.VDIM,
                3,
                device=device
            ).float()

    def get_synth_data_cls(self):
        return dc.SynthDataset
        
    def get_cmn_args(self):
        return LYT_CMN_ARGS

    def get_pt_arg_list(self):
        return LYT_PT_ARGS

    def get_ft_arg_list(self):
        return LYT_FT_ARGS

    def extra_prob_eval_log_info(self):
        return [
            ('Color Iou', 'info_color_iou', 'mval_cnt'),
            ('DDOF', 'info_ddof', 'mval_cnt'),
        ]

