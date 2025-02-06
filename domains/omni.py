import sys
sys.path.append('domains')
import dom_common as dc
import executors.omniglot.ex_omni as ex

import random, torch
from utils import device
from copy import deepcopy

TARGET_TRAIN_PATH = f'data/omni/gt_omni_train.txt'
TARGET_VAL_PATH = f'data/omni/gt_omni_val.txt'
TARGET_TEST_PATH = f'data/omni/gt_omni_test.txt'

OMN_CMN_ARGS = [

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
    ('-mpt', '--max_param_tokens', 64, int),

    ('-msl', '--max_seq_len', 128, int), 
    ('-mp', '--max_prim_enc', 16, int),

    ('-est', '--es_threshold', 0.00001,  float),    
    ('-esm', '--es_metric', 'Obj',  str),

    ('-emn', '--eval_metric_name', 'icd', str),
    ('-ddp', '--ddof_pen', 0.001, float),

    # Paths for target data
    ('-ttrp', '--target_train_path', TARGET_TRAIN_PATH, str),
    ('-tvp', '--target_val_path', TARGET_VAL_PATH, str),
    ('-ttep', '--target_test_path', TARGET_TEST_PATH, str),
]

OMN_PT_ARGS = [
    # bs -> pretraining batch size
    # beams -> pretraining beam size
    
    ('-bs', '--batch_size', 40, int),
    ('-beams', '--beams', 5, int),    
]

OMN_FT_ARGS = [

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
    ('-bs', '--batch_size', 20, int),

    ('-ts', '--train_size', 964, int),    
    ('-evs', '--eval_size', 100,  int),
    ('-ets', '--etest_size', 659,  int),        

    ('-beams', '--beams', 5, int),
    ('-esb', '--es_beams', 5, int),
    ('-eps', '--epochs', 50, int),
    ('-evp', '--eval_per', 5, int),    

    ('-fsg_nt', '--fsg_num_tasks', 659, int),
    ('-fsg_ppt', '--fsg_prompts_per_task', 1, int),
    ('-fsg_gpp', '--fsg_gens_per_prompt', 5, int),
    ('-fsg_pen', '--fsg_prompt_ex_num', 5, int),
    ('-fsg_ten', '--fsg_target_ex_num', 5, int),

    
]

class TargetDataset(dc.TargetBase):
    def __init__(
        self, args, device, mode, loc_ex
    ):
        
        self.mode = mode
        self.args = args
        self.device= device
                    
        self.vinput = []
        self.sem_data = []
        self.train_keys = []
        self.val_keys = []
        self.test_keys = []
        self.stroke_data = []

        self.train_group_names = []
        self.val_group_names = []
        self.test_group_names = []

        self.extra_gt_data = None
        self.eval_batch_size = 1
        
        if self.mode == 'fsg':
            print("Loading FSG Real Data")
            with torch.no_grad():
                self.init_fsg_data(loc_ex)
                
            return

        if mode == 'coseg':
            args.train_size = 1
            args.eval_size = 1
        
        make_target_fn = ex.load_dynamic_gt_targets
        
        with torch.no_grad():
            
            train_data = make_target_fn(
                loc_ex,
                args.target_train_path,
                args.train_size,
                args.max_vis_inputs,
                mode
            )
            
            val_data = make_target_fn(
                loc_ex,
                args.target_val_path,
                args.eval_size,
                args.max_vis_inputs,
                mode
            )
                                
            test_data = ex.load_static_gt_targets(
                loc_ex,
                args.target_test_path,
                args.etest_size,
                args.max_vis_inputs,
                mode
            )   

            to_load = [
                (train_data, self.train_keys, self.train_group_names),
                (val_data, self.val_keys, self.val_group_names),
                (test_data, self.test_keys, self.test_group_names)
            ]
                                   
            for grp_set, keys, sg_names in to_load:
                
                for gname, group in grp_set:
                    kg = []

                    for d in group:

                        sem_seg = d.make_sem_seg().detach().cpu()
                            
                        kg.append(len(self.vinput))
                        self.vinput.append(d.make_image().detach().cpu())
                        self.sem_data.append(sem_seg)
                        self.stroke_data.append(d.get_stroke_data())
                            
                    sg_names.append(gname)
                    keys.append(kg)

                                    
        self.train_keys = torch.tensor(self.train_keys).long()
        self.val_keys = torch.tensor(self.val_keys).long()
        self.test_keys = torch.tensor(self.test_keys).long()
        
        self.sem_data = torch.stack(self.sem_data,dim=0)

        self.extra_gt_data = self.stroke_data
        
        self.iter_num = 0
        self.size = args.eval_size            
        self.eval_size = args.eval_size
        
        assert mode != 'fsg'

        self.vinput = torch.stack(self.vinput,dim=0)
        if len(self.vinput.shape) == 3:
            self.vinput = self.vinput.unsqueeze(-1)

        self.dyn_train_keys = self.train_keys
        self.dyn_val_keys = self.val_keys
        self.sample_dyn_keys()
            
        print(f"Key sizes ({self.vinput.shape[0]})")
        print(f"train {self.train_keys.shape[0]}")
        print(f"val {self.val_keys.shape[0]}")
        print(f"test {self.test_keys.shape[0]}")            
        
    def sample_dyn_keys(self):
        self.train_keys = []
        self.val_keys = []
        
        args = self.args

        mvi = args.max_vis_inputs
                
        for key_set, target_size, dyn_key_set in [
            (self.train_keys, args.train_size, self.dyn_train_keys),
            (self.val_keys, args.eval_size, self.dyn_val_keys),
        ]:
            ninds = torch.randperm(dyn_key_set.shape[1])
            
            while len(key_set) < target_size:

                inds = ninds[:mvi]
                ninds = ninds[mvi:]
            
                for i in range(dyn_key_set.shape[0]):
                    if len(key_set) >= target_size:
                        break                                

                    key_set.append(dyn_key_set[i,inds])
                    
        self.train_keys = torch.stack(self.train_keys,dim=0)
        self.val_keys = torch.stack(self.val_keys,dim=0)
        
    def init_fsg_data(self, loc_ex):

        args = self.args
        
        assert args.fsg_prompts_per_task * args.max_vis_inputs == args.fsg_prompt_ex_num
        
        test_tasks = ex.load_fsg_tasks(
            loc_ex,
            args.target_test_path,
            args.fsg_num_tasks,
            args.fsg_prompt_ex_num + args.fsg_target_ex_num
        )
        
        self.fsg_tasks = {}

        for tname, task_data in test_tasks.items():
            task_inds = []
            for td in task_data:
                task_inds.append(len(self.vinput))
                self.vinput.append(td.make_image().detach().cpu())
                self.stroke_data.append(td.get_stroke_data())
                
            prompt_inds = deepcopy(task_inds[:args.fsg_prompt_ex_num])

            prompts = []
            cur = []
            while True:
                if len(cur) == args.max_vis_inputs:
                    prompts.append(cur)
                    cur = []
                if len(prompt_inds) == 0:
                    break
                cur.append(prompt_inds.pop(0))
            
            target_inds = task_inds[args.fsg_prompt_ex_num:args.fsg_target_ex_num+args.fsg_prompt_ex_num]
            
            self.fsg_tasks[tname] = {'prompts': prompts, 'targets': target_inds}
            
        self.fsg_prompt_keys = torch.cat((
            [torch.tensor(v['prompts']) for v in self.fsg_tasks.values()]
        ),dim=0)

        self.vinput = torch.stack(self.vinput,dim=0)
        if len(self.vinput.shape) == 3:
            self.vinput = self.vinput.unsqueeze(-1)

        print(f"Key sizes ({self.vinput.shape[0]})")
        print(f"prompt {self.fsg_prompt_keys.shape[0]}")            

        self.extra_gt_data = self.stroke_data
        
        
# Class that defines the domain
class OMNI_DOMAIN(dc.DOMAIN_BASE):
    def __init__(self):
        self.domain_name = 'omni'
        self.base_init()
        self.name = 'omni'
        self.device = device
        
    def make_executor(self, args):
        config = {
            'MAX_TOKENS': args.max_seq_len,
            'MAX_STRUCT_TOKENS': args.max_struct_tokens,
            'MAX_PARAM_TOKENS': args.max_param_tokens,
        }
        
        self.executor = ex.OmnExecutor(config)
        self.metric_name = self.args.eval_metric_name

    def get_vis_metric(self, prog, gt, gsd=None, extra=None, prog_info=None):
        try:
            vdata,psd = self.executor.execute(prog, ret_strokes=True)
        except Exception as e:
            return None, None, None

        if extra is not None and gsd is None:
            gsd = extra
        
        try:
            with torch.no_grad():
                mval = self.vis_metric(vdata, gt, psd, gsd)
        except Exception as e:
            print(f"Failed to get vis metric with {e}")
            return self.init_metric_val(), None, None
        
        prog_len = len(prog.split())

        recon_metrics = {
            self.metric_name: mval,
            'prog_len': prog_len,
            'mval_cnt': 1
        }

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

        return vdata, metric, recon_metrics

    def vis_metric(
        self,
        pixels,
        gt,
        psd=None,
        gsd=None,
    ):
                
        return dc.image_chamfer(
            pixels.unsqueeze(0).cpu().numpy(),
            gt.unsqueeze(0).cpu().numpy()
        )[0]

    # get fine-tuning data
    def load_real_data(self, mode='train'):
        return TargetDataset(self.args, self.device, mode, self.executor)

    # more early stopping logic, what should the "bad" value of the metric be
    def init_metric_val(self):
        return 10.

    def get_obj_dir(self):
        return 'low'

    def make_blank_visual_batch(self, batch_size, group_size, device):
        if group_size is not None:
            return torch.zeros(
                batch_size,
                group_size,
                self.executor.VDIM,
                self.executor.VDIM,
                1,
                device=device
            ).float()
        else:
            return torch.zeros(
                batch_size,
                self.executor.VDIM,
                self.executor.VDIM,
                1,
                device=device
            ).float()
        
    def get_synth_data_cls(self):
        return dc.SynthDataset
                        
    def get_cmn_args(self):
        return OMN_CMN_ARGS

    def get_pt_arg_list(self):
        return OMN_PT_ARGS

    def get_ft_arg_list(self):
        return OMN_FT_ARGS

    def extra_prob_eval_log_info(self):
        return [
            ('CD', 'info_icd', 'mval_cnt'),
            ('DDOF', 'info_ddof', 'mval_cnt'),
        ]
    
