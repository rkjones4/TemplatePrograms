import torch
import numpy as np
from tqdm import tqdm
import utils
import os
import math
import utils
from utils import device

def infer_prob_programs(domain, net, data, train_pbest, val_pbest):
    args = domain.args
    
    path = args.infer_path
    
    net.eval()
                        
    results = {}    

    ITER_DATA = [
        (data.train_eval_iter, train_pbest, data.get_set_size('train'), 'train'),
        (data.val_eval_iter, val_pbest, data.get_set_size('val'), 'val'),
        (data.test_eval_iter, None, data.get_set_size('test'), 'test')
    ]

    metric_key = 'mvals'
    ex_key = 'execs'
    
    for gen, record, num, name in ITER_DATA:

        net.vis_mode = (name, net.iter_num)
        net.init_vis_logic()
        
        eval_res = {}
        
        utils.log_print(f"Inferring for {name}", args)

        for batch in \
            tqdm(gen(), total = num):
            
            keys = batch['bkeys']
            vinput = batch['vinput']
            if 'extra_gt_data' in batch:
                extra_gt_data = batch['extra_gt_data']
            else:
                extra_gt_data = None

            inp_batch = {
                'vdata': vinput,
                'extra_gt_data': extra_gt_data,
            }

            if 'vis_vdata' in batch:
                inp_batch['vis_vdata'] = batch['vis_vdata']
            
            eval_info, _eval_res = net.model_eval_fn(
                inp_batch,
                args.beams,
                ret_info=True
            )

            if 'info' not in eval_info or len(eval_info['info']) != 1:
                utils.log_print(f"Bad eval info {eval_info}", args)
                continue        
            
            for k,v in _eval_res.items():
                if k not in eval_res:
                    eval_res[k] = 0.
                eval_res[k] += v
            
            ERI = None
            
            if isinstance(eval_info['info'][0], list):
                ERI = eval_info['info'][0]
            else:
                ERI = eval_info['info']
                
            for _info in ERI:
                if _info is None:
                    continue
                for k,v in _info.items():
                    if isinstance(v, float):
                        if f'info_{k}' not in eval_res:
                            eval_res[f'info_{k}'] = 0.
                        eval_res[f'info_{k}'] += v
                                                            
            if record is not None:                                
                for i in range(len(eval_info['info'])):
                    if eval_info['info'][i] is None:
                        continue

                    if len(eval_info['info'][i]) == 0:
                        assert len(eval_info[ex_key][i]) == 0
                        assert len(eval_info[metric_key][i]) == 0
                        continue
                        
                    record.update(
                        keys[i],
                        eval_info['info'][i],
                        eval_info[ex_key][i],
                        eval_info[metric_key][i]
                    )
        
        results[name] = utils.print_results(
            domain.EVAL_LOG_INFO,
            eval_res,
            args,
            ret_early=True
        )
        
        utils.log_print(f'Eval res {name}:', args)
        for k,v in results[name].items():
            rv = round(v,3)
            utils.log_print(f"    {k}: {rv}", args)

        net.save_vis_logic()

    net.vis_mode = None
        
    return results
