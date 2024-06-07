import sys, os, torch, json
import utils
import numpy as np
import matplotlib.pyplot as plt
from utils import device
from copy import deepcopy, copy
from tqdm import tqdm
import train_utils as tru
from tqdm import tqdm

def train(domain):
    args = domain.get_pt_args()
    net = domain.load_new_net()
    
    # synthetic data sampled from the grammar randomly
    train_loader, val_loader = domain.get_synth_datasets()
    
    target_loader = domain.load_real_data()    

    assert target_loader is not None
    
    if args.load_model_path is not None:
        net.load_state_dict(
            torch.load(args.load_model_path)
        )
        
    if args.load_res_path is not None:
        res = json.load(open(args.load_res_path))
        try:
            starting_iter = int(res['eval_iters'][-1])
        except:
            starting_iter = 0            
    else:
        res = {
            'train_plots': {'train':{'iters':[]}, 'val':{'iters':[]}},
            'eval_plots': {'train':{}, 'val':{}, 'target': {}},
            'eval_iters': []
        }
        starting_iter = 0
        
    train_loader.iter_num = starting_iter
    last_print = starting_iter
    last_eval = starting_iter
    last_save = starting_iter
    
    if args.save_per is None:
        args.save_per = args.eval_per
    
    opt = torch.optim.Adam(
        net.parameters(),
        lr = args.lr,
        eps = 1e-6
    )

    save_model_count = 0

    if args.stream_mode == 'y':
        eval_data = [
            ('val', val_loader),
            ('target', target_loader),
        ]
    else:
        eval_data = [
            ('train', train_loader),
            ('val', val_loader),
            ('target', target_loader),
        ]        
    
    print("Starting Training")
    pbar = None

    while True:
        
        if pbar is None:
            pbar = tqdm(total=args.print_per)
            
        itn = train_loader.iter_num

        if itn > args.max_iters:
            break
        
        if itn - last_print >= args.print_per:
            do_print = True
            last_print = itn
            pbar.close()
            pbar = None
        else:
            do_print = False


        tru.run_train_epoch(
            args,
            res,
            net,
            opt,
            train_loader,
            val_loader,
            domain.TRAIN_LOG_INFO,
            do_print,
        )
        
        if pbar is not None:
            pbar.update(train_loader.iter_num-itn)

        
        if itn - last_eval >= args.eval_per:                    
            last_eval = itn
            
            tru.run_eval_epoch(
                args,
                res,
                net,
                eval_data,
                domain.EVAL_LOG_INFO,
                itn,
            )

        if itn - last_save >= args.save_per:
            
            last_save = itn
        
            utils.save_model(
                net.state_dict(),
                f"{args.outpath}/{args.exp_name}/models/net_CKPT_{save_model_count}.pt"
            )
            save_model_count += 1
        
