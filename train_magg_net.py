import time
from tqdm import tqdm
import os
import dill
import utils
import torch
from torch import nn, optim
from torch.nn import functional as F
import json
from copy import deepcopy
from train_prob_plad import DataGen

TMAGG_ARGS = [
    ('-mtdp', '--magg_tdata_path', None ,str),
    ('-mtm', '--magg_train_mode', 'all', str),
]

WS_TRAIN_LOG_INFO = [
    ('Train Loss', 'train_loss', 'nc'),
    ('Val Loss', 'val_loss', 'nc'),    
]


def convert_info_to_data(domain, infos):
    infd = domain.executor.make_infer_data(infos, domain.args)
    return infd

def magg_load_itns(domain):
    args = domain.args
    iter_num = [
        int(f.split('.')[0].split('_')[-1]) \
        for f in os.listdir(args.magg_tdata_path) \
        if 'magg_tdata' in f
    ]
    iter_num.sort()
    return iter_num

class dummy:
    def __init__(self, data):
        self.data = data

def magg_load_data(
    domain, itn
):
    
    args = domain.args    
    TPB = {}
    VPB = {}
    GD = []
            
    path = f'{args.magg_tdata_path}/magg_tdata_{itn}.pt'
    print(f"Loading {itn} from {path}")

    R = torch.load(path)
        
    for keys, infos in tqdm(list(zip(R['train']['keys'], R['train']['infos']))[:args.train_size]):
        d = convert_info_to_data(domain, infos)
        TPB[keys] = (None, d)
                
    for keys, infos in tqdm(list(zip(R['val']['keys'], R['val']['infos']))[:args.eval_size]):
        d = convert_info_to_data(domain, infos)
        VPB[keys] = (None, d)
            
    for infos in tqdm(R['gen_infos'][:args.ws_train_size]):
        GD.append(convert_info_to_data(domain, infos))

    TPBD = dummy(TPB)
    VPBD = dummy(VPB)

    utils.log_print(f"Sizes train/val/gen : {(len(TPBD.data),len(VPBD.data),len(GD))}", args)
    
    return TPBD, VPBD, GD

    
def train_magg_net(domain):
    
    args = domain.get_ft_args(TMAGG_ARGS)
    target_data = domain.load_real_data()
    
    magg_net = domain.load_magg_model(
        args.load_gen_model_path
    )

    magg_itns = magg_load_itns(domain)

    utils.log_print(f"Magg itns {magg_itns}", args)
    
    if args.magg_train_mode == 'all':
        utils.log_print(f"Combining all maggs", args)

        TPB = {}
        VPB = {}
        gen_data = []
        
        for itn in magg_itns:
            itn_train_pbest, itn_val_pbest, itn_gen_data = magg_load_data(domain, itn)

            TPB.update(itn_train_pbest.data)
            VPB.update(itn_val_pbest.data)            
            gen_data += itn_gen_data

        TPBD = dummy(TPB)
        VPBD = dummy(VPB)

        utils.log_print(f"Sizes train/val/gen : {(len(TPBD.data),len(VPBD.data),len(gen_data))}", args)
        
        magg_net = run_train_ep(
            domain, magg_net, TPBD, VPBD, gen_data, target_data,
        )

    elif 'last_k' in args.magg_train_mode:

        k = int(args.magg_train_mode.split(':')[1])

        itns = magg_itns[-k:]
        
        utils.log_print(f"Train MAGG with Last K ({itns})", args)

        TPB = {}
        VPB = {}
        gen_data = []
        
        for itn in itns:
            itn_train_pbest, itn_val_pbest, itn_gen_data = magg_load_data(domain, itn)

            TPB.update(itn_train_pbest.data)
            VPB.update(itn_val_pbest.data)            
            gen_data += itn_gen_data

        TPBD = dummy(TPB)
        VPBD = dummy(VPB)

        utils.log_print(f"Sizes train/val/gen : {(len(TPBD.data),len(VPBD.data),len(gen_data))}", args)
        
        magg_net = run_train_ep(
            domain, magg_net, TPBD, VPBD, gen_data, target_data,
        )
    
    
    elif args.magg_train_mode == 'seq':
        for itn in magg_itns:
            train_pbest, val_pbest, gen_data = magg_load_data(domain, itn)

            utils.log_print(f"Training MAGG on {itn}", args)
            
            magg_net = run_train_ep(
                domain, magg_net, train_pbest, val_pbest, gen_data, target_data,
            )

    
    
    utils.save_model(magg_net.state_dict(), f"{args.outpath}/{args.exp_name}/mean_agg_net.pt")

def run_train_ep(domain, magg_net, train_pbest, val_pbest, gen_data, target_data):

    args = domain.args
    
    path = f'{args.outpath}/{args.exp_name}/train_out'        
    
    epochs = args.epochs

    train_gen = DataGen(
        domain,
        train_pbest,
        target_data.get_train_vinput(),
        gen_data
    )

    val_gen = DataGen(
        domain,
        val_pbest,
        target_data.get_train_vinput(),
        None
    )
    
    opt = optim.Adam(
        magg_net.parameters(),
        lr=args.lr
    )
    best_test_metric = 100.
    
    patience = args.infer_patience
    num_worse = 0
    eval_count = 0

    for epoch in range(epochs):
        start = time.time()
        train_losses = []
        val_losses = []
        magg_net.train()
        
        for batch in train_gen.train_iter():
            loss, _ = magg_net.model_train_batch(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())            

        magg_net.eval()
        with torch.no_grad():
            for batch in val_gen.train_iter():
                loss, _ = magg_net.model_train_batch(batch)
                val_losses.append(loss.item())

        eval_res = {
            'train_loss': torch.tensor(train_losses).float().mean().item(),
            'val_loss': torch.tensor(val_losses).float().mean().item(),
            'nc': 1.0
        }

        results = utils.print_results(
            WS_TRAIN_LOG_INFO,
            eval_res,
            args,
            ret_early=True
        )

        METRIC = eval_res['val_loss']
            
        if METRIC >= best_test_metric:
            num_worse += 1
        else:
            num_worse = 0
            best_test_metric = METRIC
            utils.save_model(magg_net.state_dict(), f"{path}/magg_best_dict.pt")

        if num_worse >= patience:            
            magg_net.load_state_dict(torch.load(f"{path}/magg_best_dict.pt"))
            break

        end = time.time()
        utils.log_print(
            f"Epoch {epoch}/{epochs} => Train / Val : {round(eval_res['train_loss'], 3)} / {round(eval_res['val_loss'], 3)} "
            f"| {end-start}"
            ,args
        )
                
    return magg_net
