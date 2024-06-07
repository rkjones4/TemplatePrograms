import torch
from torch import optim
import numpy as np
import time
import utils
from utils import device
import os
from tqdm import tqdm
import random

# Logic to train wake sleep model

WS_TRAIN_LOG_INFO = [
    ('Train Loss', 'train_loss', 'nc'),
    ('Val Loss', 'val_loss', 'nc'),    
]

class WSDataGen:
    def __init__(
        self,
        domain,
        pbest,
    ):

        args = domain.args
        ex = domain.executor

        self.args = args
        self.domain = domain
        self.batch_size = args.batch_size
        
        self.keys = []
        self.data = []

        self.max_vis_inputs = args.max_vis_inputs
            
        for keys, (_, d) in pbest.data.items():
            
            self.keys.append(keys)
            self.data.append(d)
        
        self.train_size = len(self.keys)

        ex = self.domain.executor
        with torch.no_grad():
            self.ex_data = ex.make_batch(self.data, self.args)
        
    def train_iter(self):
        inds = torch.randperm(len(self.data))
        
        while len(inds) > 0:
            binds = inds[:self.batch_size]
            inds = inds[self.batch_size:]

            with torch.no_grad():
                g_batch = {
                    k: v[binds].to(self.domain.device) for k,v in
                    self.ex_data.items()
                }

            yield g_batch


def make_ws_gens(
    domain, gen_model, train_pbest, val_pbest
):

    gen_model, ge = train_gen_model(
        domain, gen_model, train_pbest, val_pbest
    )
    
    with torch.no_grad():
        print("Sampling gen model")
        gen_data = sample_ws_gens(domain, gen_model)

        group = [g for g in gen_data[:10]]
        images = []
        for g in group:
            images += [i for i in g.get_images()]
        num_rows = 5

        try:
            gen_model.ex.render_group(
                images,
                name=f'{domain.args.ws_save_path}/drm_render_{gen_model.gen_epoch}',
                rows=num_rows
            )
        except Exception as e:
            utils.log_print("Failed to save dream images with {e}", domain.args)
                
    gen_model.gen_epoch += 1
                              
    return gen_model, gen_data, ge

def sample_ws_gens(domain, gen_model):

    gen_model.seen_group_sigs = set()
    gen_model.seen_structs = set()
    
    gen_data = []
    
    pbar= tqdm(total=domain.args.ws_train_size)
    res = {}

    gen_model.grace_num = domain.args.ws_grace_num
    
    while len(gen_data) < domain.args.ws_train_size:                
        try:
            samples = gen_model.sample_gens(
                domain.args.gen_beam_size,
            )
        except Exception as e:
            utils.log_print(f"FAILED WAKE SLEEP batch with {e}", domain.args)
            continue
        
        gen_data += samples
        pbar.update(len(samples))
        
    pbar.close()
    
    return gen_data
    
        
# returns gen_model, gen_data, ge
def train_gen_model(
    domain, gen_model, train_pbest, val_pbest
):
    
    args = domain.args

    path = args.ws_save_path
    
    epochs = args.epochs

    train_gen = WSDataGen(
        domain,
        train_pbest,
    )

    val_gen = WSDataGen(
        domain,
        val_pbest
    )

    opt = optim.Adam(
        gen_model.parameters(),
        lr=args.lr
    )

    # metric is loss
    best_test_metric = 100.

    utils.save_model(gen_model.state_dict(), f"{path}/best_dict.pt")

    patience = args.infer_patience
    num_worse = 0
        
    for epoch in range(epochs):
        start = time.time()
        train_losses = []
        val_losses = []
        
        gen_model.train()
        
        for batch in train_gen.train_iter():
            
            loss, _ = gen_model.model_train_batch(batch)
                                        
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_losses.append(loss.item())            
        
        gen_model.eval()
        with torch.no_grad():
            for batch in val_gen.train_iter():
                loss, _ = gen_model.model_train_batch(batch)
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
        
        ## EVAL

        METRIC = eval_res['val_loss']
            
        if METRIC >= best_test_metric:
            num_worse += 1
        else:
            num_worse = 0
            best_test_metric = METRIC
            utils.save_model(gen_model.state_dict(), f"{path}/best_dict.pt")

        # early stopping on validation set 
        if num_worse >= patience:
            # load the best model and stop training
            gen_model.load_state_dict(torch.load(f"{path}/best_dict.pt"))
            break

        end = time.time()
        utils.log_print(
            f"Epoch {epoch}/{epochs} => Train / Val : {round(eval_res['train_loss'], 3)} / {round(eval_res['val_loss'], 3)} "
            f"| {end-start}"
            ,args
        )
        
    return gen_model, epochs
