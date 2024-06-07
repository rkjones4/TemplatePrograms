import torch
from torch import optim
import numpy as np
import time
import utils
from utils import device
import os
from tqdm import tqdm
import random

# fine-tuning logic

class DataGen:
    def __init__(
        self,
        domain,
        pbest,
        target_vinput,
        gen_data,
    ):

        args = domain.args
        ex = domain.executor

        self.args = args
        self.domain = domain
        self.batch_size = args.batch_size
                
        self.target_vinput = target_vinput
                
        self.keys = []
        self.data = []

        
        self.max_vis_inputs = args.max_vis_inputs
            
        for keys, (_, d) in pbest.data.items():
            
            self.keys.append(keys)
            self.data.append(d)
            
        if gen_data is not None:
            self.gen_data = gen_data
        else:
            self.gen_data = []

        self.st_weight = self.args.st_weight
        self.lest_weight = self.args.lest_weight
        self.ws_weight = self.args.ws_weight

        if self.ws_weight >= 1.0 and len(self.gen_data) == 0:
            print("WS weight is 1.0 but no gen data, defaulting to LEST_ST")
            self.ws_weight = 0.
            self.lest_weight = .5
            self.st_weight = .5
            
        self.train_size = len(self.keys) + len(self.gen_data)

        with torch.no_grad():
            self.preload_data()

    def preload_data(self):

        self.lest_data = {}
        self.st_data = {}
        self.ws_data = {}            
        
        if self.lest_weight > 0.:
            print("Pre loading LEST data")
            self.preload_mode(
                self.lest_data,
                self.data,
                None,
                None
            )
            
        if self.ws_weight > 0:
            print("Pre loading WS data")
            self.preload_mode(
                self.ws_data,
                self.gen_data,
                None,
                None
            )
        else:
            self.gen_data = []
            
        if self.st_weight > 0.:
            print("Pre loading ST data")
            self.preload_mode(
                self.st_data,
                self.data,
                self.keys,
                self.target_vinput
            )

        if self.lest_weight <= 0. and self.st_weight <= 0.:
            self.data = []

    def preload_mode(self, sdata, idata, ikeys, vdata):
        ex = self.domain.executor

        for ind in tqdm(list(range(len(idata)))):
            d = [idata[ind]]

            b = ex.make_batch(d, self.args)
            
            for k,v in b.items():
                if k not in sdata:
                    sdata[k] = []
                sdata[k].append(v[0])

        sdata.update({
            k:torch.stack(V,dim=0) for k,V in sdata.items()
        })

        if vdata is None:
            return
                
        sdata['vdata'] = self.domain.make_blank_visual_batch(
            batch_size=len(ikeys),
            group_size=self.max_vis_inputs,
            device=torch.device('cpu')
        )        

        if self.domain.name == 'shape':
            vdata = [
                self.domain.executor.conv_scene_to_vinput(vd) for vd in vdata
            ]
        
        for i,ik in tqdm(list(enumerate(ikeys))):
            if self.max_vis_inputs is None:
                t_ind = ik
                pixels = vdata[t_ind]
                try:
                    sdata['vdata'][i] = pixels.cpu()
                except:
                    assert len(pixels.shape) == 2
                    sdata['vdata'][i,:,:,0] = pixels.cpu()
                    
            else:                
                for j,t_ind in enumerate(ik):                    
                    pixels = vdata[t_ind]
                    try:
                        sdata['vdata'][i,j] = pixels.cpu()
                    except:
                        assert len(pixels.shape) == 2
                        sdata['vdata'][i,j] = pixels.cpu()
            
    def sample_plad_mode(self):
        comb_modes = ['lest', 'st', 'ws']
        
        comb_weights = [self.lest_weight, self.st_weight, self.ws_weight]

        return np.random.choice(
            comb_modes,
            p = comb_weights
        )
        
    def train_iter(self):
        tar_inds = list(range(len(self.data)))
        random.shuffle(tar_inds)

        gen_inds = list(range(len(self.gen_data)))
        random.shuffle(gen_inds)
        
        while len(tar_inds) > 0 or len(gen_inds) > 0:
            
            pmode = self.sample_plad_mode()

            if pmode == 'ws':
                if len(gen_inds) <= 0:
                    continue
                else:
                    binds = torch.tensor(gen_inds[:self.batch_size])
                    gen_inds = gen_inds[self.batch_size:]
                    yield from self.mode_batch(
                        self.ws_data,
                        binds
                    )
            
            elif pmode in ('st', 'lest'):
                if len(tar_inds) == 0:
                    continue
                else:                    
                    binds = torch.tensor(tar_inds[:self.batch_size])
                    tar_inds = tar_inds[self.batch_size:]            
                
                    if pmode == 'lest':
                        yield from self.mode_batch(
                            self.lest_data,
                            binds
                        )
                        
                    elif pmode == 'st':
                        yield from self.mode_batch(
                            self.st_data,
                            binds
                        )
                    

    def mode_batch(self, data, binds):
        batch = {
            k: V[binds].to(device) for k,V in data.items()
        }

        yield batch
        
def train_rec(
    domain, net, gen_data, target_data, pbest
):
    
    args = domain.args

    path = args.infer_path
    
    epochs = args.epochs

    train_gen = DataGen(
        domain,
        pbest,
        target_data.get_train_vinput(),
        gen_data
    )

    val_gen = target_data.val_eval_iter

    opt = optim.Adam(
        net.parameters(),
        lr=args.lr
    )

    best_test_metric = domain.init_metric_val()

    utils.save_model(net.state_dict(), f"{path}/best_dict.pt")

    patience = args.infer_patience
    num_worse = 0
    eval_count = 0
        
    for epoch in range(epochs):
        start = time.time()
        losses = []
        net.train()
        
        for batch in train_gen.train_iter():

            loss, _ = net.model_train_batch(batch)
                                        
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())            
            
        eval_count += 1
        
        if (eval_count % args.eval_per) != 0:
            num_worse += 1
            end = time.time()
            utils.log_print(
                f"Epoch {epoch}/{epochs} => TRAIN ONLY "
                f"|  LOSS : {round(torch.tensor(losses).mean().item(), 3)} | {end-start}"
                , args
            )
            continue        
                        
        net.eval()
        eval_res = {}
        
        with torch.no_grad():
            for batch in val_gen():
                keys = batch['bkeys']
                vinput = batch['vinput']
                extra_gt_data = batch['extra_gt_data']
                try:
                    _eval_res = net.model_eval_fn(
                        {
                            'vdata': vinput,
                            'extra_gt_data': extra_gt_data
                        },
                        args.es_beams,
                    )

                except Exception as e:
                    utils.log_print(
                        f"Failed train infer for {keys} with {e}", args
                    )                
                    continue
                
                for k,v in _eval_res.items():
                    if k not in eval_res:
                        eval_res[k] = 0.
                    eval_res[k] += v
                
        results = utils.print_results(
            domain.EVAL_LOG_INFO,
            eval_res,
            args,
            ret_early=True
        )
        
        ## EVAL

        METRIC = results[domain.obj_name]
        if 'Errors' in results:
            ERR = results['Errors']
        else:
            ERR = -1

        # Always save network, if we improved the metric
        if domain.should_save(METRIC, best_test_metric, 0.0):
            utils.save_model(net.state_dict(), f"{path}/best_dict.pt")

        # Only reset count if we pass the threshold
        if not domain.should_save(METRIC, best_test_metric, args.threshold):
            num_worse += 1
        else:
            num_worse = 0
            best_test_metric = METRIC
                                
        end = time.time()
        utils.log_print(
            f"Epoch {epoch}/{epochs} => Obj[Err] : {round(METRIC, 3)}[{round(ERR,2)}] "
            f"|  LOSS : {round(torch.tensor(losses).mean().item(), 3)} | {end-start}"
            ,args
        )

        # early stopping on validation set 
        if num_worse >= patience:
            # load the best model and stop training
            utils.log_print("Early stopping inner loop", args)
            net.load_state_dict(torch.load(f"{path}/best_dict.pt"))
            return epoch + 1

    return epochs
