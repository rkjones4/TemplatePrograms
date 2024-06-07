import dill
import utils
import torch
from prob_infer import infer_prob_programs
import json
from copy import deepcopy
import train_prob_plad
import wake_sleep

class BestPrograms:
    def __init__(self, domain, name):
        self.domain = domain
        self.data = {}
        self.name = name
        
        IMV = domain.init_metric_val()
        
    def update(self, key, infos, execs, mvals):
        key = tuple(key.tolist())
                
        score = torch.tensor(mvals).mean().item()
        
        if key in self.data and (not self.domain.comp_metric(
            score, self.data[key][0]
        )):
            return

        try:
            infd = self.domain.executor.make_infer_data(infos, self.domain.args)  
        except Exception as e:            
            utils.log_print(f"Failed BP update for {key} with {e}", self.domain.args)
            return
                
        self.data[key] = (score, infd)

    def record(self):
        args = self.domain.args
        utils.log_print("Recording best programs", args)
        dill.dump(
            self.data,
            open(
                f'model_output/{args.exp_name}/best_progs_{self.name}.dl',
                "wb"
            )
        )
        
class Logger:
    def __init__(self, domain):
                
        self.res = {
            'train': {},
            'val': {},
            'test': {},        
        }

        self.Round = 0
        
        self.inf_epochs = [0]
        self.gen_epochs = [0]
        
        self.best_val = domain.init_metric_val()
        self.best_epoch = 0
        self.domain = domain

    def log(self, iter_res, net):
        for sname, svals in iter_res.items():
            for mname, mval in svals.items():
                if mname not in self.res[sname]:
                    self.res[sname][mname] = []
                self.res[sname][mname].append(mval)

        json.dump(
            {**self.res, **{'epochs':self.inf_epochs, 'gen_epochs':self.gen_epochs}},
            open(f"model_output/{self.domain.args.exp_name}/res.json" ,'w')
        )
        
        utils.make_joint_plots(
            self.res, self.inf_epochs, self.domain.args
        )        

        if self.domain.should_save(iter_res['val']['Obj'], self.best_val, self.domain.args.threshold):
            utils.log_print("Replacing best model", self.domain.args)
            self.best_val = iter_res['val']['Obj']
            self.best_epoch = self.inf_epochs[-1]                    
            utils.save_model(net.state_dict(), f"model_output/{self.domain.args.exp_name}/inf_net.pt")

        try:
            utils.save_model(net.state_dict(), f"model_output/{self.domain.args.exp_name}/train_out/ep_{self.inf_epochs[-1]}_inf_net.pt")
        except Exception as e:
            print("Failed to save model version")
            
    def check_early_stop(self):
        if self.inf_epochs[-1] >= self.domain.args.max_iters:
            return True
        utils.log_print(f"ROUND {self.Round} (Inf Epochs: {self.inf_epochs[-1]})", self.domain.args)
            
    def add_epochs(self, ie, ge):
        self.inf_epochs.append(ie + self.inf_epochs[-1])
        self.gen_epochs.append(ge + self.gen_epochs[-1])
        self.Round += 1

def magg_record_data(
    path,
    iter_num,
    gen_data,
    train_pbest,
    val_pbest
):
    save_path = f'{path}/magg_tdata_{iter_num}.pt'

    print(f"Saving MAGG TRAIN DATA to {save_path}")
    
    R = {}

    for name, dset in [
        ('train', train_pbest),
        ('val', val_pbest),
    ]:
        R[name] = {'keys': [], 'infos': []}
        
        for keys, (_, d) in dset.data.items():
            R[name]['keys'].append(keys)
            R[name]['infos'].append(d.infos)

    try:
        R['gen_infos'] = []
        if gen_data is not None:
            for d in gen_data:
                R['gen_infos'].append(d.infos)
    except Exception as e:
        print(f"Failed to save WS for MAGG DATA with {e}")
                
    torch.save(R, save_path)
        
# Fine-tune a recognition network towards a domain of interest
def fine_tune(domain):

    # Load args, rec net, target distribution of real_data
    args = domain.get_ft_args()
    net = domain.load_pretrained_net()    
    target_data = domain.load_real_data()
    
    # If doing WS, create a generative model
    if 'WS' in args.ft_mode:
        print("Loading Gen Model")
        gen_model = domain.load_gen_model(args.load_gen_model_path)                        
    else:
        gen_model = None
                
    train_pbest = BestPrograms(domain, 'train')
    val_pbest = BestPrograms(domain, 'val')
    logger = Logger(domain)
    
    while True:
        if logger.check_early_stop(): break

        utils.log_print("Dynamic resampling of keys and pbest", args)

        target_data.sample_dyn_keys()
        train_pbest.data = {}
        val_pbest.data = {}
        
        net.iter_num = logger.inf_epochs[-1]
        # Run Inf Net over real_data to update best_prog data structure
        with torch.no_grad():
            
            iter_res = infer_prob_programs(
                domain, net, target_data, train_pbest, val_pbest
            )
        
        # Record inference output
        utils.save_model(
            net.state_dict(),
            f"model_output/{domain.args.exp_name}/last_ckpt.pt"
        )

        # Plotting / eval metric logic
        logger.log(iter_res, net)
        
        # Stop early based on val metric
        if logger.inf_epochs[-1] - logger.best_epoch > args.iter_patience:
            utils.log_print("Stopping early", args)
            break                    
            
        if gen_model is not None:

            utils.log_print("Training gen model", args)
            # next gen model, training data from gen, number of gen epochs
            gen_model, gen_data, ge = wake_sleep.make_ws_gens(
                domain, gen_model, train_pbest, val_pbest
            )
            utils.save_model(
                gen_model.state_dict(),
                f'model_output/{domain.args.exp_name}/gen_model.pt'
            )            
        else:
            ge = 0
            gen_data = None
            
        ie = train_prob_plad.train_rec(
            domain,
            net,
            gen_data,            
            target_data,
            train_pbest,
        )

        logger.add_epochs(ie, ge)                            
                        
        try:
            magg_record_data(
                args.infer_path,
                net.iter_num,
                gen_data,
                train_pbest,
                val_pbest
            )
        except Exception as e:
            utils.log_print(f"Failed to save magg data with {e}", args)        
