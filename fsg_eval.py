import os
from tqdm import tqdm
import utils
from utils import device
import torch
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import search

SAMPLE_TIMEOUT = 60

FSGE_ARGS = [    
    ('-tbm', '--temp_beams', 5, int), # 40 -> change for expensive inference
    ('-ebm', '--exp_beams', 5, int), # 10 -> change for expensive inference
]

def load_gen_net(domain):

    print(f"Loading gen model from {domain.args.load_gen_model_path}")

    gen_net = domain.load_gen_model(
        domain.args.load_gen_model_path
    )
    gen_net.model_name = 'magg'
        
    gen_net.eval()
    gen_net.to(device)
        
    return gen_net
           
    
def build_sample_fn(inf_net, gen_net, domain, args):

    ex = domain.executor

    def sample_fn(key, vdata, num_gen, extra=None):
        
        if domain.name == 'shape' and args.vin_type == 'voxel':
            inp_group = torch.stack([
                domain.executor.conv_scene_to_vinput(vd) for vd in vdata
            ],dim=0).unsqueeze(0).float().to(inf_net.device)
        else:
            inp_group = vdata.unsqueeze(0).to(inf_net.device)
            
        eval_info, _eval_res = search.split_beam_search(
            inf_net,
            {
                'vdata': inp_group,
                'extra_gt_data': extra
            },
            args.temp_beams,
            args.exp_beams,
        )
        
        degen = False
        for _prt in eval_info['info']:
            if len(_prt) == 0:
                degen = True
                break

        if degen:
            print(f"Returned null inference for {key}")
            return None, None, None

        if 'struct' not in eval_info['info'][0][0]:            
            print("Bad inference")
            print(eval_info['info'])
            return None, None, None

        for k,v in _eval_res.items():
            if k not in domain.eval_res:
                domain.eval_res[k] = 0.
            domain.eval_res[k] += v

        recon_progs = None
        recon_mvals = None

        try:
            recon_mvals = eval_info['mvals'][0]
            recon_progs = [e['expr'] for e in eval_info['info'][0]]
        except Exception as e:
            print(f"Failed to save info with {e}")
            pass
        
        try:
            recons = eval_info['execs'][0]
        except:
            recons = None
            
        struct_tokens = eval_info['info'][0][0]['struct']
        struct_tokens = [
            ex.HOLE_TOKEN if \
            (ex.STRUCT_LOC_TOKEN in t or ex.PARAM_LOC_TOKEN in t) \
            else t\
            for t in struct_tokens
        ]

        struct_prog = domain.executor.TLang.tokens_to_tensor(
            struct_tokens
        )
        
        seq = torch.zeros(1, gen_net.struct_net.ms, device=device).long()
        seq[0,:struct_prog.shape[0]] = struct_prog
        seq_lens = torch.tensor([struct_prog.shape[0]]).long()
        
        gens = []
        gen_progs = []

        seen = set()
        
        seq = torch.cat((struct_prog.to(device), torch.zeros(1).long().to(device)),dim=0)

        t = time.time()

        bcodes = gen_net.get_seed_codes(
            inp_group, num_gen
        )
        
        while len(gens) < num_gen:

            if (time.time() - t) > SAMPLE_TIMEOUT:
                print(f"only found {len(gens)} gens")
                break                        
            
            _gens = gen_net.sample_gen_from_seed(
                bcodes,
                seq,
                num_gen
            )
            
            for pixels, tokens in _gens:

                sig = tuple(tokens)
                if sig in seen:
                    continue

                seen.add(sig)
                
                if len(pixels.shape) == 2:
                    pixels = pixels.unsqueeze(-1)
                
                gens.append(pixels.cpu())
                gen_progs.append(tokens)


        ret_info = {
            'struct_prog': struct_prog,
            'recon_progs': recon_progs,
            'recon_mvals': recon_mvals,
            'recons': recons
        }
                        
        return (
            gens[:num_gen],
            gen_progs[:num_gen],
            ret_info
        )
                    
    return sample_fn
    
def run_fs_gen_eval(domain, sample_fn, target_data):
    
    args = domain.args

    gen_info = {}

    infer_info = {}
    
    for tname, tinfo in tqdm(list(target_data.fsg_tasks.items())):

        infer_info[tname] = {
            'metrics': [],
            'recon_progs': [],
            'recon_mvals': []
        }
        
        prompts = tinfo['prompts']
        targets = tinfo['targets']

        gen_vdata = []
        
        for i, prompt in enumerate(prompts):
                        
            vkg = torch.tensor(prompt).long()
            seed_vdata = target_data.vinput[vkg]
            extra_gt_data = target_data.get_extra_gt_data(vkg)

            
            try:
                gen, gen_progs, extra_info = sample_fn(
                    vkg,
                    seed_vdata,                    
                    args.fsg_gens_per_prompt,
                    extra_gt_data,
                )
            except Exception as e:
                print(f"Failed to gen for {tname} {i} with {e}")
                gen = None
                
            if gen is None:
                continue

            if domain.name == 'shape':
                gen = [domain.executor.conv_scene_to_vinput(g.squeeze()).float().to(g.device) for g in gen]
                
                extra_info['recons'] = [
                    domain.executor.conv_scene_to_vinput(r.squeeze()).float().to(r.device) for r in extra_info['recons']
                ]
            
            gen_vdata += gen
            gen_info[f'{tname}_{i}'] = gen_progs
            
            infer_info[tname]['recon_progs'].append(extra_info['recon_progs'])
            infer_info[tname]['recon_mvals'].append(extra_info['recon_mvals'])
            
            if i < args.num_write:

                genstack = torch.stack(gen,dim=0)

                if domain.name == 'shape':
                    seed_vdata = torch.stack(
                        [domain.executor.conv_scene_to_vinput(s).float().to(seed_vdata.device) for s in seed_vdata], dim=0
                    )                        
                                    
                if len(genstack.shape) == len(seed_vdata.shape) + 1:
                    genstack = genstack.squeeze(-1)
                    
                recstack = torch.stack(extra_info['recons'],dim=0).unsqueeze(-1).to(seed_vdata.device)

                if len(recstack.shape) == len(seed_vdata.shape) + 1:
                    recstack = recstack.squeeze(-1)
                    
                comb = torch.cat((
                    seed_vdata,
                    recstack,
                    genstack
                ), dim=0)
                    
                domain.executor.render_group(
                    comb,
                    f'{args.outpath}/{args.exp_name}/vis/fsg_{tname}_ind{i}',
                    rows = (comb.shape[0] // args.max_vis_inputs)
                )                                                                                            
            
        if len(gen_vdata) == 0:
            print(f"No results for {tname}")
            continue

    torch.save(gen_info, f'{args.outpath}/{args.exp_name}/gen_info.pt')    
    torch.save(infer_info, f'{args.outpath}/{args.exp_name}/infer_info.pt')
    
    
def fsg_eval(domain):
    
    args = domain.get_ft_args(FSGE_ARGS)
    args.eval_batch_size = 1
    utils.init_pretrain_run(args)
    
    print(f"Loading inf net from {domain.args.load_model_path}")
    inf_net = domain.load_pretrained_net()
    inf_net.eval()
    inf_net.to(device)        
    gen_net = load_gen_net(domain)
        
    target_data = domain.load_real_data(mode='fsg')
    
    with torch.no_grad():

        sample_fn = build_sample_fn(
            inf_net,
            gen_net,
            domain,
            args
        )
        domain.eval_res = {}

        run_fs_gen_eval(
            domain,
            sample_fn,
            target_data,
        )
        
        res = utils.print_results(
            domain.EVAL_LOG_INFO,
            domain.eval_res,
            args,
            ret_early=True
        )

        utils.log_print(f'Inference Results:', args)
        for k,v in res.items():
            rv = round(v,3)
            utils.log_print(f"    {k}: {rv}", args)
