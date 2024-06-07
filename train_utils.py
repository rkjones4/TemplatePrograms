import sys, os, torch, json, time, utils
import math
from tqdm import tqdm

def model_train_batch(batch, net, opt):

    loss, br = net.model_train_batch(batch)
               
    if opt is not None:
        if net.acc_count == 0:
            opt.zero_grad()            

        aloss = loss / (net.acc_period * 1.)
        aloss.backward()
        net.acc_count += 1
        
        if net.acc_count == net.acc_period:
            opt.step()
            net.acc_count = 0
        
    return br

def model_train(loader, net, opt):
    
    if opt is None:
        net.eval()
        log_period = 1e20
    else:
        net.train()
        if 'log_period' in net.__dict__:
            log_period = net.acc_period * net.log_period
        else:
            log_period = 1e20
    
    ep_result = {}
    bc = 0.

    for batch in loader:

        bc += 1.

        if bc > log_period:
            break

        if isinstance(batch, dict):
            bk = 'vdata'
        else:
            bk = 0

        if 'iter_num' in loader.__dict__:
            loader.iter_num += batch[bk].shape[0]
        
        batch_result = model_train_batch(batch, net, opt)
        for key in batch_result:                        
            if key not in ep_result:                    
                ep_result[key] = batch_result[key]
            else:
                ep_result[key] += batch_result[key]


    ep_result['batch_count'] = bc
    
    return ep_result


def run_train_epoch(
    args,
    res,
    net,
    opt,
    train_loader,
    val_loader,
    LOG_INFO,
    do_print,
    epoch = None
):
    
    json.dump(res, open(f"{args.outpath}/{args.exp_name}/res.json" ,'w'))

    t = time.time()

    if epoch is None:
        itn = train_loader.iter_num
        if do_print:
            utils.log_print(f"\nBatch Iter {itn}:", args)
    else:
        itn = epoch
        if do_print:
            utils.log_print(f"\nEpoch {itn}:", args)
        

    if train_loader is not None:
        train_loader.mode = 'train'

    if val_loader is not None:
        val_loader.mode = 'train'

    train_result = model_train(
        train_loader,
        net,
        opt
    )
    if epoch is None:
        train_itn = train_loader.iter_num
        slice_name = 'iters'
    else:
        train_itn = epoch
        slice_name = 'epochs'
        
    utils.update_res(
        LOG_INFO,
        res['train_plots']['train'],
        train_result,
        slice_name,
        train_itn
    )    
    
    if do_print:            
        
        with torch.no_grad():
            val_result = model_train(
                val_loader,
                net,
                None,
            )

        utils.update_res(
            LOG_INFO,
            res['train_plots']['val'],
            val_result,
            slice_name,
            train_itn,
        )    
                        
        utils.log_print(
            f"Train results: ", args
        )
            
        utils.print_results(
            LOG_INFO,
            train_result,
            args,
        )

        utils.log_print(
            f"Val results: ", args
        )
            
        utils.print_results(
            LOG_INFO,
            val_result,
            args,
        )
                 
        utils.make_info_plots(
            LOG_INFO,
            res['train_plots'],
            slice_name,
            'train',
            args,
        )
            
        utils.log_print(
            f"    Time = {time.time() - t}",
            args
        )

def run_eval_epoch(
    args,
    res,
    net,
    eval_data,
    EVAL_LOG_INFO,
    itn
):
                
    with torch.no_grad():
        
        net.eval()        
                    
        t = time.time()                

        eval_results = {}
        for key, loader in eval_data:

            if loader.mode == 'train':
                loader.mode = 'eval'

            net.vis_mode = (key, itn)
            net.init_vis_logic()
            
            eval_results[key] = model_eval(
                args,
                loader,
                net,
            )
            
            net.save_vis_logic()
            
            utils.log_print(
                f"Evaluation {key} set results:",
                args
            )

            utils.print_results(
                EVAL_LOG_INFO,
                eval_results[key],
                args
            )
                        
        utils.log_print(f"Eval Time = {time.time() - t}", args)

        res['eval_iters'].append(itn)
                
        utils.make_comp_plots(
            EVAL_LOG_INFO,
            eval_results,            
            res['eval_plots'],
            res['eval_iters'],
            args,
            'eval'
        )



def check_early_stop(res, args, obj_dir):
    eps = res['eval_iters']
    if 'val' not in res['eval_plots'] or \
       args.es_metric not in res['eval_plots']['val']:
        utils.log_print("!! SKIPPING EARLY STOP !!", args)
        return -1
    
    metric_res = torch.tensor(res['eval_plots']['val'][args.es_metric])
    cur_ep = eps[-1]
    
    for i, ep in enumerate(eps[:metric_res.shape[0]]):
        if cur_ep - ep <= args.es_patience:
            metric_res[i] -= args.es_threshold

    if obj_dir == 'high':
        best_ep_ind = metric_res.argmax().item()
    elif obj_dir == 'low':
        best_ep_ind = metric_res.argmin().item()
    else:
        assert False
        
    best_ep = eps[best_ep_ind]

    # early stopping logic
    
    if cur_ep - best_ep > args.es_patience:
        utils.log_print(
            f"Stopping early at epoch {cur_ep}, "
            f"choosing iter {best_ep} with val {args.es_metric} " 
            f"of {metric_res[best_ep_ind].item()}",
            args
        )
        utils.log_print(
            f"Final test value for {args.es_metric} : {res['eval_plots']['test'][args.es_metric][best_ep_ind]}",
            args
        )
        return best_ep

    return -1


def model_eval(
    args,
    loader,
    net,
):

    res = {}

    pbar = tqdm(total=math.ceil(loader.eval_size / loader.eval_batch_size))    
    
    for count, batch in enumerate(loader):

        _res = net.model_eval_fn(
            batch
        )

        for k,v in _res.items():
            if k not in res:
                res[k] = v
            else:
                res[k] += v

        pbar.update(1)
                        
    res['count'] = count + 1
    res['nc'] = 1
    pbar.close()
    
    return res
