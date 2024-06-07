from tqdm import tqdm
import torch, random, os
import json
import sys
sys.path.append('domains')
import dom_common as dc
from copy import deepcopy

def load_split(args):

    import data_split as split

    SPLIT_MAP = {'train': set(), 'val': set(), 'test': set()}

    C2I = {}
    for name, att_list in [
        ('train', split.TRAIN_SPLIT),
        ('val', split.VAL_SPLIT),
        ('test', split.TEST_SPLIT),
    ]:
        for at in att_list:
            cat = at.split(':')[0]
            ctype = ':'.join(at.split(':')[1:])

            SPLIT_MAP[name].add(f'{cat}_{ctype}')
            if cat not in C2I:
                C2I[cat] = len(C2I)

    return SPLIT_MAP, C2I

def load_gt_data(args, delim='_'):

    J = json.load(open('data/layout/lay_data.json'))

    R = {}

    all_names, all_tokens, all_infos = J['names'], J['tokens'], J['infos']    
    
    _, C2I = load_split(args)
    
    for name, tokens, info in zip(all_names, all_tokens, all_infos):

        cat = None
        for pcat in C2I:
            if f'{pcat}_' in name:
                assert cat is None
                cat = pcat                

        styp = f'_'.join(name[len(cat)+1:].split('_')[:-1])        
            
        gname = f'{cat}{delim}{styp}'

        if gname not in R:
            R[gname] = []
            
        R[gname].append((tokens, info))

    return R


def make_gt_prog(loc_ex, gtokens, ginfo):
    P = loc_ex.prog_cls(loc_ex)
    P.run(gtokens)
    P.add_sem_info(ginfo)
    return P

    
def load_static_gt_targets(
    data, set_name, num, group_num, mode, loc_ex, args
):

    R = []

    pbar = tqdm(total=num)

    SPLIT_MAP, _ = load_split(args)
    valid_types = SPLIT_MAP[set_name]
    
    while len(R) < num:
        for gname, group_tokens in data.items():

            if gname not in valid_types:
                continue
            
            if len(R) >= num:
                break

            if len(group_tokens) < group_num:
                continue
            
            if group_num > 1:                
                G = []
                while len(G) < group_num:
                    gtokens, ginfo = group_tokens.pop(0)
                    G.append(make_gt_prog(loc_ex, gtokens, ginfo))
                R.append((gname, G))
                                
            else:
                gtokens, ginfo = group_tokens.pop(0)
                R.append((gname, make_gt_prog(loc_ex, gtokens, ginfo)))

            pbar.update(1)
    pbar.close()
                
    assert len(R) == num

    return R

def load_dynamic_gt_targets(
    data, set_name, max_groups, _B, _C, loc_ex, args
):
    
    R = []

    SPLIT_MAP, _ = load_split(args)
    valid_types = SPLIT_MAP[set_name]
    
    for gname, group_tokens in tqdm(list(data.items())):
        if gname not in valid_types:
                continue

        if len(R) >= max_groups:
            break
        
        G = []
        while len(group_tokens) > 0:
            gtokens, ginfo = group_tokens.pop(0)
            G.append(make_gt_prog(loc_ex, gtokens, ginfo))
        R.append((gname, G))

    return R
                
class TargetDataset(dc.TargetBase):
    def __init__(self, args, device, executor, mode):
        
        self.args = args
        self.device=device
        self.mode = mode        

        self.extra_gt_data = []
        self.vinput = []

        self.sem_data = []

        self.train_keys = []
        self.val_keys = []
        self.test_keys = []

        self.train_group_names = []
        self.val_group_names = []
        self.test_group_names = []

        self.eval_batch_size = 1
        
        if self.mode == 'fsg':
            print("Loading FSG Real Data")
            with torch.no_grad():
                self.init_fsg_data(executor)
                
            return

        if mode == 'coseg':
            args.train_size = 1
            args.eval_size = 1
        
        make_target_fn = load_dynamic_gt_targets
            
        with torch.no_grad():
            load_data = load_gt_data(args)
            
            train_data = make_target_fn(
                deepcopy(load_data),
                'train',
                args.train_size,
                args.max_vis_inputs,
                mode,
                executor,
                args
            )

            val_data = make_target_fn(
                deepcopy(load_data),
                'val',
                args.eval_size,
                args.max_vis_inputs,
                mode,
                executor,
                args
            )

            
            test_data = load_static_gt_targets(
                deepcopy(load_data),
                'test',
                args.etest_size,
                args.max_vis_inputs,
                mode,
                executor,
                args
            )

            to_load = [
                (train_data, self.train_keys, self.train_group_names),
                (val_data, self.val_keys, self.val_group_names),
                (test_data, self.test_keys, self.test_group_names)
            ]
            print("Loading data to images")
            
            for grp_set, keys, sg_names in to_load:
                
                for gname, group in tqdm(grp_set):
                    kg = []

                    for d in group:                                                        
                        kg.append(len(self.vinput))
                        self.vinput.append(d.make_image().detach().cpu())
                        self.extra_gt_data.append(d.get_state_sig())
                        self.sem_data.append(d.make_sem_seg().detach().cpu())
                            
                    sg_names.append(gname)
                    keys.append(kg)

        self.iter_num = 0
        self.size = args.eval_size            
        self.eval_size = args.eval_size
        
        assert mode != 'fsg'

        self.vinput = torch.stack(self.vinput,dim=0)
        self.train_keys = torch.tensor(self.train_keys).long()
        self.val_keys = torch.tensor(self.val_keys).long()
        self.test_keys = torch.tensor(self.test_keys).long()
                
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

        print("Sampling dynamic target keys")
        
        args = self.args

        mvi = args.max_vis_inputs
                
        for key_set, target_size, dyn_key_set in [
            (self.train_keys, args.train_size, self.dyn_train_keys),
            (self.val_keys, args.eval_size, self.dyn_val_keys),
        ]:
            ninds = torch.randperm(dyn_key_set.shape[1])
            
            assert len(key_set) <= target_size

            inds = ninds[:mvi]
            ninds = ninds[mvi:]
            
            for i in range(dyn_key_set.shape[0]):
                if len(key_set) >= target_size:
                    break                                

                key_set.append(dyn_key_set[i,inds])
                    
        self.train_keys = torch.stack(self.train_keys,dim=0)
        self.val_keys = torch.stack(self.val_keys,dim=0)
        

    def init_fsg_data(self, executor):

        args = self.args
        
        assert args.fsg_prompts_per_task * args.max_vis_inputs == args.fsg_prompt_ex_num

        load_data = load_gt_data(args)

        test_data = load_static_gt_targets(
            load_data,
            'test',
            args.fsg_num_tasks,
            args.fsg_prompt_ex_num + args.fsg_target_ex_num,
            'fsg',
            executor,
            args
        )                
        
        self.fsg_tasks = {}
        
        for tname, task_data in test_data:
                      
            random.shuffle(task_data)

            task_inds = []
            for td in task_data:
                task_inds.append(len(self.vinput))
                self.vinput.append(td.make_image().detach().cpu())
                self.extra_gt_data.append(td.get_state_sig())
                
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

        print(f"Key sizes ({self.vinput.shape[0]})")
        print(f"prompt {self.fsg_prompt_keys.shape[0]}")

