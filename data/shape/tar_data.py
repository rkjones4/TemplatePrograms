from tqdm import tqdm
import torch, random, os
import json
import sys
sys.path.append('domains')
import dom_common as dc
from copy import deepcopy

def prog_to_scene(prog, mp):
    scene = torch.zeros(mp, 6)
    for i,prm in enumerate(list(prog.split(':')[1].split('+'))):
        params = prm.split(',')        
        assert len(params) == 6
        params = [float(p) for p in params]
        scene[i] = torch.tensor(params)

    return scene


def preload_data(path, max_prims):
    progs = {}
    print("Preloading data")

    with open(path) as f:
        for line in tqdm(f):
            cat,ind = line.split(':')[0].split('_')
            scene = prog_to_scene(line, max_prims)
            progs[(cat,int(ind))] = scene
            
    return progs

    
def load_gt_targets(
    data, groups
):

    R = []
    
    for gname, group in tqdm(groups):
        R.append((gname, [data[gi] for gi in group]))
        
    return R

                
class TargetDataset(dc.TargetBase):
    def __init__(self, args, device, executor, mode):

        self.args = args
        self.device=device
        self.mode = mode        

        self.extra_gt_data = None
        self.vinput = []

        self.sem_data = []

        self.train_keys = []
        self.val_keys = []
        self.test_keys = []

        self.train_group_names = []
        self.val_group_names = []
        self.test_group_names = []

        self.eval_batch_size = 1
        
        self.max_prims = executor.MAX_PRIMS
        
        if self.mode == 'fsg':
            print("Loading FSG Real Data")
            with torch.no_grad():
                self.init_fsg_data(executor)
                
            return

        if self.mode == 'coseg':
            print("Loading Coseg Data")
            with torch.no_grad():
                self.init_coseg_data(executor)
            return
        
        make_target_fn = load_gt_targets
            
        with torch.no_grad():
            load_data = preload_data('data/shape/parsed_progs.txt', self.max_prims)
            
            train_val_groups = torch.load('data/shape/train_groups.pt')
            test_info = torch.load('data/shape/test_groups.pt')

            if len(train_val_groups) < args.train_size + args.eval_size:
                assert False, 'too many groups in train and val'    
                            
            train_groups = [
                (f'train_group_{i}', group)
                for i, group in enumerate(train_val_groups[:args.train_size])
            ]

            val_groups = [
                (f'val_group_{i}', group)
                for i, group in enumerate(train_val_groups[args.train_size:args.train_size + args.eval_size])
            ]
                
            test_groups = []
            for name, info in list(test_info.items())[:args.etest_size]:
                test_groups.append(
                    (name, [(c,i) for c,i,_ in info['fsge'][:args.max_vis_inputs]])
                )
            
            train_data = make_target_fn(
                load_data,
                train_groups,
            )
            
            val_data = make_target_fn(
                load_data,
                val_groups
            )
                        
            test_data = make_target_fn(
                load_data,
                test_groups
            )

            to_load = [
                (train_data, self.train_keys, self.train_group_names),
                (val_data, self.val_keys, self.val_group_names),
                (test_data, self.test_keys, self.test_group_names)
            ]

            
            print("Loading Prob data to vdata")
                
            for grp_set, keys, sg_names in to_load:
                    
                for gname, group in tqdm(grp_set):

                    kg = []

                    for vd in group:             
                        kg.append(len(self.vinput))
                        self.vinput.append(vd)                        
                            
                    sg_names.append(gname)
                    keys.append(torch.tensor(kg).long())
                                            
        self.iter_num = 0
        self.size = args.eval_size            
        self.eval_size = args.eval_size
        
        assert mode != 'fsg'

        self.vinput = torch.stack(self.vinput,dim=0)
        print(f"Key sizes ({self.vinput.shape})")
        
        self.test_keys = torch.stack(self.test_keys, dim=0)            
        self.dyn_train_keys = self.train_keys
        self.dyn_val_keys = self.val_keys
        self.sample_dyn_keys()
        
        print(f"train {self.train_keys.shape}")
        print(f"val {self.val_keys.shape}")        
        print(f"test {self.test_keys.shape}")


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
            
            while len(key_set) < target_size:                
            
                for i in range(len(dyn_key_set)):
                    if len(key_set) >= target_size:
                        break                                

                    inds = torch.randperm(dyn_key_set[i].shape[0])[:mvi]
                    
                    key_set.append(dyn_key_set[i][inds])
                    
        self.train_keys = torch.stack(self.train_keys,dim=0)
        self.val_keys = torch.stack(self.val_keys,dim=0)
        

    def init_coseg_data(self, executor):

        args = self.args
        
        self.vinput = []
        self.sem_data = []
        self.test_keys = []
        self.test_group_names = []

        load_data = preload_data('data/shape/parsed_progs.txt', self.max_prims)

        test_info = torch.load('data/shape/test_groups.pt')
        
        test_data = load_coseg_progs(
            load_data,
            test_info,
            args.etest_size,
            args.max_vis_inputs,
        )

        to_load = [
            (test_data, self.test_keys, self.test_group_names)
        ]
                                   
        for grp_set, keys, sg_names in to_load:                
            for gname, group in grp_set:
                kg = []
                for vd, sem_seg in group:
                    kg.append(len(self.vinput))
                    self.vinput.append(vd)
                    self.sem_data.append(torch.tensor(sem_seg).long())

                sg_names.append(gname)
                keys.append(kg)

        self.test_keys = torch.tensor(self.test_keys).long()
        self.vinput = torch.stack(self.vinput,dim=0)

        print(f"Key sizes ({self.vinput.shape})")
        print(f"test {self.test_keys.shape}")
        
    def init_fsg_data(self, executor):

        args = self.args
        
        assert args.fsg_prompts_per_task * args.max_vis_inputs == args.fsg_prompt_ex_num

        
        load_data = preload_data('data/shape/parsed_progs.txt', self.max_prims)
        test_info = torch.load('data/shape/test_groups.pt')

        test_groups = []
        for name, info in list(test_info.items())[:args.fsg_num_tasks]:
            test_groups.append(
                (name, [(c,i) for c,i,_ in info['fsge']])
            )

        test_data = load_gt_targets(
            load_data,
            test_groups
        )
        
        self.fsg_tasks = {}
        
        for tname, task_data in test_data:                      

            task_inds = []
            for td in task_data:
                task_inds.append(len(self.vinput))
                self.vinput.append(td)
                                
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


def load_coseg_progs(load_data, test_info, num_progs, max_vis_inputs):
        
    data = []

    for name, info in list(test_info.items()):
        if len(data) >= num_progs:
            break

        group = []
        UM = {}

        if len(info['coseg']) == 0:
            continue
        
        assert len(info['coseg']) == max_vis_inputs
                
        for cat, ind, labels in info['coseg'] :

            assert labels is not None
            
            vd = load_data[cat, ind]
            sem_labels = []
            
            for l in labels:
                if l not in UM:
                    UM[l] = len(UM)
                sem_labels.append(UM[l])
            
            group.append((vd, sem_labels))

        data.append((name, group))

    return data
