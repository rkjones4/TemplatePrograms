import sys
sys.path.append('models')
import model_utils as mu
from utils import device
import utils
import torch
import cv2
import random
import numpy  as np

DOM_PROB_TRAIN_LOG_INFO = [
    ('Loss', 'loss', 'batch_count'),
    ('Struct Loss', 'struct_loss', 'batch_count'),
    ('Deriv Loss', 'deriv_loss', 'batch_count'),
    ('Param Loss', 'param_loss', 'batch_count'),
    ('Struct Acc', 'struct_corr', 'struct_total'),
    ('Deriv Acc', 'deriv_corr', 'deriv_total'),    
    ('Param Acc', 'param_corr', 'param_total')
]

DOM_PROB_EVAL_LOG_INFO = [    
    ('Obj', 'mval', 'mval_cnt'),    
    ('Prog Len', 'prog_len', 'rec_cnt'),        
    ('GPP Len', 'gpp_len', 'gpp_cnt'),
    ('DV Len', 'dv_len', 'dv_cnt'),
    ('PRM Len', 'prm_len', 'prm_cnt'),
    ('NH Avg', 'nh_amt', 'gpp_cnt'),
    ('Static %', 'static_cnt', 'gpp_cnt'),
    ('Shared %', 'shared_cnt', 'gpp_cnt'),    
    ('Errors', 'errs', 'gpp_cnt'),
]

DOM_CMN_ARGS = [

    ('-en', '--exp_name', None,  str),
    
    ('-rd', '--rd_seed', 42,  int),
    ('-o', '--outpath', 'model_output',  str),
    ('-dp', '--dropout', .1, float),
    ('-lr', '--lr', 0.0001,  float),
                
    ('-nl', '--num_layers', 8, int),
    ('-nh', '--num_heads', 16, int),
    ('-hd', '--hidden_dim', 256, int),    
    
    ('-lmp', '--load_model_path', None, str),
    ('-lgmp', '--load_gen_model_path', None, str),
    ('-lrp', '--load_res_path', None, str),            
    
    ('-logp', '--log_period', 10, int),    
    
    ('-mvi', '--max_vis_inputs', 5, int),

    ('-accp', '--acc_period', 1, int),
    ('-nw', '--num_write', 10, int),
    
    ('-gbms', '--gen_beam_size', 100, int),
]

DOM_PT_ARGS = [    

    ('-ts', '--train_size', 200, int),    
    ('-evs', '--eval_size', 200,  int),
    ('-ets', '--etest_size', 200,  int),
    
    ('-mi', '--max_iters', 100000000, int),
    ('-esp', '--es_patience', 2500000, int),
    ('-prp', '--print_per', 10000, int),
    ('-evp', '--eval_per', 100000, int),
    ('-svp', '--save_per', None, int),
    ('-strm', '--stream_mode', 'y', str), # can be set to 's' for static
]

DOM_FT_ARGS = [
    
    ('-ftm', '--ft_mode', 'LEST_ST_WS', str),    
    ('-wts', '--ws_train_size', 30000, int),  
    ('-evp', '--eval_per', 1, int),
    
    ('-mi', '--max_iters', 10000, int),
    ('-infp', '--infer_patience', 10, int),
    ('-itrp', '--iter_patience', 250, int),
                            
    ('-esp', '--es_patience', 10,  int),
    
    ('-lest_w', '--lest_weight', 0., float),
    ('-st_w', '--st_weight', 0., float),
    ('-ws_w', '--ws_weight', 0., float),

    ('-ws_gn', '--ws_grace_num', 2, int),

]


class DOMAIN_BASE:
    
    def base_init(self):
        self.DOM_CMN_ARGS = DOM_CMN_ARGS
        self.obj_name = 'Obj'
        self.device = device        
    
    def vis_metric(self, pixels, gt):
        assert False, 'domain sub-class should implement'

    def load_real_data(self):
        assert False, 'domain sub-class should implement'

    def init_metric_val(self):
        assert False, 'domain sub-class should implement'

    def get_obj_dir(self):
        assert False, 'domain sub-class should implement'

    def make_blank_visual_batch(self):
        assert False, 'domain sub-class should implement'

    def get_synth_data_cls(self):
        assert False, 'domain sub-class should implement'
    
    def load_new_net(self):

        net = mu.load_inf_net(self)
            
        net.acc_count = 0
        net.acc_period = self.args.acc_period
        net.log_period = self.args.log_period

        net.to(self.device)
        
        return net
    
    def load_pretrained_net(self, model_path = None):
        if model_path is None:
            assert self.args.load_model_path is not None
            model_path = self.args.load_model_path
            
        net = self.load_new_net()

        net.load_state_dict(
            torch.load(
                model_path
            )
        )
        net.to(self.device)
        return net

    # early stopping logic using evaluation metric, and threshold
    def should_save(self, cur_val, best_val, thresh):

        if self.get_obj_dir() == 'high':
            thresh_val = cur_val - thresh
        elif self.get_obj_dir() == 'low':
            thresh_val = cur_val + thresh
        else:
            assert False, f'bad obj dir {self.get_obj_dir()}'
            
        if self.comp_metric(thresh_val, best_val):
            return True
        else:
            return False


    # is it better for the evaluation metric to be high or low?
    def comp_metric(self, a, b):

        if self.get_obj_dir() == 'high':
            if a > b:
                return True
            else:
                return False
            
        elif self.get_obj_dir() == 'low':
            if a < b:
                return True
            else:
                return False
        else:
            assert False, f'bad obj dir {self.get_obj_dir()}'    
        
    # what shape do the visual inputs take
    def get_input_shape(self):
        return self.executor.get_input_shape()

    def get_synth_datasets(self):

        SynthDatasetCls = self.get_synth_data_cls()
        
        train_loader = SynthDatasetCls(self.args, 'train', self.executor, self.device)
        val_loader = SynthDatasetCls(self.args, 'val', self.executor, self.device)

        eval_size = min(
            [
                v for v in
                (self.args.eval_size, train_loader.size, val_loader.size)
                if v is not None
            ]
        )

        train_loader.eval_size = eval_size
        val_loader.eval_size = eval_size

        train_loader.num_write = min(eval_size-1, self.args.num_write)
        val_loader.num_write = min(eval_size-1, self.args.num_write)
                
        return train_loader, val_loader

    def extra_prob_train_log_info(self):
        return []

    def extra_prob_eval_log_info(self):
        return []
    
    # pretraining arguments helper function
    def get_pt_args(self, extra_args = []):

        ARGS = utils.mergeArgs(
            extra_args + self.get_cmn_args() + self.get_pt_arg_list(),
            self.DOM_CMN_ARGS + DOM_PT_ARGS,
        )
        
        self.args = utils.getArgs(ARGS)        

        self.make_executor(self.args)
        
        self.TRAIN_LOG_INFO = DOM_PROB_TRAIN_LOG_INFO + self.extra_prob_train_log_info()
        self.EVAL_LOG_INFO =  DOM_PROB_EVAL_LOG_INFO + self.extra_prob_eval_log_info()
            
        self.executor.TLang.add_shared_tokens(self.executor.NUM_SHARED_TOKENS)
                
        utils.init_pretrain_run(self.args)        
        
        return self.args

    # finetuning arguments helper function
    def get_ft_args(self, extra_args=[]):
        
        ARGS = utils.mergeArgs(
            extra_args + self.get_cmn_args() + self.get_ft_arg_list(),
            self.DOM_CMN_ARGS + DOM_FT_ARGS
        )
        
        args = utils.getArgs(ARGS)

        args.infer_path = f"model_output/{args.exp_name}/train_out/"
        args.ws_save_path = f"model_output/{args.exp_name}/ws_out/"

        if 'LEST' in args.ft_mode.split('_'):
            args.lest_weight = 1.0

        if 'ST' in args.ft_mode.split('_'):
            args.st_weight = 1.0

        if 'WS' in args.ft_mode.split('_'):
            args.ws_weight = 1.0
        
        norm = args.lest_weight + args.st_weight  + args.ws_weight

        if norm > 0:
                
            args.lest_weight = args.lest_weight / norm
            args.st_weight = args.st_weight / norm
            args.ws_weight = args.ws_weight / norm
        
        self.args = args

        self.make_executor(self.args)
        
        self.EVAL_LOG_INFO = DOM_PROB_EVAL_LOG_INFO + \
            self.extra_prob_eval_log_info()

        self.executor.TLang.add_shared_tokens(self.executor.NUM_SHARED_TOKENS)
            
        utils.init_exp_model_run(self.args)                                   
                       
        return self.args

    def load_gen_model(self, gen_model_path=None):
        args = self.args
                
        gen_model = self.load_pretrained_net(gen_model_path)
                            
        gen_model.model_train_batch = gen_model.ws_model_train_batch
        gen_model.model_name = 'blank_gen'

        gen_model.gen_epoch = 0
        gen_model.to(self.device)
        
        return gen_model

    def load_magg_model(self, magg_model_path=None):

        magg_model = self.load_pretrained_net(magg_model_path)

        args = self.args
                
        magg_model.model_train_batch = magg_model.magg_model_train_batch
        magg_model.model_name = 'magg'
        magg_model.to(self.device)
        return magg_model

    def get_deriv_dof(self, prog_info):

        ddof = 0.
        
        if prog_info['deriv'] is not None:
            for fn in prog_info['deriv']:
                ddof += 1.
                inp_types = self.executor.TLang.get_inp_types(fn)
                for it in inp_types:
                    if it in self.executor.DEF_PARAM_TYPES and \
                       it not in self.executor.FLOAT_PARAM_TYPES:
                        ddof += 1.

        if prog_info['param'] is not None:
            tokens = prog_info['param']
            for t in tokens:
                if self.executor.TLang.get_out_type(t) in self.executor.FLOAT_PARAM_TYPES:
                    continue
                ddof += 1.
            
        return ddof
    
class SynthDataset:
    def __init__(
        self, args, set_name, ex, device
    ):

        
        self.mode = 'train'
        self.args = args
        self.ex = ex
        self.device= device
        
        self.set_name = set_name
                    
        self.batch_size = args.batch_size
        self.eval_batch_size = 1
        
        self.data = []                

        self.iter_num = 0
        self.inds = []
        
        assert args.stream_mode in ('s', 'y')
        if set_name == 'train':
            if args.stream_mode == 'y':
                self.do_stream = True
                self.size = None
            else:
                self.do_stream = False
                self.size = args.train_size
        else:
            self.do_stream = False
            self.size = args.eval_size
            
        self.eval_size = None
        
        if self.size is None:
            return

        with torch.no_grad():
            data = self.sample_data(self.size, print_info=True)
                

    def sample_data(self, num, print_info=False):


        if print_info:
            print(f"Preloading Prob Data for {self.set_name} "
                  f"({self.size} | {self.args.max_vis_inputs})"
            )

        sample_fn = self.ex.group_prog_random_sample
            
        self.data = sample_fn(
            num,
            self.args.max_vis_inputs,
            use_pbar = print_info
        )
            

                            
            
    def __iter__(self):

        if self.mode == 'train':

            if self.do_stream:
                yield from self.stream_iter()
            else:            
                yield from self.train_static_iter()
                            
        elif self.mode == 'eval':
            yield from self.eval_iter()

        else:
            assert False, f'bad mode {self.mode}'

    def make_stream_data(self):

        self.sample_data(
            self.batch_size * self.args.log_period
        )
        
        inds = list(range(len(self.data)))
        random.shuffle(inds)

        self.inds = inds
        
    def stream_iter(self):
        if len(self.inds) == 0:
            with torch.no_grad():
                self.make_stream_data()

        while len(self.inds) > 0:
            
            binds = self.inds[:self.batch_size]
            self.inds = self.inds[self.batch_size:]
            
            bdata = [self.data[bi] for bi in binds]

            with torch.no_grad():                
                    
                batch = self.ex.make_batch(bdata, self.args)
                
                g_batch = {
                    k: v.to(self.device) for k,v in
                    batch.items() if k != 'vis_vdata'
                }
                
            yield g_batch

            
    def train_static_iter(self):

        if len(self.inds) == 0:
            self.inds = list(range(len(self.data)))
            random.shuffle(self.inds)
            
        while len(self.inds) > 0:
            binds = self.inds[:self.batch_size]
            self.inds = self.inds[self.batch_size:]

            bdata = [self.data[bi] for bi in binds]            
            
            with torch.no_grad():
                
                batch = self.ex.make_batch(bdata, self.args)
                
                g_batch = {
                    k: v.to(self.device) for k,v in
                    batch.items() if k != 'vis_vdata'
                }
                
            yield g_batch
    
    def eval_iter(self):
        inds = torch.arange(len(self.data[:self.eval_size]))
        
        for start in range(
            0, inds.shape[0], 1
        ):
            binds = inds[start:start+1]

            bdata = [self.data[bi] for bi in binds]
                
            with torch.no_grad():
                batch = self.ex.make_batch(bdata, self.args)                
                
                g_batch = {
                    'vdata': batch['vdata'].to(self.device)
                }
                if 'vis_vdata' in batch:
                    g_batch['vis_vdata'] = batch['vis_vdata']
                
            g_batch['extra_gt_data'] = None
            
            yield g_batch



def image_chamfer(images1, images2):
    """
    Chamfer distance on a minibatch, pairwise.
    :param images1: Bool Images of size (N, 64, 64). With background as zeros
    and forground as ones
    :param images2: Bool Images of size (N, 64, 64). With background as zeros
    and forground as ones
    :return: pairwise chamfer distance
    """
    # Convert in the opencv data format
    images1 = images1.astype(np.uint8)
    images1 = images1 * 255
    images2 = images2.astype(np.uint8)
    images2 = images2 * 255
    N = images1.shape[0]
    size = images1.shape[-1]

    D1 = np.zeros((N, size, size))
    E1 = np.zeros((N, size, size))

    D2 = np.zeros((N, size, size))
    E2 = np.zeros((N, size, size))
    summ1 = np.sum(images1, (1, 2))
    summ2 = np.sum(images2, (1, 2))

    # sum of completely filled image pixels
    filled_value = int(255 * size**2)
    defaulter_list = []
    for i in range(N):
        img1 = images1[i, :, :]
        img2 = images2[i, :, :]

        if (summ1[i] == 0) or (summ2[i] == 0) or (summ1[i] == filled_value) or (summ2[\
                i] == filled_value):
            # just to check whether any image is blank or completely filled
            defaulter_list.append(i)
            continue
        edges1 = cv2.Canny(img1, 1, 3)
        sum_edges = np.sum(edges1)
        if (sum_edges == 0) or (sum_edges == size**2):
            defaulter_list.append(i)
            continue
        dst1 = cv2.distanceTransform(
            ~edges1, distanceType=cv2.DIST_L2, maskSize=3)

        edges2 = cv2.Canny(img2, 1, 3)
        sum_edges = np.sum(edges2)
        if (sum_edges == 0) or (sum_edges == size**2):
            defaulter_list.append(i)
            continue

        dst2 = cv2.distanceTransform(
            ~edges2, distanceType=cv2.DIST_L2, maskSize=3)
        D1[i, :, :] = dst1
        D2[i, :, :] = dst2
        E1[i, :, :] = edges1
        E2[i, :, :] = edges2
    distances = np.sum(D1 * E2, (1, 2)) / (
        np.sum(E2, (1, 2)) + 1) + np.sum(D2 * E1, (1, 2)) / (np.sum(E1, (1, 2)) + 1)

    distances = distances / 2.0
    # This is a fixed penalty for wrong programs
    distances[defaulter_list] = 10
    return distances


class TargetBase:
    def __init__(self):
        self.name = 'base'        
        
    def get_set_size(self, name):
        if name == 'train':
            return self.train_keys.shape[0]
        elif name == 'val':
            return self.val_keys.shape[0]
        elif name == 'test':
            return self.test_keys.shape[0]
        elif name == 'prompt':
            return self.fsg_prompt_keys.shape[0]

    def get_train_vinput(self):
        return self.vinput
    
    def train_eval_iter(self):
        keys = self.train_keys
        yield from self.eval_iter(keys)

    def val_eval_iter(self):
        keys = self.val_keys
        yield from self.eval_iter(keys)

    def test_eval_iter(self):
        keys = self.test_keys        
        yield from self.eval_iter(keys)

    def prompt_eval_iter(self):
        keys = self.fsg_prompt_keys
        yield from self.eval_iter(keys)
        
    def eval_iter(self, keys, ret_keys=True):
        for start in range(
            0, keys.shape[0], 1
        ):
            bkeys = keys[start:start+1]

            try:
                vinput = self.vinput[bkeys.flatten()].view(
                    bkeys.shape[0], bkeys.shape[1], \
                    self.vinput.shape[1], self.vinput.shape[2], self.vinput.shape[3]
                ).float().to(self.device)
            except:
                vinput = self.vinput[bkeys.flatten()].view(
                    bkeys.shape[0], bkeys.shape[1], \
                    self.vinput.shape[1], self.vinput.shape[2]
                ).float().to(self.device)
                
            if self.extra_gt_data is not None:
                extra_gt_data = [[self.extra_gt_data[bi] for bi in bg] for bg in bkeys]
            else:
                extra_gt_data = None

            if 'name' in self.__dict__ and self.name == 'shape':
                
                assert vinput.shape[0] == 1
                nvdata = []
                for j in range(vinput.shape[1]):
                    nvdata.append(self.ex.conv_scene_to_vinput(vinput[0,j]))

                nvdata = torch.stack(nvdata,dim=0).unsqueeze(0).float().to(vinput.device)
                
                if ret_keys:
                    yield {
                        'vis_vdata': vinput,
                        'vinput': nvdata,
                        'extra_gt_data': None,
                        'bkeys': bkeys
                    }
                else:
                    yield {
                        'vis_vdata': vinput,
                        'vdata': nvdata,
                        'extra_gt_data': None,
                    }
                    
                
            else:
                if ret_keys:
                    yield {
                        'bkeys': bkeys,
                        'vinput': vinput,
                        'extra_gt_data': extra_gt_data
                    }
                else:
                    yield {
                        'vdata': vinput,
                        'extra_gt_data': extra_gt_data
                    }

    def get_extra_gt_data(self, keys):

        if self.extra_gt_data is None:
            return None
        else:
            return [[self.extra_gt_data[bi] for bi in keys]]

    def __iter__(self):
        if self.mode == 'eval':
            yield from self.eval_iter(self.test_keys, False)
        else:
            assert False
