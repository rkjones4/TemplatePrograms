import os
import types
import math
import utils
import torch
from utils import device
import matplotlib.pyplot as plt
import train_utils as tru
import executors.common.lutils as lu
import search

PRT_MAP = lu.CMAP

COSEG_LOG_INFO = [
    ('Miou', 'miou', 'cs_count'),
    ('Acc', 'acc', 'cs_count'),
]

COSEG_ARGS = [
    ('-tbm', '--temp_beams', 5, int), # 40 -> change for expensive inference
    ('-ebm', '--exp_beams', 5, int), # 10 -> change for expensive inference
]

class CosegLoader:
    def __init__(self, tar_data, args):
        self.vinput = tar_data.vinput
        self.sem_data = tar_data.sem_data
        self.iter_num = 0
        
        self.eval_batch_size = 1
        
        self.set_name = 'test'        
        self.keys = tar_data.test_keys
        self.group_names = tar_data.test_group_names
                    
        self.save_path = f'{args.outpath}/{args.exp_name}/vis/'
        print(f"Loaded {self.keys.shape} keys from {self.set_name}")
        self.eval_size = len(self.keys)
        
    def __iter__(self):        

        for i, key_set in enumerate(self.keys):
                            
            try:
                sd =  self.sem_data[key_set].to(device)
            except:
                sd = [self.sem_data[ks] for ks in key_set]
                try:
                    sd = torch.stack(sd, dim=0).to(device)
                except:
                    pass
            
            keystr = ':'.join([str(ks.item()) for ks in key_set])
            
            yield {
                'vdata': self.vinput[key_set].to(device),
                'sdata': sd,
                'keys': keystr,
                'save_path': self.save_path,
                'set_name': self.set_name,
                'gname': self.group_names[i]
            }
        
def make_image(ex, preds, thr):
    
    img = torch.zeros(ex.VDIM * ex.VDIM, 3, device=preds.device)
        
    votes = preds.argmax(dim=1)
    occ = (preds.max(dim=1).values >= thr).nonzero().flatten()

    occ_votes = votes[occ]
        
    for v in occ_votes.cpu().unique():
        inds = occ[(occ_votes == v).nonzero().flatten()]
        col = PRT_MAP[v.item()].to(preds.device)
        img[inds,:] = col 
            
    return img.reshape(ex.VDIM, ex.VDIM, 3)

    
def calc_coseg_metrics(sem_segs, sem_datas):

    assert len(sem_segs) == len(sem_datas)

    res = {
        'miou': 0.,
        'cs_count': 0.,
        'acc': 0.,
    }

    mious = []
    
    for i in range(len(sem_segs)):

        if i == 0:
            continue
                
        raw_seg = sem_segs[i]
        raw_gt = sem_datas[i]    
        
        sem_pred = raw_seg.argmax(dim=1)
        
        exp_gt = raw_gt.view(-1, raw_gt.shape[-1])
        sem_gt = exp_gt.argmax(dim=1)
        
        occ_inds = (raw_gt.view(-1,raw_gt.shape[-1]).abs().sum(dim=1) > 0.01).nonzero().flatten()
        
        assert (exp_gt[occ_inds].abs().sum(dim=1) >= 0.01).all()
        
        ious = []

        for j in range(sem_datas.shape[-1]):

            p_occ = (sem_pred[occ_inds] == j)
            g_occ = (sem_gt[occ_inds] == j)

            union = (p_occ | g_occ).float().sum().item()

            if union < 1:
                continue

            inter = (p_occ & g_occ).float().sum().item()

            ious.append(inter/union)

        miou = torch.tensor(ious).mean().item()
        mious.append(miou)                
                
        acc = (sem_pred[occ_inds] == sem_gt[occ_inds]).float().mean().item()
        
        res['miou'] += miou
        res['acc'] += acc
        res['cs_count'] += 1.

    return res, mious
        
def eval_dom(domain):
    args = domain.get_ft_args(COSEG_ARGS)
    net = domain.load_pretrained_net()
    target_data = domain.load_real_data(mode='coseg')
    
    os.system(f'mkdir {args.outpath}/{args.exp_name}/plots/eval > /dev/null 2>&1')
    net.iter_num = 0

    test_loader = CosegLoader(target_data, args)    
    test_loader.mode = 'coseg'
    
    eval_data = [        
        ('test', test_loader),
    ]
    
    res = {
        'eval_iters': [],
        'eval_plots': {'test': {}}
    }

    if net.domain.name == 'shape':
        net.model_eval_fn = types.MethodType(coseg_shape_model_eval_fn, net)
    else:
        net.model_eval_fn = types.MethodType(coseg_model_eval_fn, net)
            
    tru.run_eval_epoch(
        args,
        res,
        net,
        eval_data,
        COSEG_LOG_INFO,
        0,
    )

def make_coseg_pred(inp, parts, p2s_map, valid_parts, num_sem_parts):
    gt_occ_pixels = (inp.view(-1, inp.shape[-1]).sum(dim=1) > 0.0).nonzero().flatten()

    for i in range(parts.shape[1]):
        if i not in valid_parts:
            parts[:,i] = -1.0
        
    raw_part_seg = parts.argmax(dim=1)

    occ_part_seg = raw_part_seg[gt_occ_pixels]
        
    part_seg = torch.zeros(parts.shape,device=device)
    sem_seg = torch.zeros(parts.shape[0], num_sem_parts,device=device)
    
    for v in valid_parts:
        vprt_occ = (occ_part_seg == v).nonzero().flatten()   

        part_seg[gt_occ_pixels[vprt_occ], v] = 1.0
        sem_seg[gt_occ_pixels[vprt_occ], p2s_map[v.item()]] = 1.0

    return part_seg, sem_seg
    
def find_coseg_map(inp, sem, parts):
    
    gt_occ_pixels = (inp.view(-1, inp.shape[-1]).sum(dim=1) > 0.0).nonzero().flatten()
    raw_part_seg = parts.argmax(dim=1)

    occ_part_seg = raw_part_seg[gt_occ_pixels]
        
    valid_parts = occ_part_seg.cpu().unique()

    sem_exp = sem.view(-1, sem.shape[-1])
        
    sem_map = {}        
        
    for v in valid_parts:
        vprt_occ = (occ_part_seg == v).nonzero().flatten()                                    
        best_sem_match = sem_exp[gt_occ_pixels[vprt_occ]].argmax(dim=1).mode().values

        sem_map[v.item()] = best_sem_match.item()

    return sem_map, valid_parts


def coseg_model_eval_fn(
    self, batch, 
):
    assert self.domain.name != 'shape'
    
    inp, sem_data = batch['vdata'], batch['sdata']

    ex = self.ex
        
    part_preds, recons, eval_info = make_part_preds(self, inp)

    p2s_map, valid_parts = find_coseg_map(inp[0], sem_data[0], part_preds[0])
    
    part_segs, sem_segs = [], []

    nsp = sem_data[0].shape[-1]
    
    for i in range(len(inp)):
        part_seg, sem_seg = make_coseg_pred(
            inp[i], part_preds[i], p2s_map, valid_parts, nsp
        )
        part_segs.append(part_seg)
        sem_segs.append(sem_seg)

    parse_images = get_parse_images(self, inp, eval_info)
        
    imgs = [inp[i] for i in range(len(inp))] + \
        recons + \
        parse_images + \
        [make_image(ex, ps, 0.01) for ps in part_segs] + \
        [make_image(ex, ss, 0.01) for ss in sem_segs] + \
        [
            make_image(ex, sem_data[i].view(-1, sem_data[i].shape[-1]), 0.01) 
            for i in range(len(sem_data)) 
        ] 

    _res, mious = calc_coseg_metrics(
        sem_segs,
        sem_data
    )

    save_name = f'{batch["save_path"]}/set_{batch["set_name"]}_keys_{batch["keys"]}_mious_{mious}'
        
    self.ex.render_group(imgs, name=save_name, rows=6)
            
    res = {}

    for k,v, in _res.items():
        res[k] = v

    return res


########
# PART PRED LOGIC
########

def make_part_preds(net, vdata):

    args = net.domain.args
    
    eval_info, _ = search.split_beam_search(
        net,
        {
            'vdata': vdata.unsqueeze(0),
            'extra_gt_data': None
        },
        args.temp_beams,
        args.exp_beams,
    )
    
    assert len(eval_info['info'][0]) == vdata.shape[0]
        
    recons = []
    new_part_preds = []

    ex = net.ex
    
    for mps in eval_info['info'][0]:
        expr = mps['expr']
        struct = mps['struct']
            
        out_expr = ex.add_part_info(expr, struct)
            
        img = ex.execute(expr)
        recons.append(img)            
            
        new_part_pred = ex.make_new_part_pred(out_expr)
        new_part_preds.append(new_part_pred)            
                        
    return new_part_preds, recons, eval_info

def get_parse_images(net, vdata, eval_info):
    assert len(eval_info['info'][0]) == vdata.shape[0]

    parse_imgs = []

    for mps in eval_info['info'][0]:
        expr = mps['expr']
        struct = mps['struct']

        out_expr = net.ex.add_part_info(expr, struct)

        if net.domain.name == 'omni':
            P = net.ex.prog_cls(net.ex)
            P.run(expr.split())
            parse = P.make_sem_seg()
            img = P.make_sem_img(parse)
        elif net.domain.name == 'layout':
            P = net.ex.prog_cls(net.ex)
            P.run(out_expr.split())
            P.sem_state = P.state
            img = P.make_sem_image()
        else:
            assert False, f'image parse for {net.domain.name}, not supported'

        parse_imgs.append(img)

    return parse_imgs

######################
## Shape Specific Logic
######################

query_points = None

def make_query_points():
    global query_points
    DIM = 64
    a = (torch.arange(DIM).float() / (DIM-1.)) - .5
    b = a.unsqueeze(0).unsqueeze(0).repeat(DIM, DIM, 1)
    c = a.unsqueeze(0).unsqueeze(2).repeat(DIM, 1, DIM)
    d = a.unsqueeze(1).unsqueeze(2).repeat(1, DIM, DIM)
    query_points = torch.stack((b,c,d), dim=3).view(-1, 3).to(device)

def label_voxels_with_prims(domain, prim_info, sem_info):
    global query_points
    if query_points is None:
        make_query_points()

    assert len(prim_info.shape) == 2
    NPI = (prim_info.abs().sum(dim=1) > 0.01).sum().item()

    prims = prim_info[:NPI]

    if sem_info is not None:
        labels = sem_info[:NPI]
    else:
        labels = None
    
    assert (prims.abs().sum(dim=1) > 0.01).all()
    
    ucubes = prims.unsqueeze(0)
        
    cent_pts = query_points.unsqueeze(1) - ucubes[:,:,3:6]    

    cube_sdfs = (
        cent_pts.abs() - ( ucubes[:,:,:3] / 2.)
    ).max(dim=2).values
    
    vthresh = (1.0 / 64) / 1.41

    order = prims[:,:3].prod(dim=1).argsort(dim=0,descending=True).tolist()
        
    votes = torch.zeros(query_points.shape[0],device=query_points.device).long() - 1

    for o in order:
        if sem_info is None:
            si = 0
        else:
            si = labels[o]
        inside = cube_sdfs[:,o] <= vthresh
        votes[inside] = si

    qpts = query_points[(votes >= 0)]
    qlabels = votes[(votes >= 0)]
    
    return qpts, qlabels
            
def make_shape_query_info(domain, group_prim_info, group_sem_info):    
    Qpts = []
    query_labels = []

    for prim_info, sem_info in zip(group_prim_info, group_sem_info):

        qpts, qlabels = label_voxels_with_prims(domain, prim_info, sem_info.to(prim_info.device))

        Qpts.append(qpts)
        query_labels.append(qlabels)

    return Qpts, query_labels
                                        
def coseg_shape_model_eval_fn(self, batch):
    group_prim_info, group_sem_info = batch['vdata'], batch['sdata']
    
    assert group_prim_info.shape[-1] == 6
    assert len(group_prim_info.shape) == 3
    assert group_prim_info.shape[0] == len(group_sem_info)

    vinput = torch.stack(
        [self.domain.executor.conv_scene_to_vinput(pi) for pi in group_prim_info],
        dim = 0
    )

    query_pts, query_labels = make_shape_query_info(
        self.domain,
        group_prim_info,
        group_sem_info
    )

    part_preds, recons = make_shape_part_preds(
        self,
        vinput.float(),
        query_pts,
    )

    ref_gt_scene = query_pts[0]
    ref_gt_labels = query_labels[0]
    ref_pred_parts = part_preds[0]

    p2s_map = {}
        
    for pi in ref_pred_parts.cpu().unique():
        pp_inds = (ref_pred_parts == pi).nonzero().flatten()        
        gtlbs = ref_gt_labels[pp_inds]
        plb = torch.mode(gtlbs).values
        p2s_map[pi.item()] = plb.item()

    part_segs, sem_segs = [], []
    
    for i in range(len(part_preds)):
       
        part_seg = part_preds[i]
        
        sem_seg = torch.zeros(
            part_seg.shape[0], device=part_seg.device
        ).long() - 1

        for pi, ps in p2s_map.items():
            pinds = (part_seg == pi)
            sem_seg[pinds] = ps

        if not (sem_seg >= 0).all():            
            sem_seg[(sem_seg < 0)] = torch.mode(ref_gt_labels).values.item()
                        
        part_segs.append(part_seg)
        sem_segs.append(sem_seg)
                
    imgs = []
        
    imgs += [(query_pts[i].cpu(), part_segs[i].cpu()) for i in range(vinput.shape[0])]
    imgs += [(query_pts[i].cpu(), sem_segs[i].cpu()) for i in range(vinput.shape[0])]
    imgs += [(query_pts[i].cpu(), query_labels[i].cpu()) for i in range(vinput.shape[0])]
        
    _res, mious = shape_calc_coseg_metrics(
        sem_segs,
        query_labels
    )
    
    save_name = f'{batch["save_path"]}/set_{batch["set_name"]}_keys_{batch["keys"]}_mious_{mious}'    
    render_pc_grid(imgs, name=save_name, rows=3)

    res = {}
        
    for k,v, in _res.items():
        res[k] = v

    return res

def add_pc_to_axis(ax, shape):
    pc, labels = shape

    for l in labels.cpu().unique():
        lpts = pc[(labels == l)]

        x = lpts[:,0].cpu().numpy()
        y = lpts[:,1].cpu().numpy()
        z = lpts[:,2].cpu().numpy()

        c = PRT_MAP[l.item()]
        ax.scatter(x, z, y, c=c.unsqueeze(0))            

def render_pc_grid(shapes, rows, name):
    
    if rows == 1:
        fig = plt.figure(figsize=(16,2))
    else:
        fig = plt.figure(figsize=(16,8))    

    extent = 0.5
    for i, shape in enumerate(shapes):            
        ax = fig.add_subplot(rows, math.ceil(len(shapes)/rows), i+1, projection='3d')
        ax.axis('off')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_proj_type('persp')
        ax.set_box_aspect(aspect = (1,1,1))
        
        ax.set_xlim(-extent, extent)
        ax.set_ylim(extent, -extent)
        ax.set_zlim(-extent, extent)

        add_pc_to_axis(ax, shape)

    plt.tight_layout()

    if name is None:
        plt.show()        
    else:
        plt.savefig(f'{name}.png')
        
    plt.close('all')
    plt.clf()

def shape_calc_coseg_metrics(sem_segs, sem_datas):

    assert len(sem_segs) == len(sem_datas)

    res = {
        'miou': 0.,
        'cs_count': 0.,
        'acc': 0.,
    }

    mious = []
    
    for i in range(len(sem_segs)):

        if i == 0:
            continue
                
        sem_pred = sem_segs[i].cpu()
        sem_gt = sem_datas[i].cpu()    
                
        ious = []
        
        for j in sem_datas[0].cpu().unique():

            p_occ = (sem_pred == j)
            g_occ = (sem_gt == j)

            union = (p_occ | g_occ).float().sum().item()

            if union < 1:
                continue

            inter = (p_occ & g_occ).float().sum().item()
            
            ious.append(inter/union)

        miou = torch.tensor(ious).mean().item()
        mious.append(miou)
                
        acc = (sem_pred == sem_gt).float().mean().item()
        
        res['miou'] += miou
        res['acc'] += acc
        res['cs_count'] += 1.

    return res, mious

def make_shape_part_preds(net, vdata, query_pts):

    G_labels, G_prim_info, eval_info = make_part_preds(net, vdata)

    part_preds = []
    for labels, prim_info, pts in zip(G_labels, G_prim_info, query_pts):
        
        NPI = (prim_info.abs().sum(dim=1) > 0.01).sum().item()
        prims = prim_info[:NPI]
        
        assert (prims.abs().sum(dim=1) > 0.01).all()
    
        ucubes = prims.unsqueeze(0)
        
        cent_pts = pts.to(ucubes.device).unsqueeze(1) - ucubes[:,:,3:6]    

        cube_sdfs = (
            cent_pts.abs() - ( ucubes[:,:,:3] / 2.)
        ).max(dim=2).values

        votes = labels[cube_sdfs.argmin(dim=1)]
            
        vthresh = (1.0 / 32) / 1.41

        order = prims[:,:3].prod(dim=1).argsort(dim=0,descending=True).tolist()
            
        for o in order:
            si = labels[o]
            inside = cube_sdfs[:,o] <= vthresh
            votes[inside] = si

        part_preds.append(votes)
            
    return part_preds, G_prim_info

