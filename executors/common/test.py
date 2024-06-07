import sys
sys.path.append('executors')
sys.path.append('executors/layout')
sys.path.append('executors/omniglot')
sys.path.append('executors/shape')
sys.path.append('./')
import ex_layout
import ex_omni
import ex_shape

def manual_test(ex):
    lines = ' '.join(sys.argv[2:])
    ex.execute(lines, vis=True)

class Dummy:
    pass

def check_eval_interm_logic(ex, data):
    import eutils as eu
    import tutils as tu
    
    dargs = Dummy
    dargs.max_struct_tokens = ex.MAX_STRUCT_TOKENS
    dargs.max_deriv_tokens = ex.MAX_DERIV_TOKENS
    dargs.max_param_tokens = ex.MAX_PARAM_TOKENS
    dargs.max_vis_inputs = 5

    ex.TLang.add_shared_tokens(ex.NUM_SHARED_TOKENS)
    
    for d in data:
        batch = tu.make_batch(ex, [d], dargs)
                
        struct_seq = batch['struct_seq'][0][:batch['struct_seq_weight'][0].sum().long().item()]

        deriv_inp = batch['deriv_seq'][0,0]

        s2di = tu.conv_struct_out_to_deriv_inp(ex, struct_seq.clone())[0]

        if not (s2di == deriv_inp[:s2di.shape[0]]).all():
            print("Failed struct out to deriv inp")
            a = 1/0

        for i in range(dargs.max_vis_inputs):

            if batch['deriv_seq_weight'][0,i].sum() < 1.0:
                continue
                        
            dind = (batch['deriv_seq'][0,i][1:] == 0).nonzero().flatten()[2]
            pind = (batch['param_seq'][0,i][1:] == 0).nonzero().flatten()[0] + 3
            
            dinp = batch['deriv_seq'][0,i,:dind]
            pinp = batch['param_seq'][0,i,:pind]

            d2pi, nct = tu.conv_deriv_out_to_param_inp(ex, dinp.clone())

            if nct > 0 and not (d2pi == pinp).all():
                print("failed deriv out to param inp")
                a = 1/0
                        
    print("Checked eval interim logic")

        
        
def check_inf_to_train_logic(ex, data):
    
    import eutils as eu
    import tutils as tu
    
    dargs = Dummy
    
    for d in data:
        infos = []                
        st, dt, pt = d.make_prob_prog_batch(
            ex, dargs
        )        
        str_inp = dt[0][0][:1+dt[0][0][1:].index('START')]
        
        for v in d.valid:
            infos.append({'struct':str_inp,'expr': v.prog})
        
        infd = eu.InferData(ex, infos, dargs)
        pst, pdt, ppt = infd.sd.make_prob_prog_batch(ex, dargs)

        if st != pst or dt != pdt or pt != ppt:
            print("failed check inf to train logic")
            a = 1/0

        for i,v in enumerate(d.valid):

            if ex.ex_name == 'shape':
                pdrv = ex.find_deriv(v.prog, str_inp)
            else:
                pdrv = tu.find_deriv(ex, v.prog, str_inp)
            
            tar = [' '.join(vl) for vl in v.struct_derivs.values()]

            if pdrv != tar:
                print("failed check inf to train logic")
                a = 1/0
                        
    print("Checked inf to train logic")

    
if __name__ == '__main__':

    nm = sys.argv[1] 

    config = {
        'MAX_TOKENS': 128,
        'MAX_STRUCT_TOKENS': 64,
        'MAX_DERIV_TOKENS': 24,
        'MAX_PARAM_TOKENS': 128
    }
    
    if nm == 'lay':
        ex = ex_layout.LayExecutor(config)
    elif nm == 'omni':
        ex = ex_omni.OmnExecutor(config)                
    elif nm == 'shape':
        ex = ex_shape.ShapeExecutor(config)
    else:
        assert False
                    
    md = sys.argv[2]

    data = ex.group_prog_random_sample(int(sys.argv[3]), int(sys.argv[4]), vis_progs=('vis' in md), use_pbar=True)
        
    check_inf_to_train_logic(ex, data)
    check_eval_interm_logic(ex, data)
