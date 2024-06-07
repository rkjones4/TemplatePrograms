import os, sys
import torch
import utils
import json




def load_domain(args):    

    if args.domain_name == 'layout':
        from domains.layout import LAYOUT_DOMAIN
        return LAYOUT_DOMAIN()

    elif args.domain_name == 'omni':
        from domains.omni import OMNI_DOMAIN
        return OMNI_DOMAIN()

    elif args.domain_name == 'shape':
        from domains.shape import SHAPE_DOMAIN
        return SHAPE_DOMAIN()
    
    else:
        assert False, f'bad domain name {args.domain_name}'
    
def main():
    main_args = utils.getArgs([
        ('-mm', '--main_mode', None, str), # Set the main mode ['finetune', 'pretrain']
        ('-dn', '--domain_name', None, str), # Set the domain ['layout', 'omni', 'shape']
    ])
    
    domain = load_domain(main_args)
    
    if main_args.main_mode == 'finetune':
        import finetune as ft
        return ft.fine_tune(domain)

    elif main_args.main_mode == 'pretrain':
        import pretrain as pre
        return pre.train(domain)

    elif main_args.main_mode == 'fsg_eval':
        import fsg_eval as fsg_eval
        return fsg_eval.fsg_eval(domain)

    elif main_args.main_mode == 'coseg':
        import coseg_task as coseg        
        return coseg.eval_dom(domain)

    elif main_args.main_mode == 'train_magg':
        import train_magg_net as tmagg
        tmagg.train_magg_net(domain)
        
    else:
        assert False, f'bad main main {main_args.main_mode}'

            
if __name__ == '__main__':    
    main()

