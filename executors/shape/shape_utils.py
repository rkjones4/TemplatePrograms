import torch
from copy import deepcopy

def split_progs(ex, expr):
    local_progs = []

    cur = []

    for token in expr.split()[1:-1]:
        cur.append(token)
        if token == 'end':
            local_progs.append(cur)
            cur = []

    return local_progs

def reformat_struct(ex, struct):
    local_progs = split_progs(ex, ' '.join(struct))

    to_insert = []

    root_prog = local_progs.pop(0)

    new_struct = ['START']
    
    for token in root_prog:
        if token == 'hier':
            to_insert.append(local_progs.pop(0))
            new_struct.append('hier')
            
        elif ex.STRUCT_LOC_TOKEN in token:
            to_insert.append([token])
            new_struct.append('hier')
        else:
            new_struct.append(token)
            
    for ti in to_insert:
        new_struct += ti
            
    new_struct += ['end']

    return new_struct

def find_deriv(ex, expr, struct):

    # Shape works differently
    assert ex.ex_name == 'shape'
    
    if ex.STRUCT_LOC_TOKEN not in ' '.join(struct):
        return []

    # List of root prog tokens, then all sub prog tokens, missing first start and last end
    local_progs = split_progs(ex, expr)

    derivs = []

    local_ind = 0
    
    for token in struct:
        if token == 'SubProg':
            break

        elif token == 'hier':
            local_ind += 1

        elif ex.STRUCT_LOC_TOKEN in token:
            local_ind += 1

            lt = local_progs[local_ind]

            lst = []

            for t in lt:
                if ex.TLang.get_out_type(t) in ex.DEF_STRUCT_TYPES:
                    lst.append(t)
            
            derivs.append(' '.join(lst))
                                    
    return derivs


def remove_deriv(ex, expr, struct):

    # Shape works differently
    assert ex.ex_name == 'shape'
    
    if ex.STRUCT_LOC_TOKEN not in ' '.join(struct):
        return expr

    # List of root prog tokens, then all sub prog tokens, missing first start and last end
    local_progs = split_progs(ex, expr)

    derivs = []

    hole_inds = set()
    local_ind = 0
    
    for token in struct:
        if token == 'SubProg':
            break

        elif token == 'hier':
            local_ind += 1

        elif ex.STRUCT_LOC_TOKEN in token:
            local_ind += 1
            hole_inds.add(local_ind)

    ndp = ['START']
    for i,lp in enumerate(local_progs):
        if i in hole_inds:
            continue
        ndp += lp

    ndp += ['end']
    
    return ' '.join(ndp)
