import random
from make_data import Concept, Part, _get_uni_val

def make_fish():
    fish_cat_vars = {
        'top_fin': ('yes', 'no'),
        'mouth': ('bubbles', 'yes', 'no'), 
        'flipper': ('yes', 'no'),
        'tail': ('green', 'blue')
    }

    def fish_sample_var_fn(cvars):
        svars = {}
        if cvars['top_fin'] == 'yes':
            svars['top_fin_prim'] = random.choice(['triangle', 'square'])
        elif cvars['top_fin'] == 'no':
            svars['top_fin_prim'] = None

        svars['bubble_prim'] = None
        svars['mouth_prim'] = None
        svars['mouth_color'] = 'red'
        svars['bubble_color'] = 'blue'
        
        if cvars['mouth'] == 'yes':
            svars['mouth_prim'] = 'circle'            
            svars['eye_color'] = random.choice(['green', 'blue'])
            
        elif cvars['mouth'] == 'no':            
            svars['eye_color'] = random.choice(['green', 'red', 'blue'])
            
        elif cvars['mouth'] == 'bubbles':
            svars['bubble_prim'] = 'circle'            
            svars['eye_color'] = random.choice(['green', 'red'])
            
        if cvars['flipper'] == 'yes':
            svars['lower_flipper_prim'] = 'square'
            svars['upper_flipper_prim'] = random.choice(['square', None])
        elif cvars['flipper'] == 'no':
            svars['lower_flipper_prim'] = None
            svars['upper_flipper_prim'] = None

        if cvars['tail'] == 'green':
            svars['color_0'] = 'green'
        elif cvars['tail'] == 'blue':
            svars['color_0'] = 'blue'            

        return svars
        
    Fish = Concept(
        'fish',
        fish_cat_vars,
        fish_sample_var_fn
    )
            
    Fish.add_part(Part(
        name = 'body',
        prim_info=('static', 'circle'),
        size_info=(
            ('uni', .4, .8),
            ('uni', .2, .5),
        ),
        loc_info=(
            ('static', 0.),
            ('static', 0.),
        ),
        color_info=('static', 'grey'),
        sem_info='body'
    ))
    Fish.add_part(Part(
        name = 'tail',
        prim_info=('static', 'triangle'),
        size_info=(
            ('uni', .1, .2),
            ('ref', 'tail', 'width')
        ),
        loc_info=(
            ('rel', 'left', ('body', 'right')),
            ('rel', 'bot', ('body', 'center_height'))
        ),
        color_info=('ref', 'share', 'color_0'),
        sem_info='tail'
    ))
    Fish.add_part(Part(
        name = 'top_fin',
        prim_info=('ref', 'share', 'top_fin_prim'),
        size_info = (
            ('uni', .15, .3),
            ('uni', .05, .1),
        ),
        loc_info=(
            ('static', 0.),
            ('rel', 'bot', ('body', 'top'))
        ),
        color_info=('ref', 'share', 'color_0'),
        sem_info='fin'
    ))
    Fish.add_part(Part(
        name = 'eye',
        prim_info=('static', 'circle'),
        size_info=(
            ('expr', 'div', ('ref', 'body', 'width'), ('uni', 3., 5.)),
            ('expr', 'div', ('ref', 'body', 'height'), ('uni', 2., 4.)),
        ),
        loc_info=(
            ('prel', 'left', ('body', 'left'), ('static', 1.25)),
            ('prel', 'top', ('body', 'top'), ('static', 1.25))
        ),
        color_info=('ref', 'share', 'eye_color'),
        sem_info='eye'
    ))
    Fish.add_part(Part(
        name = 'lower_flipper',
        prim_info=('ref', 'share', 'lower_flipper_prim'),
        size_info=(
            ('expr', 'div', ('ref', 'body', 'width'), ('uni', 3.,5.)),
            ('static', .05)
        ),
        loc_info=(
            ('prel', 'left', ('body', 'center_width'), ('uni', .5, 1.0)),
            ('prel', 'bot', ('body', 'bot'), ('uni', 1.25, 2.)),
        ),
        color_info=('ref', 'share', 'color_0'),
        sem_info='fin'
    ))
    Fish.add_part(Part(
        name = 'upper_flipper',
        prim_info=('ref', 'share', 'upper_flipper_prim'),
        size_info=(
            ('ref', 'lower_flipper', 'width'),
            ('ref', 'lower_flipper', 'height'),
        ),
        loc_info=(
            ('ref', 'lower_flipper', 'x_pos'),            
            ('expr', 'add', ('rel', 'top', ('lower_flipper', 'top')), ('static', .2)),
        ),
        color_info=('ref', 'share', 'color_0'),
        sem_info='fin'
    ))
    Fish.add_part(Part(
        name = 'mouth',
        prim_info=('ref', 'share', 'mouth_prim'),
        size_info=(
            ('uni', .05, .15),
            ('static', .05),
        ),
        loc_info=(
            ('prel', 'left', ('body', 'left'), ('static', 1.25)),
            ('prel', 'top', ('body', 'center_height'), ('static', 1.5))
        ),
        color_info=('ref', 'share', 'mouth_color'),
        sem_info='mouth'
    ))

    Fish.add_part(Part(
        name = 'bubble_bot',
        prim_info=('ref', 'share', 'bubble_prim'),
        size_info=(
            ('static', .1),
            ('static', .1),
        ),
        loc_info=(
            ('expr', 'add', ('rel', 'right', ('body', 'left')), ('static', -0.05)),
            ('rel', 'bot', ('body', 'top'))
        ),
        color_info=('ref', 'share', 'bubble_color'),
        sem_info='bubble'
    ))

    Fish.add_part(Part(
        name = 'bubble_top',
        prim_info=('ref', 'share', 'bubble_prim'),
        size_info=(
            ('static', .1),
            ('static', .1),
        ),
        loc_info=(
            ('expr', 'add', ('rel', 'center', ('bubble_bot', 'center_width')), ('uni', 0.05, 0.15)),
            ('expr', 'add', ('rel', 'bot', ('bubble_bot', 'top')), ('uni', 0.1, 0.2)),            
        ),
        color_info=('ref', 'share', 'bubble_color'),
        sem_info='bubble'
    ))
    
    return Fish

def make_person():
    
    person_cat_vars = {
        'hat': ('pointy', 'regular', 'no'),
        'must': ('yes', 'no'),
        'tie': ('yes', 'no'),
        'eye': ('green', 'blue')
    }
    
    def person_sample_var_fn(cvars):
        svars = {}

        svars['pointy_hat_prim'] = None
        svars['top_hat_prim'] = None
        svars['cap_prim'] = None
        
        if cvars['hat'] == 'regular':            
            if random.random() < .5:
                svars['top_hat_prim'] = 'square'                
            else:
                svars['cap_prim'] = 'circle'
                
        elif cvars['hat'] == 'pointy':
            svars['pointy_hat_prim'] = 'triangle'            
        
        if cvars['must'] == 'yes':
            svars['must_prim'] = 'square'
        elif cvars['must'] == 'no':
            svars['must_prim'] = None

        if cvars['tie'] == 'yes':
            svars['tie_prim'] = 'triangle'
        elif cvars['tie'] == 'no':
            svars['tie_prim'] = None
            
        if cvars['eye'] == 'green':
            svars['color_2'] = 'green'
            svars['color_0'] = random.choice(['red', 'blue'])
        elif cvars['eye'] == 'blue':
            svars['color_2'] = 'blue'
            svars['color_0'] = random.choice(['red', 'green'])

        svars['color_1'] = random.choice(['green', 'red', 'blue'])
        while svars['color_1'] == svars['color_0']:
            svars['color_1'] = random.choice(['green', 'red', 'blue'])

        return svars
        
    Person = Concept(
        'person',
        person_cat_vars,
        person_sample_var_fn
    )
            
    Person.add_part(Part(
        name = 'head',
        prim_info=('static', 'circle'),
        size_info=(
            ('uni', .6, .8),
            ('uni', .6, .8),
        ),
        loc_info=(
            ('static', 0.),
            ('uni', -.2, 0.),
        ),
        color_info=('static', 'grey'),
        sem_info='head'
    ))
    Person.add_part(Part(
        name = 'left_eye',
        prim_info=('static', 'circle'),
        size_info=(
            ('expr', 'div', ('ref', 'head', 'width'), ('uni', 3.5, 5.5)),
            ('ref', 'left_eye', 'width'),
        ),
        loc_info=(
            ('prel', 'right', ('head', 'center_width'), ('uni', 1.2, 1.8)),
            ('prel', 'bot', ('head', 'center_height'), ('uni', 1.0,1.5))
        ),
        color_info=('ref', 'share', 'color_2'),
        sem_info='eye'
    ))
    Person.add_part(Part(
        name = 'right_eye',
        prim_info=('static', 'circle'),
        size_info=(
            ('ref', 'left_eye', 'width'),
            ('ref', 'left_eye', 'height'),
        ),
        loc_info=(
            ('expr', 'div', ('ref', 'left_eye', 'x_pos'), ('static', -1.0)),
            ('ref', 'left_eye', 'y_pos')
        ),
        color_info=('ref', 'share', 'color_2'),
        sem_info='eye'
    ))
    Person.add_part(Part(
        name = 'nose',
        prim_info=('static', 'triangle'),
        size_info=(
            ('uni', .05, .15),
            ('uni', .1, .2)
        ),
        loc_info=(
            ('static', 0.),
            ('prel', 'top', ('left_eye', 'bot'), ('uni', 0.2, 0.5))
        ),
        color_info=('ref', 'share', 'color_0'),
        sem_info='nose'
    ))            
    Person.add_part(Part(
        name = 'must',
        prim_info=('ref', 'share', 'must_prim'),
        size_info = (
            ('expr', 'div', ('ref', 'head', 'width'), ('static', 2.)),
            ('static', .05),
        ),
        loc_info=(
            ('static', 0.),
            ('expr', 'add', ('rel', 'top', ('nose', 'bot')), ('static', -0.05))
        ),
        color_info=('ref', 'share', 'color_1'),
        sem_info='must'
    ))
    Person.add_part(Part(
        name = 'mouth',
        prim_info=('static', 'circle'),
        size_info=(
            ('uni', .15, .3),
            ('uni', .05, .1),
        ),
        loc_info=(
            ('static', 0.),
            ('prel', 'top', ('must', 'bot'), ('uni', 1.25, 1.5))
        ),
        color_info=('ref', 'share', 'color_0'),
        sem_info='mouth'
    ))

    Person.add_part(Part(
        name = 'top_hat_upper',
        prim_info=('ref', 'share', 'top_hat_prim'),
        size_info=(
            ('uni', .35, .45),
            ('uni', .15, .2)
        ),
        loc_info=(
            ('static', 0.),
            ('expr', 'sub', ('static', .9), ('ref', 'top_hat_upper', 'height')),
        ),
        color_info=('static', 'red'),
        sem_info='hat'
    ))
    
    Person.add_part(Part(
        name = 'top_hat_bot',
        prim_info=('ref', 'share', 'top_hat_prim'),
        size_info=(
            ('uni', .8, .95),
            ('static', .05)
        ),
        loc_info=(
            ('static', 0.),
            ('rel', 'top', ('top_hat_upper', 'bot')),
        ),
        color_info=('static', 'red'),
        sem_info='hat'
    ))

    Person.add_part(Part(
        name = 'cap',
        prim_info=('ref', 'share', 'cap_prim'),
        size_info=(
            ('uni', .5, .7),
            ('uni', .1, .15)
        ),
        loc_info=(
            ('uni', .25,.4),
            ('rel', 'center', ('head', 'top')),
        ),
        color_info=('static', 'red'),
        sem_info='hat'
    ))

    Person.add_part(Part(
        name = 'pointy_hat',
        prim_info=('ref', 'share', 'pointy_hat_prim'),
        size_info=(
            ('uni', .55, .65),
            ('uni', .25, .35)
        ),
        loc_info=(
            ('static', 0.),
            ('expr', 'sub', ('static', .9), ('ref', 'top_hat_upper', 'height')),
        ),
        color_info=('static', 'red'),
        sem_info='hat'
    ))

    Person.add_part(Part(
        name = 'tie',
        prim_info = ('ref', 'share', 'tie_prim'),
        size_info = (            
            ('uni', 0.25, 0.3),
            ('static', 0.1),
        ),
        loc_info = (
            ('expr', 'mul', ('ref', 'tie', 'width'), ('static', 0.75)),
            ('rel', 'top', ('head', 'bot')),
        ),
        color_info=('ref', 'share', 'color_2'),
        top_info= ('static', ('symReflect', 'AX')),
        sem_info='tie'
    ))
    
    return Person


def make_caterpillar():

    caterpillar_cat_vars = {
        'body': ('green', 'red'),
        'feet': ('circle', 'triangle'),
        'ant': ('yes', 'no'),
        'bluefeet': ('yes', 'no')
    }


    def caterpillar_sample_var_fn(cvars):
        svars = {}
                
        if cvars['body'] == 'green':
            svars['color_0'] = 'green'
        elif cvars['body'] == 'red':
            svars['color_0'] = 'red'

        if cvars['feet'] == 'circle':
            svars['feet_prim'] = 'circle'
        elif cvars['feet'] == 'triangle':
            svars['feet_prim'] = 'triangle'

        if cvars['ant'] == 'yes':
            svars['ant_prim'] = 'circle'            
        elif cvars['ant'] == 'no':
            svars['ant_prim'] = None

        if cvars['bluefeet'] == 'yes':
            svars['color_1'] = 'blue'
        elif cvars['bluefeet'] == 'no':
            svars['color_1'] = svars['color_0']

        return svars
            
    Caterpillar = Concept(
        'caterpillar',
        caterpillar_cat_vars,
        caterpillar_sample_var_fn
    )

    Caterpillar.add_part(Part(
        name = 'body_right',
        prim_info=('static', 'circle'),
        size_info=(
            ('uni', .15, .2),
            ('uni', .15, .2),
        ),
        loc_info=(
            ('uni', .4, .5),
            ('uni', -.2, .2)
        ),
        color_info=('ref', 'share', 'color_0'),
        sem_info='body',
        top_info= ('fn', cat_body_si)
    ))
    Caterpillar.add_part(Part(
        name = 'feet_left',
        prim_info=('ref', 'share', 'feet_prim'),
        size_info=(
            ('static', 0.1),
            ('static', 0.1)
        ),
        loc_info=(
            ('ref', 'share', 'foot_loc'),
            ('rel', 'top', ('body_right', 'bot'))
        ),
        color_info=('ref', 'share', 'color_1'),
        sem_info='foot',
        top_info= ('fn', cat_foot_si)
    ))
    Caterpillar.add_part(Part(
        name = 'head',
        prim_info=('static', 'circle'),
        size_info=(
            ('uni', .15, .2),
            ('ref', 'head', 'width')
        ),
        loc_info=(
            ('prel', 'left', ('body_right', 'right'), ('uni', .4, .6)),
            ('expr', 'add', ('ref', 'body_right', 'y_pos'), ('uni', .1, .15)),
        ),
        color_info=('static', 'grey'),
        sem_info='head'
    ))
    Caterpillar.add_part(Part(
        name = 'left_ant',
        prim_info=('ref', 'share', 'ant_prim'),
        size_info=(
            ('uni', .05, .1),
            ('ref', 'left_ant', 'width')
        ),
        loc_info=(
            ('prel', 'left', ('head', 'left'), ('static', .75)),
            ('prel', 'bot', ('head', 'top'), ('static', .75))
        ),
        color_info=('ref', 'share', 'color_0'),
        sem_info='ant',        
    ))
    Caterpillar.add_part(Part(
        name = 'right_ant',
        prim_info=('ref', 'share', 'ant_prim'),
        size_info=(
            ('ref', 'left_ant', 'width'),
            ('ref', 'left_ant', 'width')
        ),
        loc_info=(
            ('prel', 'right', ('head', 'right'), ('static', .75)),
            ('prel', 'bot', ('head', 'top'), ('static', .75))
        ),
        color_info=('ref', 'share', 'color_0'),
        sem_info='ant',
    ))
    
    
    return Caterpillar

def cat_body_si(SP, samples, shared_vars):
    
    width = SP.width

    max_k = int(1./(width*2))
            
    K = random.randint(2, min(max_k, 4))

    min_extent = width * K * 2.25

    assert 'foot_loc' not in shared_vars    
    shared_vars['foot_loc'] = SP.x_pos
    shared_vars['foot_k'] = K - 1
    shared_vars['foot_extent'] = min_extent - (min_extent/ K)
    
    return ('symTranslate', -min_extent, 0, K)

def cat_foot_si(SP, samples, shared_vars):
        
    return ('symTranslate', -1 * shared_vars['foot_extent'], 0, shared_vars['foot_k'])

    
def make_flower():

    flower_cat_vars = {
        'petalcount': ('four', 'six'),
        'leaves': ('double', 'side', 'none'), 
        'petalcolor': ('red','blue'),
        'pot': ('yes', 'no')
    }

    def flower_sample_var_fn(cvars):
        svars = {}

        svars['flower_four_prim'] = None
        svars['flower_six_prim'] = None
        
        if cvars['petalcount'] == 'four':
            svars['flower_four_prim'] = 'circle'
        elif cvars['petalcount'] == 'six':
            svars['flower_six_prim'] = 'circle'

        if cvars['petalcolor'] == 'red':
            svars['color_flower'] = 'red'
        elif cvars['petalcolor'] == 'blue':
            svars['color_flower'] = 'blue'

        if cvars['pot'] == 'yes':
            svars['pot_prim'] = 'square'
        elif cvars['pot'] == 'no':
            svars['pot_prim'] = None

        svars['dbl_leaf_prim'] = None

        svars['left_leaf_prim'] = None
        svars['left_flower_prim'] = None
        svars['right_leaf_prim'] = None
        svars['right_flower_prim'] = None
        
        if cvars['leaves'] == 'double':
            svars['dbl_leaf_prim'] = 'triangle'
        elif cvars['leaves'] == 'side':
            if random.random() < 0.5:
                svars['left_leaf_prim'] = 'circle'
                if random.random() < 0.5:                
                    svars['left_flower_prim'] = 'circle'
            else:
                svars['right_leaf_prim'] = 'circle'
                if random.random() < 0.5:                
                    svars['right_flower_prim'] = 'circle'
        
        svars['color_base'] = random.choice(['red', 'blue', 'grey'])

        while svars['color_base'] == svars['color_flower']:
            svars['color_base'] = random.choice(['red', 'blue', 'grey'])
        
        return svars


    Flower = Concept(
        'flower',
        flower_cat_vars,
        flower_sample_var_fn
    )

    Flower.add_part(Part(
        name='pot',
        prim_info=('ref', 'share', 'pot_prim'),
        size_info = (
            ('uni', .5,.7),
            ('static', .1),
        ),
        loc_info = (
            ('static', 0.),
            ('uni', -.8, -1.0)
        ),
        color_info=('static', 'grey'),
        sem_info='pot'
    ))
    Flower.add_part(Part(
        name='stem',
        prim_info=('static', 'square'),
        size_info = (
            ('uni', .05,.1),
            ('expr', 'div', ('ref', 'pot', 'y_pos'), ('static', -2.0))
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'bot', ('pot', 'top'))
        ),
        color_info=('static', 'green'),
        sem_info = 'stem'
    ))
    Flower.add_part(Part(
        name='base',
        prim_info=('static', 'circle'),
        size_info= (
            ('uni', .15, .2),
            ('ref', 'base', 'width')
        ),
        loc_info=(
            ('static', 0.),
            ('static', 0.),
        ),
        color_info=('ref', 'share', 'color_base'),
        sem_info = 'base'
    ))

    Flower.add_part(Part(
        name='left_leaf',
        prim_info=('ref', 'share', 'left_leaf_prim'),
        size_info = (
            ('uni', .2,.25),
            ('static', .05),
        ),
        loc_info = (
            ('rel', 'right', ('stem', 'left')),
            ('expr', 'add', ('rel', 'center', ('stem', 'center_height')), ('uni', -.2, -.1))
        ),
        color_info = ('static', 'green'),
        sem_info = 'leaf',        
    ))

    Flower.add_part(Part(
        name='left_flower',
        prim_info=('ref', 'share', 'left_flower_prim'),
        size_info = (
            ('static', .15),
            ('static', .15),
        ),
        loc_info = (
            ('rel', 'center', ('left_leaf', 'left')),
            ('rel', 'center', ('left_leaf', 'center_height')),
        ),
        color_info = ('ref', 'share', 'color_flower'),
        sem_info = 'flower',        
    ))

    Flower.add_part(Part(
        name='right_leaf',
        prim_info=('ref', 'share', 'right_leaf_prim'),
        size_info = (
            ('uni', .2,.25),
            ('static', .05),
        ),
        loc_info = (
            ('rel', 'left', ('stem', 'right')),
            ('expr', 'add', ('rel', 'center', ('stem', 'center_height')), ('uni', -.2, -.1))
        ),
        color_info = ('static', 'green'),
        sem_info = 'leaf',        
    ))

    Flower.add_part(Part(
        name='right_flower',
        prim_info=('ref', 'share', 'right_flower_prim'),
        size_info = (
            ('static', .15),
            ('static', .15),
        ),
        loc_info = (
            ('rel', 'center', ('right_leaf', 'right')),
            ('rel', 'center', ('right_leaf', 'center_height')),
        ),
        color_info = ('ref', 'share', 'color_flower'),
        sem_info = 'flower',        
    ))
    
    Flower.add_part(Part(
        name='flower_four',
        prim_info=('ref', 'share', 'flower_four_prim'),
        size_info = (
            ('uni', .1,.2),
            ('ref', 'flower_four', 'width')
        ),
        loc_info= (
            ('static', 0.),
            ('rel', 'top', ('base', 'bot'))
        ),
        color_info=('ref', 'share', 'color_flower'),
        sem_info='flower',
        top_info = ('static', ('symRotate', str(4)))
    ))
    Flower.add_part(Part(
        name='flower_six',
        prim_info=('ref', 'share', 'flower_six_prim'),
        size_info = (
            ('uni', .1,.2),
            ('ref', 'flower_six', 'width')
        ),
        loc_info= (
            ('static', 0.),
            ('rel', 'top', ('base', 'bot'))
        ),
        color_info=('ref', 'share', 'color_flower'),
        sem_info='flower',
        top_info = ('static', ('symRotate', str(6)))
    ))

    Flower.add_part(Part(
        name='dbl_left_leaf',
        prim_info=('ref', 'share', 'dbl_leaf_prim'),
        size_info = (
            ('uni', .15,.3),
            ('static', .1),
        ),
        loc_info = (
            ('prel', 'right', ('stem', 'left'), ('static', .75)),
            ('expr', 'add', ('rel', 'bot', ('stem', 'bot')), ('static', 0.05))
        ),
        color_info = ('static', 'green'),
        sem_info = 'leaf',
        top_info = ('static', ('symReflect', 'AX'))
    ))
                                
    return Flower

def make_mushroom():

    
    mushroom_cat_vars = {
        'stem': ('square', 'circle'),
        'dots': ('ref', 'trans'),
        'dotcolor': ('red', 'blue'),
        'hang': ('yes', 'no')        
    }

    def mushroom_sample_var_fn(cvars):
        svars = {}

        svars['square_stem_prim'] = None
        svars['circle_stem_prim'] = None

        if cvars['stem'] == 'square':
            svars['square_stem_prim'] = 'square'
        elif cvars['stem'] == 'circle':
            svars['circle_stem_prim'] = 'circle'

        svars['dot_ref_prim'] = random.choice(['triangle', 'circle', 'square'])
            
        if cvars['dots'] == 'trans':
            svars['dot_trans_prim'] = svars['dot_ref_prim']
        elif cvars['dots'] == 'ref':
            svars['dot_trans_prim'] = None

        if cvars['dotcolor'] == 'red':
            svars['color_dot'] = 'red'
        elif cvars['dotcolor'] == 'blue':
            svars['color_dot'] = 'blue'

        if cvars['hang'] == 'yes':
            svars['hang_prim'] = 'triangle'
        elif cvars['hang'] == 'no':
            svars['hang_prim'] = None
            
        return svars

    Mushroom = Concept(
        'mushroom',
        mushroom_cat_vars,
        mushroom_sample_var_fn
    )

    Mushroom.add_part(Part(
        name='top',
        prim_info=('static', 'triangle'),
        size_info = (
            ('uni', .4,.8),
            ('uni', .25,.5),
        ),
        loc_info = (
            ('static', 0.),
            ('uni', .1, .3)
        ),
        color_info=('static', 'grey'),
        sem_info = 'top'
    ))

    Mushroom.add_part(Part(
        name='square_stem',
        prim_info=('ref', 'share', 'square_stem_prim'),
        size_info = (
            ('uni', .1,.2),
            ('uni', .1,.4),
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'top', ('top', 'bot'))
        ),
        color_info=('static', 'green'),
        sem_info = 'stem'
    ))

    Mushroom.add_part(Part(
        name='circle_stem_top',
        prim_info=('ref', 'share', 'circle_stem_prim'),
        size_info = (
            ('uni', .1,.2),
            ('uni', .1,.2),
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'top', ('top', 'bot'))
        ),
        color_info=('static', 'green'),
        sem_info = 'stem'
    ))

    Mushroom.add_part(Part(
        name='circle_stem_bot',
        prim_info=('ref', 'share', 'circle_stem_prim'),
        size_info = (
            ('ref', 'circle_stem_top', 'width'),
            ('uni', .1,.2),
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'top', ('circle_stem_top', 'bot'))
        ),
        color_info=('static', 'green'),
        sem_info = 'stem'
    ))

    Mushroom.add_part(Part(
        name='dot_ref',
        prim_info=('ref', 'share', 'dot_ref_prim'),
        size_info = (
            ('uni', .05, .1),
            ('ref', 'dot_ref', 'width')
        ),
        loc_info = (
            ('prel', 'left', ('top', 'center_width'), ('uni', 2.5, 3.5)),
            ('prel', 'bot', ('top', 'bot'), ('uni', 1.1, 1.3))
        ),
        color_info=('ref', 'share', 'color_dot'),
        sem_info = 'dot',
        top_info = ('static', ('symReflect', 'AX') )
    ))

    Mushroom.add_part(Part(
        name='dot_trans',
        prim_info=('ref', 'share', 'dot_trans_prim'),
        size_info = (
            ('uni', .05, .1),
            ('ref', 'dot_trans', 'width')
        ),
        loc_info = (
            ('static', 0.),
            ('prel', 'bot', ('top', 'bot'), ('uni', 1.3, 1.5))
        ),
        color_info=('ref', 'share', 'color_dot'),
        sem_info = 'dot',
    ))

    Mushroom.add_part(Part(
        name='hang',
        prim_info=('ref', 'share', 'hang_prim'),
        size_info = (
            ('uni', .05, .15),
            ('ref', 'hang', 'width'),
        ),
        loc_info = (
            ('rel', 'left', ('top', 'left')),
            ('rel', 'top', ('top', 'bot')),
        ),
        color_info=('static', 'grey'),
        sem_info = 'hang',
        top_info = ('static', ('symReflect', 'AX') )
    ))
    
    return Mushroom

def make_crab():
    crab_cat_vars = {
        'arms': ('top_in', 'top_out', 'side'), 
        'feet': ('yes', 'no'),
        'crabcolor': ('red','blue'),
        'clawcolor': ('grey', 'match')
    }

    def crab_sample_var_fn(cvars):
        svars = {}

        svars['side_arm_prim'] = None
        svars['side_claw_prim'] = None
        svars['top_arm_prim'] = None
        svars['top_claw_prim'] = None
        svars['top_in_claw_prim'] = None
        
        if  cvars['arms'] == 'side':
            svars['side_arm_prim'] = 'square'
            svars['side_claw_prim'] = 'triangle'

        elif cvars['arms'] == 'top_out':
            svars['top_arm_prim'] = 'square'
            svars['top_claw_prim'] = 'triangle'

        elif cvars['arms'] == 'top_in':
            svars['top_in_claw_prim'] = 'triangle'
            
        if cvars['feet'] == 'yes':
            svars['feet_prim'] = 'circle'
        elif cvars['feet'] == 'no':
            svars['feet_prim'] = None

        if cvars['crabcolor'] == 'red':
            svars['body_color'] = 'red'
        if cvars['crabcolor'] == 'blue':
            svars['body_color'] = 'blue'

        if cvars['clawcolor'] == 'grey':
            svars['tip_color'] = 'grey'
        elif cvars['clawcolor'] == 'match':
            svars['tip_color'] = svars['body_color']

        svars['eye_prim'] = random.choice(['triangle', 'circle', 'square'])

        return svars 

    Crab = Concept(
        'crab',
        crab_cat_vars,
        crab_sample_var_fn
    )

    Crab.add_part(Part(
        name = 'body',
        prim_info=('static', 'square'),
        size_info=(
            ('uni', .4,.6),
            ('uni', .25,.4),
        ),
        loc_info=(
            ('static', 0.),
            ('static', 0.),
        ),
        color_info=('ref', 'share', 'body_color'),
        sem_info='body'
    ))

    Crab.add_part(Part(
        name = 'eye',
        prim_info=('ref', 'share', 'eye_prim'),
        size_info=(
            ('uni', .05,.15),
            ('uni', .05,.1),
        ),
        loc_info=(
            ('uni', .2, .25),
            ('uni', .05, .1),
        ),
        color_info=('static', 'green'),
        sem_info='eye',
        top_info=('static', ('symReflect', 'AX'))
    ))

    Crab.add_part(Part(
        name = 'feet',
        prim_info=('ref', 'share', 'feet_prim'),
        size_info=(
            ('uni', .15,.25),
            ('uni', .05,.15),
        ),
        loc_info=(
            ('prel', 'left', ('body', 'left'), ('uni', 1.,1.5)),
            ('rel', 'top', ('body', 'bot')),
        ),
        color_info=('ref', 'share', 'tip_color'),
        sem_info='foot',
        top_info=('static', ('symReflect', 'AX'))
    ))

    Crab.add_part(Part(
        name = 'side_arm_low',
        prim_info=('ref', 'share', 'side_arm_prim'),
        size_info=(
            ('uni', .15,.2),
            ('static', .05)
        ),
        loc_info=(
            ('rel', 'left', ('body', 'right')),
            ('rel', 'top', ('body', 'top')),
        ),
        color_info=('static', 'grey'),
        sem_info='arm',
        top_info=('static', ('symReflect', 'AX'))
    ))
    
    Crab.add_part(Part(
        name = 'side_arm_up',
        prim_info=('ref', 'share', 'side_arm_prim'),
        size_info=(
            ('ref', 'side_arm_low', 'height'),
            ('uni', .1, .2),
        ),
        loc_info=(
            ('rel', 'right', ('side_arm_low', 'right')),
            ('rel', 'bot', ('side_arm_low', 'top')),
        ),
        color_info=('static', 'grey'),
        sem_info='arm',
        top_info=('static', ('symReflect', 'AX'))
    ))

    Crab.add_part(Part(
        name = 'side_claw',
        prim_info=('ref', 'share', 'side_claw_prim'),
        size_info=(
            ('uni', .15, .25),
            ('static', .05),
        ),
        loc_info=(
            ('rel', 'center', ('side_arm_up', 'center_width')),
            ('rel', 'bot', ('side_arm_up', 'top')),
        ),
        color_info=('ref', 'share', 'tip_color'),
        sem_info='claw',
        top_info=('static', ('symReflect', 'AX'))
    ))

    Crab.add_part(Part(
        name = 'top_arm',
        prim_info=('ref', 'share', 'top_arm_prim'),
        size_info=(
            ('static', .05),
            ('uni', .1, .15),
        ),
        loc_info=(
            ('prel', 'right', ('body', 'right'), ('uni', 1.1, 1.25)),
            ('rel', 'bot', ('body', 'top')),
        ),
        color_info=('static', 'grey'),
        sem_info='arm',
        top_info=('static', ('symReflect', 'AX'))
    ))

    Crab.add_part(Part(
        name = 'top_claw',
        prim_info=('ref', 'share', 'top_claw_prim'),
        size_info=(
            ('uni', .15, .25),
            ('uni', .1, .15),
        ),
        loc_info=(
            ('rel', 'center', ('top_arm', 'center_width')),
            ('rel', 'bot', ('top_arm', 'top')),
        ),
        color_info=('ref', 'share', 'tip_color'),
        sem_info='claw',
        top_info=('static', ('symReflect', 'AX'))
    ))

    Crab.add_part(Part(
        name = 'top_in_claw',
        prim_info=('ref', 'share', 'top_in_claw_prim'),
        size_info=(
            ('uni', .15, .25),
            ('uni', .1, .15),
        ),
        loc_info=(
            ('rel', 'center', ('top_arm', 'center_width')),
            ('rel', 'bot', ('body', 'top')),
        ),
        color_info=('ref', 'share', 'tip_color'),
        sem_info='claw',
        top_info=('static', ('symReflect', 'AX'))
    ))

    return Crab
    
def make_cat():

    cat_cat_vars = {
        'whisk': ('yes', 'no'),
        'mouth': ('yes', 'no'),
        'eyes': ('green', 'blue'),
        'paws': ('yes', 'no')
    }

    def cat_sample_var_fn(cvars):
        svars = {}

        if cvars['whisk'] == 'yes':
            svars['whisk_prim'] = random.choice(['circle', 'square'])
        elif cvars['whisk'] == 'no':
            svars['whisk_prim'] = None

        if cvars['mouth'] == 'yes':
            svars['mouth_prim'] = 'circle'
        elif cvars['mouth'] == 'no':
            svars['mouth_prim'] = None

        if cvars['eyes'] == 'green':
            svars['eye_color'] = 'green'
        elif cvars['eyes'] == 'blue':
            svars['eye_color'] = 'blue'

        if cvars['paws'] == 'yes':
            svars['paw_prim'] = 'circle'
        elif cvars['paws'] == 'no':
            svars['paw_prim'] = None


        return svars

    Cat = Concept(
        'cat',
        cat_cat_vars,
        cat_sample_var_fn
    )

    Cat.add_part(Part(
        name = 'head',
        prim_info=('static', 'circle'),
        size_info=(
            ('uni', .5, .75),
            ('uni', .5, .75),
        ),
        loc_info=(
            ('static', 0.),
            ('static', 0.),
        ),
        color_info=('static', 'grey'),
        sem_info='head'
    ))

    Cat.add_part(Part(
        name = 'eye',
        prim_info=('static', 'circle'),
        size_info=(
            ('uni', .1, .15),
            ('uni', .1, .15),
        ),
        loc_info=(
            ('prel', 'left', ('head', 'center_width'), ('uni', 1.5, 2.0)),
            ('prel', 'bot', ('head', 'center_height'), ('uni', .75, 1.5)),            
        ),
        color_info=('ref', 'share', 'eye_color'),
        sem_info='eye',
        top_info=('static', ('symReflect', 'AX'))
    ))

    Cat.add_part(Part(
        name = 'ear',
        prim_info=('static', 'triangle'),
        size_info=(
            ('uni', .15, .2),
            ('uni', .15, .2),
        ),
        loc_info=(
            ('prel', 'left', ('eye', 'center_width'), ('uni', 1., 1.5)),
            ('prel', 'bot', ('head', 'top'), ('uni', .25, .5)),            
        ),
        color_info=('ref', 'share', 'eye_color'),
        sem_info='ear',
        top_info=('static', ('symReflect', 'AX'))
    ))

    Cat.add_part(Part(
        name = 'nose',
        prim_info=('static', 'triangle'),
        size_info=(
            ('uni', .05, .1),
            ('uni', .1, .15),
        ),
        loc_info=(
            ('static', 0.),
            ('uni', -.05, .05),            
        ),
        color_info=('static', 'red'),
        sem_info='nose',
    ))

    Cat.add_part(Part(
        name = 'mouth',
        prim_info=('ref', 'share', 'mouth_prim'),
        size_info=(
            ('uni', .1, .2),
            ('uni', .05, .1),
        ),
        loc_info=(
            ('static', 0.),
            ('prel', 'bot', ('head', 'bot'), ('static', 1.5))
        ),
        color_info=('static', 'red'),
        sem_info='mouth',
    ))

    Cat.add_part(Part(
        name = 'whisk',
        prim_info=('ref', 'share', 'whisk_prim'),
        size_info=(
            ('uni', .15, .25),
            ('static', .05),
        ),
        loc_info=(
            ('rel', 'left', ('eye', 'center_width')),
            ('prel', 'top', ('nose', 'bot'), ('uni', 1.1, 1.25))
        ),
        color_info=('static', 'red'),
        sem_info='whisker',
        top_info=('static', ('symReflect', 'AX'))
    ))

    Cat.add_part(Part(
        name = 'paw',
        prim_info=('ref', 'share', 'paw_prim'),
        size_info=(
            ('uni', .1, .2),
            ('static', .1),
        ),
        loc_info=(
            ('expr', 'div', ('ref', 'head', 'width'), ('static', 1.5)),
            ('rel', 'top', ('head', 'bot')),
        ),
        color_info=('static', 'grey'),
        sem_info='body',
        top_info=('static', ('symReflect', 'AX'))
    ))

    return Cat
    
def make_turtle():
    
    turtle_cat_vars = {
        'spot': ('one', 'multi'),
        'tail': ('circle', 'triangle'),
        'legs': ('yes', 'no'),
        'spotcolor': ('red', 'blue'),
    }

    def turtle_sample_var_fn(cvars):
        svars = {}

        svars['one_spot_prim'] = None
        svars['multi_spot_prim'] = None
        
        if cvars['spot'] == 'one':
            svars['one_spot_prim'] = 'circle'
        elif cvars['spot'] == 'multi':
            svars['multi_spot_prim'] = 'circle'
        else:
            assert False

        if cvars['tail'] == 'circle':
            svars['tail_prim'] = 'circle'
        elif cvars['tail'] == 'triangle':
            svars['tail_prim'] = 'triangle'

        svars['top_leg_prim'] = None
        svars['bot_leg_prim'] = None

        if cvars['legs'] == 'yes':
            mode = random.choice(['top', 'bot', 'both'])
            if mode in ('top', 'both'):
                svars['top_leg_prim'] = 'circle'

            if mode in ('bot', 'both'):
                svars['bot_leg_prim'] = 'circle'
                        
        if cvars['spotcolor'] == 'red':
            svars['spot_color'] = 'red'
        elif cvars['spotcolor'] == 'blue':
            svars['spot_color'] = 'blue'


        return svars

    Turtle = Concept(
        'turtle',
        turtle_cat_vars,
        turtle_sample_var_fn
    )

    Turtle.add_part(Part(
        name = 'top_leg',
        prim_info=('ref', 'share', 'top_leg_prim'),
        size_info=(
            ('uni', .1, .15),
            ('uni', .1, .15),
        ),
        loc_info=(
            ('uni', .25, .4),
            ('uni', .25, .4),
        ),
        color_info=('static', 'green'),
        sem_info='leg',
        top_info=('static', ('symReflect', 'AX'))
    ))

    Turtle.add_part(Part(
        name = 'bot_leg',
        prim_info=('ref', 'share', 'bot_leg_prim'),
        size_info=(
            ('uni', .1, .15),
            ('uni', .1, .15),
        ),
        loc_info=(
            ('ref', 'top_leg', 'x_pos'),
            ('expr', 'mul', ('ref', 'top_leg', 'x_pos'), ('static', -1.0))
        ),
        color_info=('static', 'green'),
        sem_info='leg',
        top_info=('static', ('symReflect', 'AX'))
    ))
    
    
    Turtle.add_part(Part(
        name = 'body',
        prim_info=('static', 'circle'),
        size_info=(
            ('expr', 'mul', ('ref', 'top_leg', 'x_pos'), ('uni', 1.2, 1.5)),
            ('expr', 'mul', ('ref', 'top_leg', 'y_pos'), ('uni', 1.2, 1.5)),
        ),
        loc_info=(
            ('static', 0.),
            ('static', 0.),
        ),
        color_info=('static', 'grey'),
        sem_info='body'
    ))

    Turtle.add_part(Part(
        name = 'head',
        prim_info=('static', 'circle'),
        size_info=(
            ('uni', .15, .25),
            ('uni', .15, .25),
        ),
        loc_info=(
            ('static', 0.),
            ('rel', 'bot', ('body', 'top')),
        ),
        color_info=('static', 'green'),
        sem_info='head'
    ))

    Turtle.add_part(Part(
        name = 'tail',
        prim_info=('ref', 'share', 'tail_prim'),
        size_info=(
            ('uni', .1, .15),
            ('uni', .1, .2),
        ),
        loc_info=(
            ('static', 0.),
            ('rel', 'top', ('body', 'bot')),
        ),
        color_info=('static', 'green'),
        sem_info='tail'
    ))

    Turtle.add_part(Part(
        name = 'one_spot',
        prim_info=('ref', 'share', 'one_spot_prim'),
        size_info=(
            ('uni', .1, .3),
            ('uni', .1, .3),
        ),
        loc_info=(
            ('uni', -.1, .1),
            ('uni', -.1, .1),
        ),
        color_info=('ref', 'share', 'spot_color'),
        sem_info='spot',
    ))

    Turtle.add_part(Part(
        name = 'multi_spot',
        prim_info=('ref', 'share', 'multi_spot_prim'),
        size_info=(
            ('uni', .05, .2),
            ('uni', .05, .2),
        ),
        loc_info=(
            ('uni', .1, .2),
            ('uni', .1, .2),
        ),
        color_info=('ref', 'share', 'spot_color'),
        sem_info='spot',
        top_info = ('static', ('symRotate', str(2)))
    ))

    return Turtle
                        
def make_mouse():

    mouse_cat_vars = {
        'feet': ('yes', 'no'),
        'tail': ('yes', 'no'),
        'inner': ('yes', 'no'),
        'eyes': ('green', 'blue')
    }

    def mouse_sample_var_fn(cvars):
        svars = {}

        svars['nose_prim'] = random.choice(['circle', 'square', 'triangle'])
        
        if cvars['feet'] == 'yes':
            svars['feet_prim'] = random.choice(['circle', 'triangle'])
        elif cvars['feet'] == 'no':
            svars['feet_prim'] = None

        if cvars['tail'] == 'yes':
            svars['tail_prim'] = 'circle'
        elif cvars['tail'] == 'no':
            svars['tail_prim'] = None

        if cvars['inner'] == 'yes':
            svars['inner_prim'] = 'circle'
        elif cvars['inner'] == 'no':
            svars['inner_prim'] = None

        if cvars['eyes'] == 'blue':
            svars['eye_color'] = 'blue'
        elif cvars['eyes'] == 'green':
            svars['eye_color'] = 'green'

        return svars

    Mouse = Concept(
        'mouse',
        mouse_cat_vars,
        mouse_sample_var_fn
    )

    Mouse.add_part(Part(
        name = 'body',
        prim_info=('static', 'circle'),
        size_info=(
            ('uni', .45, .6),
            ('uni', .15, .2),
        ),
        loc_info=(
            ('uni', .1, .2),
            ('uni', -.25, -.4)
        ),
        color_info=('static', 'grey'),            
        sem_info='body'
    ))
    Mouse.add_part(Part(
        name = 'head',
        prim_info=('static', 'circle'),
        size_info=(
            ('uni', .2, .25),
            ('uni', .25, .3)
        ),
        loc_info=(
            ('expr', 'add', ('rel', 'right', ('body', 'left')), ('static', 0.05)),
            ('rel', 'center', ('body', 'top'))
        ),
        color_info=('static', 'grey'),            
        sem_info='body'
    ))
    Mouse.add_part(Part(
        name = 'nose',
        prim_info=('ref', 'share', 'nose_prim'),
        size_info=(
            ('static', .05),
            ('static', .05),
        ),
        loc_info=(
            ('rel', 'center', ('head', 'left')),
            ('rel', 'center', ('head', 'center_height'))
        ),
        color_info=('static', 'red'),            
        sem_info='nose'
    ))
    Mouse.add_part(Part(
        name = 'ear',
        prim_info=('static', 'circle'),
        size_info=(
            ('uni', .2, .25),
            ('uni', .3, .4),
        ),
        loc_info=(
            ('expr', 'add', ('rel', 'left', ('head', 'center_width')), ('static', 0.05) ),
            ('expr', 'add', ('rel', 'bot', ('head', 'center_height')), ('static', 0.05))
        ),
        color_info=('static', 'grey'),            
        sem_info='body'
    ))
    Mouse.add_part(Part(
        name = 'eye',
        prim_info=('static', 'circle'),
        size_info=(
            ('static', .1),
            ('static', .1),
        ),
        loc_info=(
            ('rel', 'right', ('head', 'center_width')),
            ('rel', 'bot', ('head', 'center_height'))
        ),
        color_info=('ref', 'share', 'eye_color'),
        sem_info='eye'
    ))

    Mouse.add_part(Part(
        name = 'inner_ear',
        prim_info=('ref', 'share', 'inner_prim'),
        size_info=(
            ('expr', 'sub', ('ref', 'ear', 'width'), ('static', 0.1)),
            ('expr', 'sub', ('ref', 'ear', 'height'), ('static', 0.1)),
        ),
        loc_info=(
            ('expr', 'sub', ('rel', 'center', ('ear', 'center_width')), ('static', 0.0)),
            ('expr', 'sub', ('rel', 'center', ('ear', 'center_height')), ('static', 0.05)),
        ),
        color_info=('static', 'red'),
        sem_info='ear'
    ))

    Mouse.add_part(Part(
        name = 'tail',
        prim_info=('ref', 'share', 'tail_prim'),
        size_info=(
            ('uni', .1, .15),
            ('uni', .1, .15)
        ),
        loc_info=(
            ('rel', 'center', ('body', 'right')),
            ('expr', 'add', ('rel', 'bot', ('body', 'center_height')), ('static', 0.05))
        ),
        color_info=('static', 'grey'),
        sem_info='tail'
    ))

    Mouse.add_part(Part(
        name = 'left_foot',
        prim_info=('ref', 'share', 'feet_prim'),
        size_info=(
            ('uni', .05, .1),
            ('static', .05),
        ),
        loc_info=(
            ('uni', -.25, -.15),
            ('rel', 'center', ('body', 'bot')),
        ),
        color_info=('static', 'red'),
        sem_info='feet',
        top_info=('static', ('move', 'mxtype_175', 'mytype_0', f'mflt_3' , 'mflt_3', 'symReflect', 'AX'))
    ))

    return Mouse
        
        
    

def make_ladybug():

    ladybug_cat_vars = {
        'dotnum': ('four', 'six'),
        'arms': ('yes', 'no'),
        'eyes': ('yes', 'no'),
        'centerdot': ('yes', 'no')
    }

    def ladybug_sample_var_fn(cvars):
        svars = {}

        svars['four_dot_prim'] = None
        svars['six_dot_prim'] = None
        
        if cvars['dotnum'] == 'four':
            svars['four_dot_prim'] = 'circle'
        elif cvars['dotnum'] == 'six':
            svars['six_dot_prim'] = 'circle'
        else:
            assert False
            
        if cvars['arms'] == 'yes':
            svars['arm_prim'] = random.choice(['square', 'circle'])
        elif cvars['arms'] == 'no':
            svars['arm_prim'] = None

        if cvars['eyes'] == 'yes':
            svars['eye_prim'] = 'circle'
        elif cvars['eyes'] == 'no':
            svars['eye_prim'] = None

        if cvars['centerdot'] == 'yes':
            svars['center_dot_prim'] = 'circle'
        elif cvars['centerdot'] == 'no':
            svars['center_dot_prim'] = None
            
        return svars

    Ladybug = Concept(
        'ladybug',
        ladybug_cat_vars,
        ladybug_sample_var_fn
    )

    Ladybug.add_part(Part(
        name = 'body',
        prim_info=('static', 'circle'),
        size_info=(
            ('uni', .5, .7),
            ('uni', .4, .6),
        ),
        loc_info=(
            ('static', 0.),
            ('static', 0.)
        ),
        color_info=('static', 'red'),            
        sem_info='body'
    ))

    Ladybug.add_part(Part(
        name = 'head',
        prim_info=('static', 'circle'),
        size_info=(
            ('uni', .15, .25),
            ('uni', .15, .25),
        ),
        loc_info=(
            ('static', 0.),
            ('prel', 'bot', ('body', 'top'), ('uni', 0.5, 1.0))
        ),
        color_info=('static', 'red'),            
        sem_info='body'
    ))

    Ladybug.add_part(Part(
        name = 'tail',
        prim_info=('static', 'circle'),
        size_info=(
            ('uni', .15, .25),
            ('uni', .05, .1)
        ),
        loc_info=(
            ('static', 0.),
            ('prel', 'top', ('body', 'bot'), ('uni', 0.5, 1.0))
        ),
        color_info=('static', 'grey'),            
        sem_info='tail'
    ))

    Ladybug.add_part(Part(
        name = 'center_dot',
        prim_info=('ref', 'share', 'center_dot_prim'),
        size_info=(
            ('uni', .05, .15),
            ('uni', .05, .15)
        ),
        loc_info=(
            ('static', 0.),
            ('static', 0.),            
        ),
        color_info=('static', 'grey'),            
        sem_info='dot'
    ))

    Ladybug.add_part(Part(
        name = 'four_dot',
        prim_info=('ref', 'share', 'four_dot_prim'),
        size_info=(
            ('uni', .05, .1),
            ('uni', .05, .1)
        ),
        loc_info=(
            ('uni', .15, .25),
            ('uni', .15, .25),            
        ),
        color_info=('static', 'grey'),            
        sem_info='dot',
        top_info=('static', ('symRotate', str(3)))
    ))

    Ladybug.add_part(Part(
        name = 'six_dot',
        prim_info=('ref', 'share', 'six_dot_prim'),
        size_info=(
            ('uni', .05, .1),
            ('uni', .05, .1)
        ),
        loc_info=(
            ('uni', .15, .25),
            ('uni', .15, .25),            
        ),
        color_info=('static', 'grey'),            
        sem_info='dot',
        top_info=('static', ('symRotate', str(5)))
    ))

    
    Ladybug.add_part(Part(
        name = 'eye',
        prim_info=('ref', 'share', 'eye_prim'),
        size_info=(
            ('uni', .05, .1),
            ('uni', .05, .1),
        ),
        loc_info=(
            ('prel', 'left', ('head', 'center_width'), ('static', 1.5)),
            ('rel', 'center', ('head', 'top')),
        ),
        color_info=('static', 'green'),            
        sem_info='eye',
        top_info=('static', ('symReflect', 'AX'))
    ))

    Ladybug.add_part(Part(
        name = 'top_arm',
        prim_info=('ref', 'share', 'arm_prim'),
        size_info=(
            ('uni', .1, .2),
            ('static', .05),
        ),
        loc_info=(
            ('expr', 'div', ('ref', 'body', 'width'), ('uni', 1.0, 1.2)),
            ('expr', 'div', ('ref', 'body', 'height'), ('uni', 1.75, 2)),
        ),
        color_info=('static', 'blue'),            
        sem_info='arm',
        top_info=('static', ('symReflect', 'AX'))
    ))

    Ladybug.add_part(Part(
        name = 'bot_arm',
        prim_info=('ref', 'share', 'arm_prim'),
        size_info=(
            ('uni', .1, .2),
            ('static', .05),
        ),
        loc_info=(
            ('expr', 'div', ('ref', 'body', 'width'), ('uni', 1.0, 1.2)),
            ('expr', 'div', ('ref', 'body', 'height'), ('uni', -2., -1.75)),
        ),
        color_info=('static', 'blue'),            
        sem_info='arm',
        top_info=('static', ('symReflect', 'AX'))
    ))

    return Ladybug
