import random
from make_data import Concept, Part, _get_uni_val

def make_fridge():
    fridge_cat_vars = {
        'top': ('yes', 'no'),
        'half': ('yes', 'no'),
        'ice': ('yes', 'no'),
        'handles': ('yes', 'no'),        
    }
    
    
    def fridge_sample_var_fn(cvars):

        svars = {}

        if cvars['top'] == 'yes':
            svars['top_prim'] = 'square'
        elif cvars['top'] == 'no':
            svars['top_prim'] = None

        if cvars['half'] == 'yes':
            svars['vert_div_prim'] = 'square'            
        elif cvars['half'] == 'no':
            svars['vert_div_prim'] = None

        if cvars['ice'] == 'yes':
            svars['ice_prim'] = 'square'
        elif cvars['ice'] == 'no':
            svars['ice_prim'] = None
            
        svars['center_handle_prim'] = None
        svars['bot_side_handle_prim'] = None
        svars['top_side_handle_prim'] = None

        handle_prim = 'square'
        
        if cvars['handles'] == 'yes':
            if cvars['top'] == 'yes':
                svars['top_side_handle_prim'] = handle_prim

            if cvars['half'] == 'yes':
                svars['center_handle_prim'] = handle_prim
            if cvars['half'] == 'no':
                svars['bot_side_handle_prim'] = handle_prim
                            
        return svars

    fridge = Concept(
        'fridge',
        fridge_cat_vars,
        fridge_sample_var_fn
    )

    fridge.add_part(Part(
        name = 'body_bot',
        prim_info= ('static', 'square'),
        size_info= (
            ('uni', .5, .75),
            ('uni', .45, .7),
        ),
        loc_info= (
            ('static', 0.),
            ('uni', -.3, -.1),
        ),
        color_info=('static', 'grey'),
        sem_info='body'
    ))

    fridge.add_part(Part(
        name = 'body_top',
        prim_info= ('ref', 'share', 'top_prim'),
        size_info= (
            ('ref', 'body_bot', 'width'),
            ('uni', .2, .3),
        ),
        loc_info= (
            ('static', 0.),
            ('rel', 'bot', ('body_bot', 'top')),
        ),
        color_info=('static', 'grey'),
        sem_info='body'
    ))

    fridge.add_part(Part(
        name = 'horiz_div',
        prim_info= ('ref', 'share', 'top_prim'),
        size_info= (
            ('ref', 'body_bot', 'width'),
            ('static', 0.05),
        ),
        loc_info= (            
            ('static', 0.),
            ('rel', 'center', ('body_top', 'bot')),
        ),
        color_info=('static', 'green'),
        sem_info='div'
    ))

    fridge.add_part(Part(
        name = 'vert_div',
        prim_info= ('ref', 'share', 'vert_div_prim' ),
        size_info= (
            ('static', 0.05),
            ('ref', 'body_bot', 'height'),
        ),
        loc_info= (
            ('uni', -0.05, 0.05),
            ('rel', 'center', ('body_bot', 'center_height')),
        ),
        color_info=('static', 'green'),
        sem_info='div'
    ))

    fridge.add_part(Part(
        name = 'center_right_handle',
        prim_info= ('ref', 'share', 'center_handle_prim'),
        size_info= (
            ('static', 0.05),
            ('uni', .1, .2),
        ),
        loc_info= (
            ('expr', 'add', ('rel', 'left', ('vert_div', 'right')), ('static', 0.05)),
            ('uni',-.15, -.25),
        ),
        color_info=('static', 'red'),
        sem_info='handle',
    ))

    fridge.add_part(Part(
        name = 'center_left_handle',
        prim_info= ('ref', 'share', 'center_handle_prim'),
        size_info= (
            ('static', 0.05),
            ('ref', 'center_right_handle', 'height')
        ),
        loc_info= (
            ('expr', 'add', ('rel', 'right', ('vert_div', 'left')), ('static', -0.05)),
            ('ref', 'center_right_handle', 'y_pos')
        ),
        color_info=('static', 'red'),
        sem_info='handle',
    ))

    fridge.add_part(Part(
        name = 'bot_side_handle',
        prim_info= ('ref', 'share', 'bot_side_handle_prim'),
        size_info= (
            ('static', 0.05),
            ('uni', .1, .15),
        ),
        loc_info= (
            ('expr', 'add', ('rel', 'left', ('body_bot', 'left')), ('static', 0.05)),
            ('uni',-.15, -.25),
        ),
        color_info=('static', 'red'),
        sem_info='handle',
    ))

    fridge.add_part(Part(
        name = 'top_side_handle',
        prim_info= ('ref', 'share', 'top_side_handle_prim'),
        size_info= (
            ('static', 0.1),
            ('uni', .1, .15),
        ),
        loc_info= (
            ('expr', 'add', ('rel', 'left', ('body_bot', 'left')), ('uni', 0.05, 0.15)),
            ('rel', 'center', ('body_top', 'center_height')),
        ),
        color_info=('static', 'red'),
        sem_info='handle',
    ))

    fridge.add_part(Part(
        name = 'ice_box',
        prim_info= ('ref', 'share', 'ice_prim'),
        size_info= (
            ('static', 0.15),
            ('static', .1),
        ),
        loc_info= (
            ('expr', 'sub', ('rel', 'right', ('body_bot', 'center_width')), ('uni', 0.1, 0.2)),
            ('prel', 'top', ('horiz_div', 'bot'), ('static', 1.25)),            
        ),
        color_info=('static', 'blue'),
        sem_info='ice',
    ))
    
    return fridge


def make_microwave():
    microwave_cat_vars = {
        'handle': ('yes', 'no'),
        'feet': ('yes', 'no'),
        'tray': ('yes', 'no'),
        'button': ('double', 'triple', 'single')
    }

    def microwave_sample_var_fn(cvars):
        svars = {}

        svars['single_button_prim'] = None
        svars['double_button_prim'] = None
        svars['triple_button_prim'] = None        
        
        if cvars['button'] == 'triple':
            svars['triple_button_prim'] = random.choice(['triangle'])
        elif cvars['button'] == 'double':
            svars['double_button_prim'] = random.choice(['circle'])
        elif cvars['button'] == 'single':
            svars['single_button_prim'] = random.choice(['square'])

            
        if cvars['handle'] == 'yes':
            svars['handle_prim'] = 'square'
        elif cvars['handle'] == 'no':
            svars['handle_prim'] = None

        if cvars['feet'] == 'yes':
            svars['feet_prim'] = random.choice(['square', 'circle', 'triangle'])
        elif cvars['feet'] == 'no':
            svars['feet_prim'] = None

        if cvars['tray'] == 'yes':
            svars['plate_prim'] = 'circle'
        elif cvars['tray'] == 'no':
            svars['plate_prim'] = None

        svars['feet_color'] = random.choice(['red', 'blue', 'green', 'grey'])        
        
        return svars

    microwave = Concept(
        'microwave',
        microwave_cat_vars,
        microwave_sample_var_fn
    )

    microwave.add_part(Part(
        name = 'body',
        prim_info= ('static', 'square'),
        size_info= (
            ('uni', .8, .9),
            ('uni', .4, .8),
        ),
        loc_info= (
            ('static', 0.),
            ('static', 0.),
        ),
        color_info=('static', 'grey'),
        sem_info='body',
    ))

    microwave.add_part(Part(
        name = 'window',
        prim_info= ('static', 'square'),
        size_info= (
            ('expr', 'div', ('ref', 'body', 'width'), ('uni', 1.6, 1.8)),
            ('expr', 'sub', ('ref', 'body', 'height'), ('static', 0.1)),
        ),
        loc_info= (
            ('expr', 'add', ('rel', 'left', ('body', 'left')), ('static', 0.1)),
            ('static', 0.),
        ),
        color_info=('static', 'blue'),
        sem_info='window',
    ))

    microwave.add_part(Part(
        name = 'plate',
        prim_info= ('ref', 'share', 'plate_prim'),
        size_info= (
            ('expr', 'sub', ('ref', 'window', 'width'), ('static', 0.05)),
            ('uni', 0.05, 0.1),
        ),
        loc_info= (
            ('rel', 'center', ('window', 'center_width')), 
            ('expr', 'add', ('rel', 'bot', ('window', 'bot')), ('static', 0.05)),            
        ),
        color_info=('static', 'red'),
        sem_info='plate',
    ))

    microwave.add_part(Part(
        name = 'handle',
        prim_info= ('ref', 'share', 'handle_prim'),
        size_info= (
            ('static', 0.1),
            ('expr', 'sub', ('ref', 'window', 'height'), ('uni', 0.05, 0.1)),
        ),
        loc_info= (
            ('expr', 'add', ('rel', 'left', ('window', 'right')), ('static', 0.05)),
            ('rel', 'center', ('window', 'center_height')),
        ),
        color_info=('static', 'red'),
        sem_info='handle',
    ))

    microwave.add_part(Part(
        name = 'feet',
        prim_info= ('ref', 'share', 'feet_prim'),
        size_info= (
            ('uni', 0.05, 0.15),
            ('uni', 0.05, 0.1)
        ),
        loc_info= (
            ('uni', 0.2, 0.4),
            ('rel', 'top', ('body', 'bot'))
        ),
        color_info=('ref', 'share', 'feet_color'),
        sem_info='feet',
        top_info=('static', ('symReflect', 'AX'))
    ))

    microwave.add_part(Part(
        name = 'single_button',
        prim_info= ('ref', 'share', 'single_button_prim'),
        size_info= (
            ('uni', 0.1, 0.15),
            ('uni', 0.15, 0.3)
        ),
        loc_info= (
            ('expr', 'sub', ('rel', 'right', ('body', 'right')), ('static', 0.05)),
            ('expr', 'add', ('rel', 'bot', ('body', 'bot')), ('uni', 0.05, 0.15))             
        ),
        color_info=('static', 'green'),
        sem_info='button',
    ))

    microwave.add_part(Part(
        name = 'double_button',
        prim_info= ('ref', 'share', 'double_button_prim'),
        size_info= (
            ('uni', 0.05, 0.1),
            ('uni', 0.05, 0.1)
        ),
        loc_info= (
            ('expr', 'sub', ('rel', 'right', ('body', 'right')), ('static', 0.05)),
            ('expr', 'sub', ('rel', 'top', ('body', 'top')), ('static', 0.2))                         
        ),
        color_info=('static', 'green'),
        sem_info='button',
        top_info=('static', ('symTranslate', 0., -0.4, 1)),
    ))
    
    microwave.add_part(Part(
        name = 'triple_button',
        prim_info= ('ref', 'share', 'triple_button_prim'),
        size_info= (
            ('uni', 0.05, 0.1),
            ('uni', 0.05, 0.1)
        ),
        loc_info= (
            ('expr', 'sub', ('rel', 'right', ('body', 'right')), ('static', 0.05)),
            ('expr', 'sub', ('rel', 'top', ('body', 'top')), ('static', 0.1))                         
        ),
        color_info=('static', 'green'),
        sem_info='button',
        top_info=('static', ('symTranslate', 0., -0.5, 2)),
    ))

    return microwave


def make_clock():
    clock_cat_vars = {
        'frame': ('yes', 'no'),
        'feet': ('yes', 'no'),
        'bells': ('yes', 'no'),
        'hands': ('one', 'diff', 'same')
    }

    def clock_sample_var_fn(cvars):
        svars = {}

        svars['frame_color'] = random.choice(['green', 'blue', 'red'])
        
        if cvars['frame'] == 'yes':
            svars['frame_prim'] = 'circle'            
            svars['frame_offset'] = 0.1
        elif cvars['frame'] == 'no':
            svars['frame_prim'] = None
            svars['frame_offset'] = 0.0
            
        if cvars['feet'] == 'yes':
            svars['feet_prim'] = random.choice(['square', 'circle', 'triangle'])
        elif cvars['feet'] == 'no':
            svars['feet_prim'] = None

        if cvars['bells'] == 'yes':
            svars['bell_prim'] = 'triangle'
        elif cvars['bells'] == 'no':
            svars['bell_prim'] = None

        svars['horiz_hand_color'] = random.choice(['green', 'blue', 'red'])
        svars['vert_hand_color'] = random.choice(['green', 'blue', 'red'])
            
        if cvars['hands'] == 'same':
            svars['vert_hand_color'] = svars['horiz_hand_color']
            
        elif cvars['hands'] == 'diff':
            while svars['horiz_hand_color'] == svars['vert_hand_color']:
                svars['vert_hand_color'] = random.choice(['green', 'blue', 'red'])

        svars['top_hand_prim'] = None
        svars['bot_hand_prim'] = None
        svars['left_hand_prim'] = None
        svars['right_hand_prim'] = None

        vkey = random.choice(['top_hand_prim', 'bot_hand_prim'])
        hkey = random.choice(['left_hand_prim', 'right_hand_prim'])

        svars[vkey] = 'square'
        svars[hkey] = 'square'

        if cvars['hands'] == 'one':            
            svars[random.choice([hkey, vkey])] = None
        
        return svars

    clock = Concept(
        'clock',
        clock_cat_vars,
        clock_sample_var_fn
    )

    clock.add_part(Part(
        name = 'frame',
        prim_info = ('ref', 'share', 'frame_prim'),
        size_info = (
            ('uni', 0.5, 0.8),
            ('uni', 0.5, 0.8),
        ),
        loc_info = (
            ('static', 0.),
            ('static', 0.),
        ),
        color_info=('ref', 'share', 'frame_color'),
        sem_info='frame',                
    ))

    clock.add_part(Part(
        name = 'feet',
        prim_info = ('ref', 'share', 'feet_prim'),
        size_info = (
            ('uni', 0.15, 0.25),
            ('uni', 0.1, 0.15),            
        ),
        loc_info = (
            ('expr', 'div', ('ref', 'frame', 'width'), ('uni', 1.8, 2.)),
            ('expr', 'div', ('ref', 'frame', 'height'), ('uni', -1.25, -1.0)),
        ),
        color_info=('ref', 'share', 'frame_color'),
        sem_info='feet',
        top_info= ('static', ('symReflect', 'AX'))
    ))
    
    clock.add_part(Part(
        name = 'body',
        prim_info = ('static', 'circle'),
        size_info = (
            ('expr', 'sub', ('ref', 'frame', 'width'), ('ref', 'share', 'frame_offset')),
            ('expr', 'sub', ('ref', 'frame', 'height'), ('ref', 'share', 'frame_offset')),            
        ),
        loc_info = (
            ('static', 0.),
            ('static', 0.),
        ),
        color_info=('static', 'grey'),
        sem_info='body',                
    ))

    clock.add_part(Part(
        name = 'bells',
        prim_info = ('ref', 'share', 'bell_prim'),
        size_info = (
            ('uni', 0.2, 0.3),
            ('uni', 0.2, 0.3),            
        ),
        loc_info = (
            ('expr', 'div', ('ref', 'frame', 'width'), ('uni', 1.2, 1.4)),
            ('expr', 'div', ('ref', 'frame', 'height'), ('uni', 1.2, 1.4)),
        ),
        color_info=('ref', 'share', 'frame_color'),
        sem_info='bell',
        top_info=('static', ('symReflect', 'AX'))
    ))

    clock.add_part(Part(
        name = 'right_hand',
        prim_info = ('ref', 'share', 'right_hand_prim'),
        size_info = (
            ('uni', 0.1, 0.25),
            ('static', 0.05)
        ),
        loc_info = (
            ('expr', 'div', ('ref', 'right_hand', 'width'), ('static', .9)),
            ('static', 0.0),
        ),
        color_info=('ref', 'share', 'horiz_hand_color'),
        sem_info='hand',
    ))

    clock.add_part(Part(
        name = 'left_hand',
        prim_info = ('ref', 'share', 'left_hand_prim'),
        size_info = (
            ('uni', 0.1, 0.25),
            ('static', 0.05)
        ),
        loc_info = (
            ('expr', 'div', ('ref', 'left_hand', 'width'), ('static', -.9)),
            ('static', 0.0),
        ),
        color_info=('ref', 'share', 'horiz_hand_color'),
        sem_info='hand',
    ))

    clock.add_part(Part(
        name = 'top_hand',
        prim_info = ('ref', 'share', 'top_hand_prim'),
        size_info = (
            ('static', 0.05),
            ('uni', 0.1, 0.25),            
        ),
        loc_info = (
            ('static', 0.0),
            ('expr', 'div', ('ref', 'top_hand', 'height'), ('static', .9)),            
        ),
        color_info=('ref', 'share', 'vert_hand_color'),
        sem_info='hand',
    ))

    clock.add_part(Part(
        name = 'bot_hand',
        prim_info = ('ref', 'share', 'bot_hand_prim'),
        size_info = (
            ('static', 0.05),
            ('uni', 0.1, 0.25),            
        ),
        loc_info = (
            ('static', 0.0),
            ('expr', 'div', ('ref', 'bot_hand', 'height'), ('static', -.9)),            
        ),
        color_info=('ref', 'share', 'vert_hand_color'),
        sem_info='hand',
    ))
    
    return clock

def make_car():
    car_cat_vars = {
        'window': ('yes', 'no'),
        'top': ('yes', 'no'),
        'hub': ('yes', 'no'),
        'road': ('yes', 'no')        
    }

    def car_sample_var_fn(cvars):
        svars = {}

        svars['car_color'] = random.choice(['red', 'green', 'blue'])
        
        svars['wheel_color'] = 'grey'

        svars['hub_color'] = random.choice(['red', 'green', 'blue'])
        while svars['hub_color'] == svars['car_color']:
            svars['hub_color'] = random.choice(['red', 'green', 'blue'])        
        
        if cvars['window'] == 'yes':
            svars['window_prim'] = 'square'
        elif cvars['window'] == 'no':
            svars['window_prim'] = None

        if cvars['hub'] == 'yes':
            svars['hub_prim'] = 'circle'
        elif cvars['hub'] == 'no':
            svars['hub_prim'] = None

        if cvars['top'] == 'yes':
            svars['top_prim'] = random.choice(['triangle', 'square', 'circle'])
        elif cvars['top'] == 'no':
            svars['top_prim'] = None

        if cvars['road'] == 'yes':
            svars['road_prim'] = 'square'
        elif cvars['road'] == 'no':
            svars['road_prim'] = None
            
        return svars

    car = Concept(
        'car',
        car_cat_vars,
        car_sample_var_fn
    )

    car.add_part(Part(
        name = 'body_low',
        prim_info = ('static', 'square'),
        size_info = (
            ('uni', 0.75, .9),
            ('uni', 0.2, 0.25),
        ),
        loc_info = (
            ('static', 0.),
            ('expr', 'mul', ('ref', 'body_low', 'height'), ('static', -1.0)),
        ),
        color_info=('ref', 'share', 'car_color'),
        sem_info='body',                
    ))

    car.add_part(Part(
        name = 'body_top',
        prim_info = ('static', 'square'),
        size_info = (
            ('uni', 0.5, .7),
            ('uni', 0.25, 0.35),
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'bot', ('body_low', 'top')),
        ),
        color_info=('ref', 'share', 'car_color'),
        sem_info='body',                
    ))

    car.add_part(Part(
        name = 'top',
        prim_info = ('ref', 'share', 'top_prim'),
        size_info = (
            ('static',.1),
            ('static', 0.1),
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'bot', ('body_top', 'top')),
        ),
        color_info=('static', 'grey'),
        sem_info='top',                
    ))

    car.add_part(Part(
        name = 'front_window',
        prim_info = ('static', 'square'),
        size_info = (
            ('uni', 0.15, .2),
            ('expr', 'sub', ('ref', 'body_top', 'height'), ('static', 0.1))
        ),
        loc_info = (
            ('expr', 'sub', ('rel', 'right', ('body_top', 'right')), ('static',.05)),
            ('rel', 'center', ('body_top', 'center_height')),
        ),
        color_info=('static', 'grey'),
        sem_info='window',                
    ))

    car.add_part(Part(
        name = 'both_windows',
        prim_info = ('ref', 'share', 'window_prim'),
        size_info = (
            ('uni', 0.15, .2),
            ('expr', 'sub', ('ref', 'body_top', 'height'), ('static', 0.1))
        ),
        loc_info = (
            ('expr', 'sub', ('rel', 'right', ('body_top', 'right')), ('static',.05)),
            ('rel', 'center', ('body_top', 'center_height')),
        ),
        color_info=('static', 'grey'),
        sem_info='window',
        top_info=('static', ('symTranslate', -.5, 0.0, 1))
    ))
    
    car.add_part(Part(
        name = 'wheel',
        prim_info = ('static', 'circle'),
        size_info = (
            ('uni', 0.25, 0.3),
            ('ref', 'wheel', 'width')
        ),
        loc_info = (
            ('uni', .3, .5),
            ('rel', 'center', ('body_low', 'bot')),
        ),
        color_info=('ref', 'share', 'wheel_color'),
        sem_info='wheel',
        top_info=('static', ('symReflect', 'AX'))
    ))


    car.add_part(Part(
        name = 'hub',
        prim_info = ('ref', 'share', 'hub_prim'),
        size_info = (
            ('expr', 'sub', ('ref', 'wheel', 'width'), ('static', 0.1)),
            ('expr', 'sub', ('ref', 'wheel', 'width'), ('static', 0.1)),
        ),
        loc_info = (
            ('rel', 'center', ('wheel', 'center_width')),
            ('rel', 'center', ('wheel', 'center_height')),
        ),
        color_info=('ref', 'share', 'hub_color'),
        sem_info='hub',
        top_info=('static', ('symReflect', 'AX'))
    ))

    car.add_part(Part(
        name = 'road',
        prim_info = ('ref', 'share', 'road_prim'),
        size_info = (
            ('static', .5),
            ('static', 0.1)
        ),
        loc_info = (
            ('static', 0.5),
            ('rel', 'top', ('wheel', 'bot'))
        ),
        color_info=('static', 'grey'),
        sem_info='road',
        top_info=('static', ('symReflect', 'AX'))
    ))

    return car


def make_plane():
    plane_cat_vars = {
        'wing': ('single', 'double'),
        'engine': ('body', 'wing', 'no'),
        'prop': ('yes', 'no'),
        'body': ('color', 'grey')
    }


    def plane_sample_var_fn(cvars):
        svars = {}

        if cvars['body'] == 'color':
            svars['body_color'] = random.choice(['red', 'green', 'blue'])
            svars['engine_color'] = 'grey'
        elif cvars['body'] == 'grey':
            svars['body_color'] = 'grey'
            svars['engine_color'] = random.choice(['red', 'green', 'blue'])

        svars['double_wing_prim'] = None
        svars['single_wing_prim'] = None

        if cvars['wing'] == 'double':
            svars['double_wing_prim'] = 'triangle'
        elif cvars['wing'] == 'single':
            svars['single_wing_prim'] = 'triangle'
                    
        if cvars['prop'] == 'yes':            
            svars['prop_prim'] = 'circle'
        elif cvars['prop'] == 'no':            
            svars['prop_prim'] = None

        svars['engine_wing_prim'] = None
        svars['engine_body_prim'] = None
        if cvars['engine'] == 'wing':
            svars['engine_wing_prim'] = 'circle'
        elif cvars['engine'] == 'body':
            svars['engine_body_prim'] = 'circle'
            
        svars['double_tail_prim'] = None
        svars['single_tail_prim'] = None

        if random.random() < 0.5:
            svars['double_tail_prim'] = 'triangle'
        else:
            svars['single_tail_prim'] = 'triangle'
            
        return svars

    plane = Concept(
        'plane',
        plane_cat_vars,
        plane_sample_var_fn
    )

    plane.add_part(Part(
        name = 'engine_wing',
        prim_info = ('ref', 'share', 'engine_wing_prim'),
        size_info = (
            ('uni', 0.05, 0.1),
            ('uni', 0.2, 0.4),
        ),
        loc_info = (
            ('uni', .4, .6),
            ('uni', -.2, .2),
        ),
        color_info=('ref', 'share', 'engine_color'),
        sem_info='engine',
        top_info=('static', ('symReflect', 'AX'))
    ))
    
    plane.add_part(Part(
        name = 'single_wing',
        prim_info = ('ref', 'share', 'single_wing_prim'),
        size_info = (
            ('expr', 'mul', ('ref', 'engine_wing', 'x_pos'), ('uni', 1.1, 1.5)),
            ('uni', 0.15, 0.3),
        ),
        loc_info = (
            ('static', 0.),
            ('ref', 'engine_wing', 'y_pos')
        ),
        color_info=('ref', 'share', 'body_color'),
        sem_info='wing',                
    ))

    plane.add_part(Part(
        name = 'double_wing',
        prim_info = ('ref', 'share', 'double_wing_prim'),
        size_info = (
            ('expr', 'mul', ('ref', 'engine_wing', 'x_pos'), ('uni', .55, .75)),
            ('uni', 0.15, 0.3),
        ),
        loc_info = (
            ('ref', 'double_wing', 'width'),
            ('ref', 'engine_wing', 'y_pos')
        ),
        color_info=('ref', 'share', 'body_color'),
        sem_info='wing',
        top_info=('static', ('symReflect', 'AX'))
    ))
    
    plane.add_part(Part(
        name = 'body',
        prim_info = ('static', 'circle'),
        size_info = (
            ('uni', 0.15, 0.25),
            ('uni', 0.65, 0.8),
        ),
        loc_info = (
            ('static', 0.),
            ('static', 0.),
        ),
        color_info=('ref', 'share', 'body_color'),
        sem_info='body',                
    ))

    plane.add_part(Part(
        name = 'engine_body',
        prim_info = ('ref', 'share', 'engine_body_prim'),
        size_info = (
            ('static', 0.05),
            ('uni', 0.15, 0.2),
        ),
        loc_info = (
            ('rel', 'center', ('body', 'right')),
            ('expr', 'add', ('rel', 'center', ('engine_wing', 'bot')), ('static', 0.05))
        ),
        color_info=('ref', 'share', 'engine_color'),
        sem_info='engine',
        top_info=('static', ('symReflect', 'AX'))
    ))

    plane.add_part(Part(
        name = 'prop',
        prim_info = ('ref', 'share', 'prop_prim'),
        size_info = (
            ('uni', 0.2, 0.3),
            ('static', 0.05)
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'bot', ('body', 'top')),
        ),
        color_info=('ref', 'share', 'engine_color'),
        sem_info='prop',                
    ))


    plane.add_part(Part(
        name = 'single_tail',
        prim_info = ('ref', 'share', 'single_tail_prim'),
        size_info = (
            ('uni', 0.2, 0.25),
            ('static', 0.1)
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'top', ('body', 'bot')),
        ),
        color_info=('ref', 'share', 'engine_color'),
        sem_info='tail',                
    ))

    plane.add_part(Part(
        name = 'double_tail',
        prim_info = ('ref', 'share', 'double_tail_prim'),
        size_info = (
            ('uni', 0.2, 0.25),
            ('static', 0.1)
        ),
        loc_info = (
            ('ref', 'double_tail', 'width'),
            ('rel', 'bot', ('body', 'bot')),
        ),
        color_info=('ref', 'share', 'engine_color'),
        sem_info='tail',
        top_info=('static', ('symReflect', 'AX'))
    ))


    return plane

def make_horiz_back():
    horiz_back_cat_vars = {
        'slats': ('trans', 'ref'),
        'top_bar': ('yes', 'no'),
        'bot_bar': ('yes', 'no'),
        'frame': ('grey', 'match')
    }

    def horiz_back_sample_var_fn(cvars):
        svars = {}

        svars['ref_slat_prim'] = None
        svars['trans_slat_prim'] = None
        
        if cvars['slats'] == 'ref':
            svars['ref_slat_prim'] = 'square'
        elif cvars['slats'] == 'trans':
            svars['trans_slat_prim'] = 'square'

        svars['top_bar_ref_prim'] = None
        svars['top_bar_trans_prim'] = None
        svars['bot_bar_ref_prim'] = None
        svars['bot_bar_trans_prim'] = None
        
        if cvars['top_bar'] == 'yes':
            if cvars['slats'] == 'ref':
                svars['top_bar_ref_prim'] = 'square'
            if cvars['slats'] == 'trans':
                svars['top_bar_trans_prim'] = 'square'
                
        if cvars['bot_bar'] == 'yes':
            if cvars['slats'] == 'ref':
                svars['bot_bar_ref_prim'] = 'square'
            if cvars['slats'] == 'trans':
                svars['bot_bar_trans_prim'] = 'square'

        svars['bar_color'] = random.choice(['red', 'green', 'blue'])

        if cvars['frame'] == 'grey':
            svars['frame_color'] = 'grey'
        elif cvars['frame'] == 'match':
            svars['frame_color'] = svars['bar_color']
            
        return svars

    horiz_back = Concept(
        'horiz_back',
        horiz_back_cat_vars,
        horiz_back_sample_var_fn
    )

    horiz_back.add_part(Part(
        name = 'ref_horiz_slats',
        prim_info = ('ref', 'share', 'ref_slat_prim'),
        size_info = (
            ('uni', 0.4, 0.7),
            ('uni', 0.05, 0.1),
        ),
        loc_info = (
            ('static', 0.),
            ('uni',.15, .35),
        ),
        color_info=('ref', 'share', 'bar_color'),
        sem_info='slat',
        top_info = ('static', ('symReflect', 'AY'))
    ))

    horiz_back.add_part(Part(
        name = 'trans_horiz_slats',
        prim_info = ('ref', 'share', 'trans_slat_prim'),
        size_info = (
            ('uni', 0.4, 0.7),
            ('uni', 0.05, 0.1),
        ),
        loc_info = (
            ('static', 0.),
            ('uni',.35, .4),
        ),
        color_info=('ref', 'share', 'bar_color'),
        sem_info='slat',
        top_info = ('fn', ti_trans_horiz_slats)
    ))

    horiz_back.add_part(Part(
        name = 'side_trans_bars',
        prim_info = ('ref', 'share', 'trans_slat_prim'),
        size_info = (
            ('static', 0.1),
            ('expr', 'mul', ('ref', 'trans_horiz_slats', 'y_pos'), ('static', 2.0))
        ),
        loc_info = (
            ('expr', 'mul', ('ref', 'trans_horiz_slats', 'width'), ('static', 1.0)),
            ('static', 0.),
        ),
        color_info=('ref', 'share', 'frame_color'),
        sem_info='frame',
        top_info = ('static', ('symReflect', 'AX'))
    ))

    horiz_back.add_part(Part(
        name = 'side_ref_bars',
        prim_info = ('ref', 'share', 'ref_slat_prim'),
        size_info = (
            ('static', 0.1),
            ('expr', 'mul', ('ref', 'ref_horiz_slats', 'y_pos'), ('static', 2.0))
        ),
        loc_info = (
            ('expr', 'mul', ('ref', 'ref_horiz_slats', 'width'), ('static', 1.0)),
            ('static', 0.),
        ),
        color_info=('ref', 'share', 'frame_color'),
        sem_info='frame',
        top_info = ('static', ('symReflect', 'AX'))
    ))

    horiz_back.add_part(Part(
        name = 'top_ref_bar',
        prim_info = ('ref', 'share', 'top_bar_ref_prim'),
        size_info = (            
            ('expr', 'mul', ('ref', 'side_ref_bars', 'x_pos'), ('static', 1.2)),
            ('uni', 0.05, 0.1)
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'bot', ('side_ref_bars', 'top'))
        ),
        color_info=('ref', 'share', 'frame_color'),
        sem_info='frame',
    ))

    horiz_back.add_part(Part(
        name = 'bot_ref_bar',
        prim_info = ('ref', 'share', 'bot_bar_ref_prim'),
        size_info = (            
            ('expr', 'mul', ('ref', 'side_ref_bars', 'x_pos'), ('static', 1.2)),
            ('ref', 'top_ref_bar', 'height')
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'top', ('side_ref_bars', 'bot'))
        ),
        color_info=('ref', 'share', 'frame_color'),
        sem_info='frame',
    ))

    horiz_back.add_part(Part(
        name = 'top_trans_bar',
        prim_info = ('ref', 'share', 'top_bar_trans_prim'),
        size_info = (            
            ('expr', 'mul', ('ref', 'side_trans_bars', 'x_pos'), ('static', 1.2)),
            ('uni', 0.05, 0.1)
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'bot', ('side_trans_bars', 'top'))
        ),
        color_info=('ref', 'share', 'frame_color'),
        sem_info='frame',
    ))

    horiz_back.add_part(Part(
        name = 'bot_trans_bar',
        prim_info = ('ref', 'share', 'bot_bar_trans_prim'),
        size_info = (            
            ('expr', 'mul', ('ref', 'side_trans_bars', 'x_pos'), ('static', 1.2)),
            ('ref', 'top_trans_bar', 'height')
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'top', ('side_trans_bars', 'bot'))
        ),
        color_info=('ref', 'share', 'frame_color'),
        sem_info='frame',
    ))

    
    return horiz_back

def ti_trans_horiz_slats(A,B,C):    
    return ('symTranslate', 0., _get_uni_val(-.8, -.9), random.randint(2,3))

def make_house():
    house_cat_vars = {
        'chimney': ('yes', 'no'),
        'window': ('single', 'double'),
        'bushes': ('yes', 'no'),
        'mcolor': ('yes', 'no')
    }

    def house_sample_var_fn(cvars):
        svars = {}

        if cvars['chimney'] == 'yes':
            svars['chimney_prim'] = 'square'
        elif cvars['chimney'] == 'no':
            svars['chimney_prim'] = None

        svars['double_window_prim'] = None
        svars['single_window_prim'] = None
        
        if cvars['window'] == 'single':
            svars['single_window_prim'] = random.choice(['circle', 'square'])
        elif cvars['window'] == 'double':
            svars['double_window_prim'] = random.choice(['circle', 'square'])

        svars['left_bushes_prim'] = None
        svars['right_bushes_prim'] = None
        svars['both_bushes_prim'] = None

        if cvars['bushes'] == 'yes':
            btype = random.choice(['left', 'right', 'both'])            
            svars[f'{btype}_bushes_prim'] = 'circle'                    

        svars['roof_color'] = random.choice(['blue', 'green'])
        svars['door_color'] = random.choice(['red', 'blue'])
        svars['window_color'] = random.choice(['red', 'blue'])

        if cvars['mcolor'] == 'yes':
            svars['window_color'] = svars['door_color']
        elif cvars['mcolor'] == 'no':
            while svars['door_color'] == svars['window_color']:
                svars['window_color'] = random.choice(['red', 'blue'])
                    
        return svars

    house = Concept(
        'house',
        house_cat_vars,
        house_sample_var_fn
    )

    house.add_part(Part(
        name = 'body',
        prim_info = ('static', 'square'),
        size_info = (            
            ('uni', 0.5, 0.85),
            ('uni', 0.3, 0.5),
        ),
        loc_info = (
            ('static', 0.),
            ('uni', -.3, -.1),
        ),
        color_info=('static', 'grey'),
        sem_info='body',
    ))

    house.add_part(Part(
        name = 'chimney',
        prim_info = ('ref', 'share', 'chimney_prim'),
        size_info = (            
            ('uni', 0.05, 0.1),
            ('uni', 0.3, 0.4),
        ),
        loc_info = (
            ('uni', -.35, -.15),
            ('rel', 'bot', ('body', 'top')),
        ),
        color_info=('static', 'red'),
        sem_info='chimney',
    ))
    
    house.add_part(Part(
        name = 'roof',
        prim_info = ('static', 'triangle'),
        size_info = (            
            ('ref', 'body', 'width'),
            ('uni', 0.2, 0.35),
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'bot', ('body', 'top')),
        ),
        color_info=('ref', 'share', 'roof_color'),
        sem_info='roof',
    ))

    house.add_part(Part(
        name = 'door',
        prim_info = ('static', 'square'),
        size_info = (            
            ('uni', 0.1, 0.15),
            ('uni', 0.1, 0.2),
        ),
        loc_info = (
            ('uni', -0.05, 0.05),
            ('rel', 'bot', ('body', 'bot')),            
        ),
        color_info=('ref', 'share', 'door_color'),
        sem_info='door',
    ))

    house.add_part(Part(
        name = 'single_window',
        prim_info = ('ref', 'share', 'single_window_prim'),
        size_info = (            
            ('uni', 0.1, 0.15),
            ('uni', 0.1, 0.15),
        ),
        loc_info = (
            ('expr', 'sub', ('rel', 'right', ('body', 'right')), ('uni', 0.05, 0.1)),
            ('expr', 'sub', ('rel', 'top', ('body', 'top')), ('uni', 0.05, 0.1))
        ),
        color_info=('ref', 'share', 'window_color'),
        sem_info='window',
    ))

    house.add_part(Part(
        name = 'double_window',
        prim_info = ('ref', 'share', 'double_window_prim'),
        size_info = (            
            ('uni', 0.1, 0.15),
            ('uni', 0.1, 0.15),
        ),
        loc_info = (
            ('expr', 'sub', ('rel', 'right', ('body', 'right')), ('uni', 0.05, 0.1)), 
            ('expr', 'sub', ('rel', 'top', ('body', 'top')), ('uni', 0.05, 0.1))
        ),
        color_info=('ref', 'share', 'window_color'),
        sem_info='window',
        top_info=('static', ('symReflect', 'AX'))
    ))

    house.add_part(Part(
        name = 'both_bush',
        prim_info = ('ref', 'share', 'both_bushes_prim'),
        size_info = (            
            ('uni', 0.15, 0.2),
            ('static', 0.1),
        ),
        loc_info = (
            ('rel', 'center', ('body', 'right')),
            ('rel', 'center', ('body', 'bot')),            
        ),
        color_info=('static', 'green'),
        sem_info='bush',
        top_info=('static', ('symReflect', 'AX'))
    ))

    house.add_part(Part(
        name = 'left_bush',
        prim_info = ('ref', 'share', 'left_bushes_prim'),
        size_info = (            
            ('uni', 0.15, 0.2),
            ('static', 0.1),
        ),
        loc_info = (
            ('rel', 'center', ('body', 'left')),
            ('rel', 'center', ('body', 'bot')),            
        ),
        color_info=('static', 'green'),
        sem_info='bush',
    ))

    house.add_part(Part(
        name = 'right_bush',
        prim_info = ('ref', 'share', 'right_bushes_prim'),
        size_info = (            
            ('uni', 0.15, 0.2),
            ('static', 0.1),
        ),
        loc_info = (
            ('rel', 'center', ('body', 'right')),
            ('rel', 'center', ('body', 'bot')),            
        ),
        color_info=('static', 'green'),
        sem_info='bush',
    ))    

    
    return house



def make_side_chair():

    side_chair_cat_vars = {
        'base': ('pedestal', 'regular'),
        'facing': ('right', 'left'),
        'armrest': ('yes', 'no'),
        'color': ('diff', 'same')
    }
    
    def side_chair_sample_var_fn(cvars):
        svars = {}

        svars['pedestal_single_col'] = None
        svars['pedestal_single_base'] = None
        svars['pedestal_double_prim'] = None
        svars['base_double_prim'] = None
        svars['regular_prim'] = None
        
        if cvars['base'] == 'pedestal':
            if random.random() < 0.5:
                svars['pedestal_single_col'] = 'square'
                svars['pedestal_single_base'] = 'circle'
            else:
                svars['pedestal_double_prim'] = 'square'
                svars['base_double_prim'] = 'triangle'
                
        if cvars['base'] == 'regular':
            svars['regular_prim'] = 'square'

        svars['lface_armrest_prim'] = None
        svars['rface_armrest_prim'] = None
        svars['lface_back_prim'] = None
        svars['rface_back_prim'] = None

        if cvars['facing'] == 'left':
            svars['lface_back_prim'] = 'square'
            if cvars['armrest'] == 'yes':
                svars['lface_armrest_prim'] = random.choice(['square', 'circle'])
        elif cvars['facing'] == 'right':
            svars['rface_back_prim'] = 'square'
            if cvars['armrest'] == 'yes':
                svars['rface_armrest_prim'] = random.choice(['square', 'circle'])

        base_color = random.choice(['red', 'green', 'blue'])
        top_color = random.choice(['red', 'green', 'blue'])

        if cvars['color'] == 'same':
            top_color = base_color            
        elif cvars['color'] == 'diff':
            while top_color == base_color:
                top_color = random.choice(['red', 'green', 'blue'])

        svars['base_color'] = base_color
        svars['top_color'] = top_color
        
        return svars

    side_chair = Concept(
        'side_chair',
        side_chair_cat_vars,
        side_chair_sample_var_fn
    )

    side_chair.add_part(Part(
        name = 'seat',
        prim_info = ('static', 'square'),
        size_info = (            
            ('uni', 0.4, 0.7),
            ('uni', 0.05, 0.15),
        ),
        loc_info = (
            ('static', 0.),
            ('expr', 'mul', ('ref', 'seat', 'height'), ('static', -1.0))
        ),
        color_info=('ref', 'share', 'base_color'),
        sem_info='seat',
    ))

    side_chair.add_part(Part(
        name = 'rface_back',        
        prim_info = ('ref', 'share', 'rface_back_prim'),
        size_info = (            
            ('uni', 0.1, 0.2),
            ('uni', 0.2, 0.4),
        ),
        loc_info = (
            ('rel', 'left', ('seat', 'left')),
            ('rel', 'bot', ('seat', 'top'))
        ),
        color_info=('ref', 'share', 'top_color'),
        sem_info='back',
    ))

    side_chair.add_part(Part(
        name = 'rface_arm',
        prim_info = ('ref', 'share', 'rface_armrest_prim'),
        size_info = (            
            ('uni', 0.2, 0.25),
            ('uni', 0.05, 0.1),
        ),
        loc_info = (
            ('rel', 'left', ('rface_back', 'right')),
            ('uni', 0.15, 0.25)
        ),
        color_info=('static', 'grey'),
        sem_info='arm',
    ))

    side_chair.add_part(Part(
        name = 'lface_back',
        prim_info = ('ref', 'share', 'lface_back_prim'),
        size_info = (            
            ('uni', 0.1, 0.2),
            ('uni', 0.2, 0.4),
        ),
        loc_info = (
            ('rel', 'right', ('seat', 'right')),
            ('rel', 'bot', ('seat', 'top'))
        ),
        color_info=('ref', 'share', 'top_color'),
        sem_info='back',
    ))

    side_chair.add_part(Part(
        name = 'lface_arm',
        prim_info = ('ref', 'share', 'lface_armrest_prim'),
        size_info = (            
            ('uni', 0.2, 0.25),
            ('uni', 0.05, 0.1),
        ),
        loc_info = (
            ('rel', 'right', ('lface_back', 'left')),
            ('uni', 0.15, 0.25)
        ),
        color_info=('static', 'grey'),
        sem_info='arm',
    ))

    side_chair.add_part(Part(
        name = 'regular_leg',
        prim_info = ('ref', 'share', 'regular_prim'),
        size_info = (            
            ('uni', 0.05, 0.1),
            ('uni', 0.1, 0.25),
        ),
        loc_info = (
            ('prel', 'right', ('seat', 'right'), ('uni', 1.0, 3.0)),
            ('rel', 'top', ('seat', 'bot'))
        ),
        color_info=('ref', 'share', 'base_color'),
        sem_info='leg',
        top_info=('static', ('symReflect', 'AX'))
    ))

    side_chair.add_part(Part(
        name = 'ped_single_col',
        prim_info = ('ref', 'share', 'pedestal_single_col'),
        size_info = (            
            ('static', 0.05),
            ('uni', 0.1, 0.2),
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'top', ('seat', 'bot'))
        ),
        color_info=('ref', 'share', 'base_color'),
        sem_info='leg',
    ))

    side_chair.add_part(Part(
        name = 'ped_double_col',
        prim_info = ('ref', 'share', 'pedestal_double_prim'),
        size_info = (            
            ('static', 0.05),
            ('uni', 0.15, 0.25),
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'top', ('seat', 'bot'))
        ),
        color_info=('ref', 'share', 'base_color'),
        sem_info='leg',
    ))

    side_chair.add_part(Part(
        name = 'ped_double_base',
        prim_info = ('ref', 'share', 'base_double_prim'),
        size_info = (            
            ('uni', 0.25, 0.3),
            ('uni', 0.1, 0.15),
        ),
        loc_info = (
            ('prel', 'left', ('ped_double_col', 'right'), ('static', 0.75)),
            ('rel', 'bot', ('ped_double_col', 'bot'))
        ),
        color_info=('ref', 'share', 'base_color'),
        sem_info='feet',
        top_info=('static', ('symReflect', 'AX'))
    ))

    side_chair.add_part(Part(
        name = 'ped_single_base',
        prim_info = ('ref', 'share', 'pedestal_single_base'),
        size_info = (            
            ('uni', 0.4, 0.6),
            ('static', 0.05),
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'top', ('ped_single_col', 'bot'))
        ),
        color_info=('ref', 'share', 'base_color'),
        sem_info='feet',
    ))

    return side_chair


def make_table():
    table_cat_vars = {
        'base': ('regular', 'pedestal'),
        'leaf': ('left', 'right', 'no'),
        'top': ('yes', 'no'),
        'knob': ('yes', 'no'),
    }
 
    def table_sample_var_fn(cvars):
        svars = {}

        svars['ped_prim'] = None
        svars['reg_leg_prim'] = None        
        
        if cvars['base'] == 'regular':
            svars['reg_leg_prim'] = 'square'
        elif cvars['base'] == 'pedestal':
            svars['ped_prim'] = 'square'

        svars['right_leaf_prim'] = None
        svars['left_leaf_prim'] = None
        
        if cvars['leaf'] == 'left':            
            svars['left_leaf_prim'] = 'circle'
        elif cvars['leaf'] == 'right':            
            svars['right_leaf_prim'] = 'circle'
            
        if cvars['top'] == 'yes':
            svars['top_prim'] = 'square'
        elif cvars['top'] == 'no':
            svars['top_prim'] = None

        svars['single_knob_prim'] = None
        svars['double_knob_prim'] = None

        if  cvars['knob'] == 'yes':
            if random.random() < 0.5:
                svars['single_knob_prim'] = 'triangle'
            else:
                svars['double_knob_prim'] = 'circle'
                
        other_color = random.choice(['red', 'green', 'blue', 'grey'])
        main_color = random.choice(['red', 'green', 'blue'])

        while other_color == main_color:
            other_color = random.choice(['red', 'green', 'blue', 'grey'])

        svars['other_color'] = other_color
        svars['main_color'] = main_color
        
        return svars

    table = Concept(
        'table',
        table_cat_vars,
        table_sample_var_fn
    )

    table.add_part(Part(
        name = 'drawer',
        prim_info = ('static', 'square'),
        size_info = (            
            ('uni', 0.5, 0.75),
            ('uni', 0.15, 0.2),
        ),
        loc_info = (
            ('static', 0.),
            ('uni', 0.25, 0.4),
        ),
        color_info=('ref', 'share', 'main_color'),
        sem_info='drawer',
    ))

    table.add_part(Part(
        name = 'reg_leg',
        prim_info = ('ref', 'share', 'reg_leg_prim'),
        size_info = (            
            ('uni', 0.05, 0.15),
            ('uni', 0.25, 0.35),
        ),
        loc_info = (
            ('prel', 'right', ('drawer', 'right'), ('uni', 1.0, 3.0)),
            ('rel', 'top', ('drawer', 'bot'))
        ),
        color_info=('ref', 'share', 'main_color'),
        sem_info='base',
        top_info=('static', ('symReflect', 'AX'))
    ))

    table.add_part(Part(
        name = 'ped_single_col',
        prim_info = ('ref', 'share', 'ped_prim'),
        size_info = (            
            ('uni', 0.1, 0.15),
            ('uni', 0.15, 0.25),
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'top', ('drawer', 'bot'))
        ),
        color_info=('ref', 'share', 'main_color'),
        sem_info='base',
    ))

    table.add_part(Part(
        name = 'ped_single_base',
        prim_info = ('ref', 'share', 'ped_prim'),
        size_info = (            
            ('uni', 0.3, 0.5),
            ('uni', 0.05, 0.1),
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'top', ('ped_single_col', 'bot'))
        ),
        color_info=('ref', 'share', 'main_color'),
        sem_info='base',
    ))

    table.add_part(Part(
        name = 'top',
        prim_info = ('ref', 'share', 'top_prim'),
        size_info = (            
            ('ref', 'drawer', 'width'),
            ('static', 0.05),
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'bot', ('drawer', 'top'))
        ),
        color_info=('ref', 'share', 'other_color'),
        sem_info='top',
    ))
    table.add_part(Part(
        name = 'left_leaf',
        prim_info = ('ref', 'share', 'left_leaf_prim'),
        size_info = (            
            ('uni', 0.05, 0.1),
            ('uni', 0.25, 0.35),
        ),
        loc_info = (
            ('rel', 'center', ('drawer', 'left')),
            ('rel', 'top', ('drawer', 'top'))
        ),
        color_info=('ref', 'share', 'other_color'),
        sem_info='leaf',
    ))
    table.add_part(Part(
        name = 'right_leaf',
        prim_info = ('ref', 'share', 'right_leaf_prim'),
        size_info = (            
            ('uni', 0.05, 0.1),
            ('uni', 0.25, 0.35),
        ),
        loc_info = (
            ('rel', 'center', ('drawer', 'right')),
            ('rel', 'top', ('drawer', 'top'))
        ),
        color_info=('ref', 'share', 'other_color'),
        sem_info='leaf',
    ))

    table.add_part(Part(
        name = 'single_knob',
        prim_info = ('ref', 'share', 'single_knob_prim'),
        size_info = (            
            ('static', 0.1),
            ('static', 0.1),
        ),
        loc_info = (
            ('uni', -.1, .1),
            ('rel', 'center', ('drawer', 'center_height'))
        ),
        color_info=('static', 'grey'),
        sem_info='knob',
    ))

    table.add_part(Part(
        name = 'double_knob',
        prim_info = ('ref', 'share', 'double_knob_prim'),
        size_info = (            
            ('static', 0.1),
            ('static', 0.1),
        ),
        loc_info = (
            ('uni', .25, .35),
            ('rel', 'center', ('drawer', 'center_height'))
        ),
        color_info=('static', 'grey'),
        sem_info='knob',
        top_info=('static', ('symReflect', 'AX'))
    ))

        
    return table

def make_bookshelf():
    bookshelf_cat_vars = {
        'feet': ('yes', 'no'),
        'sides': ('yes', 'no'),
        'frame': ('red', 'match'),
        'shelves': ('trans', 'ref')
    }

    def bookshelf_sample_var_fn(cvars):
        svars = {}

        if cvars['feet'] == 'yes':
            svars['feet_prim'] = random.choice(['triangle', 'circle', 'square'])
        elif cvars['feet'] == 'no':
            svars['feet_prim'] = None

        if cvars['sides'] == 'yes':
            svars['sides_prim'] = 'square'
        elif cvars['sides'] == 'no':
            svars['sides_prim'] = None

        svars['bar_color'] = random.choice(['red', 'green', 'blue'])
        
        if cvars['frame'] == 'red':
            svars['frame_color'] = 'red'
        elif cvars['frame'] == 'match':
            svars['frame_color'] = svars['bar_color'] 

        svars['ref_shelves_prim'] = None
        svars['trans_shelves_prim'] = None
        
        if cvars['shelves'] == 'ref':        
            svars['ref_shelves_prim'] = 'square'
        elif cvars['shelves'] == 'trans':
            svars['trans_shelves_prim'] = 'square'
            
        return svars

    bookshelf = Concept(
        'bookshelf',
        bookshelf_cat_vars,
        bookshelf_sample_var_fn
    )

    bookshelf.add_part(Part(
        name = 'back',
        prim_info = ('static', 'square'),
        size_info = (            
            ('uni', 0.5, 0.8),
            ('uni', 0.7, 0.85),
        ),
        loc_info = (
            ('static', 0.),
            ('static', 0.0),
        ),
        color_info=('static', 'grey'),
        sem_info='back',
    ))

    bookshelf.add_part(Part(
        name = 'top_frame',
        prim_info = ('static', 'square'),
        size_info = (            
            ('ref', 'back', 'width'),
            ('static', 0.05),
        ),
        loc_info = (
            ('static', 0.),
            ('rel', 'top', ('back', 'top')),
        ),
        color_info=('ref', 'share', 'frame_color'),
        sem_info='frame',
        top_info=('static', ('symReflect', 'AY'))
    ))

    bookshelf.add_part(Part(
        name = 'side_frame',
        prim_info = ('ref', 'share', 'sides_prim'),
        size_info = (
            ('static', 0.05),
            ('ref', 'back', 'height'),            
        ),
        loc_info = (
            ('rel', 'right', ('back', 'left')),
            ('static', 0.),            
        ),
        color_info=('ref', 'share', 'frame_color'),
        sem_info='frame',
        top_info=('static', ('symReflect', 'AX'))
    ))

    bookshelf.add_part(Part(
        name = 'feet',
        prim_info = ('ref', 'share', 'feet_prim'),
        size_info = (
            ('uni', 0.05, 0.1),
            ('ref', 'feet', 'width')
        ),
        loc_info = (
            ('uni', .15, .2),
            ('rel', 'top', ('back', 'bot')),
        ),
        color_info=('static', 'grey'),
        sem_info='feet',
        top_info=('static', ('symReflect', 'AX'))
    ))

    bookshelf.add_part(Part(
        name = 'ref_shelves',
        prim_info = ('ref', 'share', 'ref_shelves_prim'),
        size_info = (            
            ('ref', 'back', 'width'),
            ('static', 0.1),
        ),
        loc_info = (
            ('static', 0.),
            ('expr', 'sub', ('rel', 'top', ('top_frame', 'bot')), ('uni', 0.15, 0.25)),
        ),
        color_info=('ref', 'share', 'bar_color'),
        sem_info='bar',
        top_info=('static', ('symReflect', 'AY'))
    ))

    bookshelf.add_part(Part(
        name = 'trans_shelves',
        prim_info = ('ref', 'share', 'trans_shelves_prim'),
        size_info = (            
            ('ref', 'back', 'width'),
            ('static', 0.05),
        ),
        loc_info = (
            ('static', 0.),
            ('expr', 'sub', ('rel', 'top', ('top_frame', 'bot')), ('uni', 0.1, 0.2)),
        ),
        color_info=('ref', 'share', 'bar_color'),
        sem_info='bar',
        top_info=('fn', book_shelf_si)
    ))

                    
    return bookshelf


def book_shelf_si(SP, samples, shared_vars):

    pos = SP.y_pos


    max_extent = pos * -2

    max_k = int(abs(max_extent) / 0.1)

    K = random.randint(2, min(max_k, 4))
    
    return ('symTranslate', 0, max_extent, K)
