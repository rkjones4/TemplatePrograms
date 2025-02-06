# File taken from https://github.com/rkjones4/ShapeAssembly/blob/master/code/ShapeAssembly.py

import torch
import re
import numpy as np
import math
import ast
import sys
from copy import deepcopy

# Params controlling execution behavior
EPS = .01
SMALL_EPS = 1e-4
COS_DIST_THRESH = 0.9

# Helper function: write mesh to out file
def writeObj(verts, faces, outfile):
    with open(outfile, 'w') as f:
        for a, b, c in verts.tolist():
            f.write(f'v {a} {b} {c}\n')
        for a, b, c in faces.tolist():
            f.write(f"f {a+1} {b+1} {c+1}\n")

            
class Cuboid():
    """
    Cuboids are the base (and only) objects of a ShapeAssembly program. Dims are their dimensions, pos is the center of the cuboid, rfnorm (right face), tfnorm (top face) and ffnorm (front face) specify the orientation of the cuboid. The bounding volume is just a non-visible cuboid. Cuboids marked with the aligned flag behavior differently under attachment operations. 
    """
    def __init__(self, name, aligned = True, vis = True):
        
        # The default cube is unit, axis-aligned, centered at the origin
        self.dims =  torch.tensor([1.0,1.0,1.0])
        self.pos = torch.tensor([0.0,0.0,0.0])
        self.rfnorm = torch.tensor([1.0,0.0,0.0])
        self.tfnorm = torch.tensor([0.0,1.0,0.0])
        self.ffnorm = torch.tensor([0.0,0.0,1.0])
        # Keep track of all attachment obligations this cube has
        self.attachments = []
        self.move_atts = []
        # The bbox is not visible, but is still a cuboid, otherwise this should be True
        self.is_visible = vis
        self.name = name
        self.parent = None
        self.parent_axis = None
        self.aligned = aligned

    def flipCuboid(self, a_ind):
        transform = torch.ones(3)
        transform[a_ind] *= -1 
        self.pos = transform * self.pos
        self.rfnorm = -1 * (transform * self.rfnorm)
        self.tfnorm = -1 * (transform * self.tfnorm)
        self.ffnorm = -1 * (transform * self.ffnorm)
        
    # Get the corners of the cuboid
    def getCorners(self):
        xd = self.dims[0] / 2
        yd = self.dims[1] / 2
        zd = self.dims[2] / 2

        corners = torch.stack((
            (self.rfnorm * xd) + (self.tfnorm * yd) + (self.ffnorm * zd),
            (self.rfnorm * xd) + (self.tfnorm * yd) + (self.ffnorm * -1 * zd),
            (self.rfnorm * xd) + (self.tfnorm * -1 * yd) + (self.ffnorm * zd),
            (self.rfnorm * xd) + (self.tfnorm * -1 * yd) + (self.ffnorm * -1 * zd),
            (self.rfnorm * -1 * xd) + (self.tfnorm * yd) + (self.ffnorm * zd),
            (self.rfnorm * -1 * xd) + (self.tfnorm * yd) + (self.ffnorm * -1 * zd),
            (self.rfnorm * -1 * xd) + (self.tfnorm * -1 * yd) + (self.ffnorm * zd),
            (self.rfnorm * -1 * xd) + (self.tfnorm * -1 * yd) + (self.ffnorm * -1 * zd),
        ))
        return corners + self.pos

    # Get the global point specified by relative coordinates x,y,z 
    def getPos(self, x, y, z):
        
        pt = torch.stack((x, y, z))
    
        r = torch.stack((
            self.rfnorm,
            self.tfnorm,
            self.ffnorm
        )).T

        t_dims = torch.stack((self.dims[0], self.dims[1], self.dims[2]))
        
        return (r @ ((pt - .5) * t_dims)) + self.pos

    # Get the relative position of global poiunt gpt
    def getRelPos(self, gpt, normalize = False):
        O = self.getPos(
            torch.tensor(0.),
            torch.tensor(0.),
            torch.tensor(0.)
        )

        A = torch.stack([
            self.dims[0].clone() * self.rfnorm.clone(),
            self.dims[1].clone() * self.tfnorm.clone(),
            self.dims[2].clone() * self.ffnorm.clone()
        ]).T

        B = gpt - O
        p = A.inverse() @ B

        if normalize:
            return torch.clamp(p, 0.0, 1.0)
        
        return p                
    
    # Make the cuboid bigger by a multiplied factor of scale (either dim 3 or dim 1)
    def scaleCuboid(self, scale):
        self.dims *= scale

    # Make the cuboid bigger by an added factor of scale to a specific dimension
    def increaseDim(self, dim, inc):
        dim_to_scale = {            
            "height": torch.tensor([0.0, 1.0, 0.0]),
            "width": torch.tensor([0.0, 0.0, 1.0]),
            "length": torch.tensor([1.0, 0.0, 0.0])
        }
        s = dim_to_scale[dim] * inc
        self.dims += s
        
    # Move the center of the cuboid by the translation vector
    def translateCuboid(self, translation):
        self.pos += translation

    # Used to convert cuboid into triangles on its faces
    def getTriFaces(self):
        return [
            [0, 2, 1],
            [1, 2, 3],
            [0, 4, 6],
            [0, 6, 2],
            [1, 3, 5],
            [3, 7, 5],
            [4, 5, 7],
            [4, 7, 6],
            [1, 5, 4],
            [0, 1, 4],
            [2, 6, 7],
            [2, 7, 3]
        ]

    # Get the triangulation of the cuboid corners, for visualization + sampling
    def getTris(self):
        if self.is_visible:
            verts = self.getCorners()
            faces = torch.tensor(self.getTriFaces(), dtype=torch.long)
            return verts, faces
        return None, None

    # Return any attachments that are on this cuboid
    def getAttachments(self):
        return self.attachments
    
    # Return the cuboid's parameterization
    def getParams(self):
        return torch.cat((
            self.dims, self.pos, self.rfnorm, self.tfnorm, self.ffnorm
        )) 

class AttPoint():
    """ 
    Attachment Points live with the local coordinate frame [0, 1]^3 of a cuboid. They are used to connect cuboids together.
    """
    def __init__(self, cuboid, x, y, z):
        self.cuboid = cuboid
        self.x = x
        self.y = y
        self.z = z

    # To get the global position, all we need is the cuboid+face info, and the relative uv pos
    def getPos(self):
        return self.cuboid.getPos(self.x, self.y, self.z)
    
    # If we scale the height of the cuboid, what is the rate of change of this AP
    def getChangeVectorHeight(self):
        norm = self.cuboid.tfnorm
        return (self.y - .5) * norm

    # If we scale the length of the cuboid, what is the rate of change of this AP
    def getChangeVectorLength(self):
        norm = self.cuboid.rfnorm
        return (self.x - .5) * norm
        
    # If we scale the width of the cuboid, what is the rate of change of this AP
    def getChangeVectorWidth(self):
        norm = self.cuboid.ffnorm
        return (self.z - .5) * norm
        
    # get rate of change of this AP when we change the specified dimension
    def getChangeVector(self, dim):
        dim_to_sf = {
            'height': self.getChangeVectorHeight,
            'length': self.getChangeVectorLength,
            'width': self.getChangeVectorWidth,
        }
        return dim_to_sf[dim]()                

    

class Program():
    """
    A program maintains a representation of entire shape, including all of the member cuboids
    and all of the attachment points. The execute function is the entrypoint of text programs.
    """
    def __init__(self, cuboids = {}):
        self.cuboids = self.getBoundBox()
        self.cuboids.update(cuboids)
        self.commands = []
        self.parameters = []
        self.att_points = {}
        
    def flip(self, flip_axis):
        if flip_axis == 'AX':
            axis = 0
        elif flip_axis == 'AY':
            axis = 1
        elif flip_axis == 'AZ':
            axis = 2
        for name, c in self.cuboids.items():
            if name == 'bbox':
                continue
            c.flipCuboid(axis)
            
    # Each program starts off with an invisible bounding box
    def getBoundBox(self):
        bbox = Cuboid("bbox", aligned = True, vis=False)
                
        return {
            "bbox": bbox
        }


    # Get the triangles in the current scene -> first index is bounding box so skipped
    def getShapeGeo(self):
        
        if len(self.cuboids) < 2:
            return None, None
        
        cuboids = list(self.cuboids.values())
        
        verts = torch.tensor([],dtype=torch.float)
        faces = torch.tensor([],dtype=torch.long)
        
        for cube in cuboids[1:]:            
            v, f = cube.getTris()
            if v is not None and f is not None:
                faces =  torch.cat((faces, (f + verts.shape[0])))
                verts = torch.cat((verts, v))

        return verts, faces


    # Make an obj of the current scene
    def render(self, ofile = "output.obj"):        
        verts, faces = self.getShapeGeo()
        writeObj(verts, faces, ofile)

    # Parses a cuboid text line
    def parseCuboid(self, line):
        s = re.split(r'[()]', line)
        name = s[0].split("=")[0].strip()
        dim0 = None
        dim1 = None
        dim2 = None

        params = s[1].split(',')
        dim0 = torch.tensor(float(params[0]))
        dim1 = torch.tensor(float(params[1]))
        dim2 = torch.tensor(float(params[2]))

        return (name, dim0, dim1, dim2, True)

    
    # Construct a new cuboid, add it to state
    def executeCuboid(self, parse):
        name = parse[0]

        if name in self.cuboids:
            c = self.cuboids[name]            
            c.dims = torch.stack((parse[1], parse[2], parse[3]))
            
        else:            
            c = Cuboid(
                parse[0],
                aligned = True,
            )
            
            c.scaleCuboid(torch.stack((parse[1], parse[2], parse[3])))

            self.cuboids.update({
                parse[0]: c
            })
            
    # Logic for cuboids with no previous attachment. Finds a translation to satisfy the attachment
    def first_attach(self, ap, gpos):
        cur_pos = ap.getPos()
        diff = gpos - cur_pos
        ap.cuboid.translateCuboid(diff)
        return True
                
    # Given cuboids a and b, find the closest pair of points in their local coordinate frames to one another
    def getClosestPoints(self, a, b, xyz1, xyz2, is_bbox):
        assert False
        
    # For aligned cuboids with a previous attachments,
    # see if increasing any dimension would cause the fit to be improved
    def aligned_attach(self, ap, oap):
        # skip second aligned attach
        return
                                
    # Moves the attach point to the global position
    def attach(self, ap, gpos, oci, oap=None):
        assert ap.cuboid.name != "bbox", 'tried to move the bbox'        
        self.aligned_cube_attach(ap, gpos, oci, oap)

    # Aligned attachment
    def aligned_cube_attach(self, ap, gpos, oci, oap):
        prev_atts = ap.cuboid.getAttachments()
        if len(prev_atts) == 0:
            self.first_attach(ap, gpos)
        else:
            self.aligned_attach(ap, oap)            
        
        prev_atts.append((ap, gpos, oci))
        ap.cuboid.move_atts.append((ap, gpos, oci))
        
    # Parses an attach line
    def parseAttach(self, line):
        s = re.split(r'[()]', line)
        args = [a.strip() for a in s[1].split(',')]
        return (
            args[0],
            args[1],
            torch.tensor(float(args[2])),
            torch.tensor(float(args[3])),
            torch.tensor(float(args[4])),
            torch.tensor(float(args[5])),
            torch.tensor(float(args[6])),
            torch.tensor(float(args[7]))
        )

            
    # Execute an attach line, creates two attachment points, then figures out how to best satisfy new constraint
    def executeAttach(self, parse):
        ap1 = AttPoint(
            self.cuboids[parse[0]],
            parse[2],
            parse[3],
            parse[4],
        )

        ap2 = AttPoint(
            self.cuboids[parse[1]],
            parse[5],
            parse[6],
            parse[7],
        )

        ap_pt_name = f'{parse[0]}_to_{parse[1]}'
        # Attach points should have unique names
        while ap_pt_name in self.att_points:            
            ap_pt_name += '_n'
        self.att_points[ap_pt_name] = ap2
        
        ap2.cuboid.getAttachments().append((ap2, ap2.getPos(), ap1.cuboid.name))
        self.attach(ap1, ap2.getPos(), ap2.cuboid.name, ap2)

    # Parses a reflect command
    def parseReflect(self, line):
        s = re.split(r'[()]', line)
        args = [a.strip() for a in s[1].split(',')]
        return (
            args[0],
            args[1],
        )

    # Parses a translate command
    def parseTranslate(self, line):
        s = re.split(r'[()]', line)
        args = [a.strip() for a in s[1].split(',')]
        return (
            args[0],
            args[1],
            int(args[2]),
            float(args[3])
        )

    # Parses a queeze command
    def parseSqueeze(self, line):
        s = re.split(r'[()]', line)
        args = [a.strip() for a in s[1].split(',')]
        return (
            args[0],
            args[1],
            args[2],
            args[3],
            float(args[4]),
            float(args[5])
        )

    # Help function for getting direction of reflect commands
    def getRefDir(self, d):
        bbox = self.cuboids['bbox']
        if d == 'AX':
            return bbox.rfnorm.clone()
        elif d == 'AY':
            return bbox.tfnorm.clone()
        elif d == 'AZ':
            return bbox.ffnorm.clone()
        else:
            assert False, 'bad reflect argument'

    # Help function for getting direction + scale of translate commands
    def getTransDir(self, d):
        bbox = self.cuboids['bbox']
        if d == 'AX':
            return bbox.rfnorm.clone(), bbox.dims[0].clone()
        elif d == 'AY':
            return bbox.tfnorm.clone(), bbox.dims[1].clone()
        elif d == 'AZ':
            return bbox.ffnorm.clone(), bbox.dims[2].clone()
        else:
            assert False, 'bad reflect argument'
            
    # Given an axis + a center, consructs a tranformation matrix to satisfy reflection
    def getRefMatrixHomo(self, axis, center):

        m = center
        d = axis / axis.norm()

        refmat = torch.stack((
            torch.stack((1 - 2 * d[0] * d[0], -2 * d[0] * d[1], -2 * d[0] * d[2], 2 * d[0] * d[0] * m[0] + 2 * d[0] * d[1] * m[1] + 2 * d[0] * d[2] * m[2])),
            torch.stack((-2 * d[1] * d[0], 1 - 2 * d[1] * d[1], -2 * d[1] * d[2], 2 * d[1] * d[0] * m[0] + 2 * d[1] * d[1] * m[1] + 2 * d[1] * d[2] * m[2])),
            torch.stack((-2 * d[2] * d[0], -2 * d[2] * d[1], 1 - 2 * d[2] * d[2], 2 * d[2] * d[0] * m[0] + 2 * d[2] * d[1] * m[1] + 2 * d[2] * d[2] * m[2]))
        ))

        return refmat

    # Reflect a point p, about center and a direction ndir
    def reflect_point(self, p, center, ndir):
        pad = torch.nn.ConstantPad1d((0, 1), 1.0)
        reflection = self.getRefMatrixHomo(ndir, center)
        posHomo = pad(p)
        return reflection @ posHomo
    
    # Executes a reflect line by making + executing new Cuboid and attach lines
    def executeReflect(self, parse):
        c = self.cuboids[parse[0]]        
        assert c.name != "bbox", 'tried to move the bbox'
        
        rdir = self.getRefDir(parse[1])
        name = c.name + '_ref'
        
        self.executeCuboid([f'{name}', c.dims[0].clone(), c.dims[1].clone(), c.dims[2].clone(), c.aligned])
                        
        self.cuboids[f'{name}'].parent = c.name
        self.cuboids[f'{name}'].parent_axis = parse[1]
        
        atts = c.move_atts
        for att in atts:
            
            if parse[1] == 'AX':
                x = 1 - att[0].x.clone()
            else:
                x = att[0].x.clone()

            if parse[1] == 'AY':
                y = 1 - att[0].y.clone()
            else:
                y = att[0].y.clone()

            if parse[1] == 'AZ':
                z = 1 - att[0].z.clone()
            else:
                z = att[0].z.clone()
            
            n = att[2]

            cpt = att[0].getPos().clone()
            rpt = self.reflect_point(cpt, self.cuboids['bbox'].pos.clone(), rdir)
            
            rrpt = self.cuboids[n].getRelPos(rpt, True)
            
            self.executeAttach([f'{name}', f'{n}', x, y, z, rrpt[0], rrpt[1], rrpt[2]])
            
    # Executes a translate line by making + executing new Cuboid and attach lines
    def executeTranslate(self, parse):
        
        c = self.cuboids[parse[0]]
        assert c.name != "bbox", 'tried to move the bbox'
        tdir, td = self.getTransDir(parse[1])

        N = parse[2]
        scale = (td * parse[3]) / float(N)

        for i in range(1, N+1):
        

            name = c.name + f'_trans_{i}'
            self.executeCuboid([f'{name}', c.dims[0].clone(), c.dims[1].clone(), c.dims[2].clone(), c.aligned]) 
            self.cuboids[f'{name}'].parent = c.name
            
            atts = c.move_atts
            for att in atts:
                x = att[0].x
                y = att[0].y
                z = att[0].z
                n = att[2]

                cpt = att[0].getPos()
                rpt = cpt + (tdir * scale * i)

                rrpt = self.cuboids[n].getRelPos(rpt, True)

                self.executeAttach([f'{name}', f'{n}', x, y, z, rrpt[0], rrpt[1], rrpt[2]])


    # Helper function for finding opposite face
    def getOppFace(self, face):
        of = {
            'right': 'left',
            'left': 'right',
            'top': 'bot',
            'bot': 'top',
            'front': 'back',
            'back': 'front',
        }
        return of[face]

    # Local coordinate frame to center of face conversion
    def getFacePos(self, face):
        ft = {
            'right': ([1.0, 0.5, 0.5], 0, 0.),
            'left': ([0.0, 0.5, 0.5], 0, 1.),
            'top': ([.5, 1.0, 0.5], 1, 0.),
            'bot': ([.5, 0.0, 0.5], 1, 1.),
            'front': ([.5, 0.5, 1.0], 2, 0.),
            'back': ([.5, 0.5, 0.0], 2, 1.),
        }
        return ft[face]

    # Converts squeeze parameters into parameters needed for the two attachment operators.
    def getSqueezeAtt(self, face, u, v, is_bbox):
        at1, ind, val = self.getFacePos(face)
        # bbox is "flipped"
        if is_bbox:
            rval = 1-val
        else:
            rval = val
        at2 = torch.zeros(3).float()
        q = [u, v] 
        for i in range(3):
            if i == ind:
                at2[i] = rval
            else:
                at2[i] = q.pop(0)

        return torch.tensor(at1).float(), at2

    # Executes a squeeze line by making + executing new Cuboid and attach lines
    def executeSqueeze(self, parse):
        face = parse[3]
        oface = self.getOppFace(face)

        atc1, ato1 = self.getSqueezeAtt(
            face, parse[4], parse[5], parse[1] == 'bbox'
        )

        atc2, ato2 = self.getSqueezeAtt(
            oface, parse[4], parse[5], parse[2] == 'bbox'
        )        
            
        self.executeAttach([parse[0], parse[1], atc1[0], atc1[1], atc1[2], ato1[0], ato1[1], ato1[2]])
        self.executeAttach([parse[0], parse[2], atc2[0], atc2[1], atc2[2], ato2[0], ato2[1], ato2[2]])

    # Clear cuboids + attachment points, but keep the commands that made them in memory
    def resetState(self):
        self.cuboids = self.getBoundBox()
        self.att_points = {}
        
    # Supported commands and their execution functions
    # Commands are first parsed to get their type + parameters. Then, the line is executed by calling to the appropriate execute function 
    def execute(self, line):
        res = None
        if "Cuboid(" in line:
            parse = self.parseCuboid(line)
            self.executeCuboid(parse)
        
        elif "Attach(" in line:
            parse = self.parseAttach(line)
            self.executeAttach(parse)    

        elif "Reflect(" in line:
            parse = self.parseReflect(line)
            res = self.executeReflect(parse)

        elif "Translate(" in line:
            parse = self.parseTranslate(line)
            res = self.executeTranslate(parse)

        elif "Squeeze(" in line:
            parse = self.parseSqueeze(line)
            res = self.executeSqueeze(parse)

        # return any new lines generated by macros
        return res
            
    # To re-run a program given a set of commands and parameters. Often used during fitting to unstructurd geometry. 
    def runProgram(self, param_lines):
        self.resetState()

        command_to_func = {
            "Cuboid": self.executeCuboid,
            "Attach": self.executeAttach,
            "Squeeze": self.executeSqueeze,
            "Translate": self.executeTranslate,
            "Reflect": self.executeReflect
        }
        
        for command, parse in param_lines:
            func = command_to_func[command]
            func(parse)
        
def get_pos_delta(abox, rbox, pos):

    r = torch.stack((
        abox.rfnorm,
        abox.tfnorm,
        abox.ffnorm
    )).T

    return (r @ pos) + (abox.pos - rbox.pos) 


# Given a cuboid cube, and its local program bounding volume rbox, and the actual placement of its bonding volume abox, find the correct transformation for cube
def apply_delta(abox, rbox, cube):

    r = torch.stack((
        abox.rfnorm,
        abox.tfnorm,
        abox.ffnorm
    )).T

    cube.dims *=  (abox.dims / rbox.dims)    
    cube.pos = (r @ cube.pos) + (abox.pos - rbox.pos) 

    cube.rfnorm = r @ cube.rfnorm
    cube.tfnorm = r @ cube.tfnorm
    cube.ffnorm = r @ cube.ffnorm
        
# Execute a hierarchical shapeassembly program
def hier_execute(root, ret_ppred=False):
           
    bbox = Cuboid('bbox')
        
    bbox.dims = torch.tensor(
        [float(a) for a in re.split(r'[()]', root['prog'][0])[1].split(',')[:3]]
    )
    
    q = [(root, bbox, None)]
       
    scene_cubes = []
    scene_ppreds = []
    count = 0
    
    while len(q) > 0:    
        node, bbox, flip_axis = q.pop(0)

        lines = node["prog"]
        TP = Program()

        for line in lines:
            TP.execute(line)        

        if flip_axis:
            TP.flip(flip_axis)
            
        rbox = TP.cuboids.pop('bbox')
    
        add_cubes = []
        add_pps = []
        
        i_map = {}
        i = 0

        for c_key in TP.cuboids.keys():
            cub = TP.cuboids[c_key]
            child = None
            flip_axis = False

            if '_ref' not in c_key and '_trans' not in c_key:
                child = node["children"][i+1]
                i_map[c_key] = i
                i += 1
            else:
                pi = i_map[cub.parent]
                child = deepcopy(node["children"][pi+1])
                if cub.parent_axis is not None and 'prog' in child:
                    flip_axis = cub.parent_axis                
                    
            # cub is found through local execution, this brings it into global space
            apply_delta(bbox, rbox, cub)
            # if intermediate cuboid, add back into queue
            if child is not None and len(child) > 0:
                q.append((child, cub, flip_axis))
            # if leave cuboid, save these cuboid to the add list
            else:

                add_cubes.append(cub)
                pp = None
                if '#' in c_key:
                    pp = int(c_key.split('#')[1])
                add_pps.append(pp)
                
        scene_cubes += add_cubes
        scene_ppreds += add_pps

    if ret_ppred:
        return scene_cubes, scene_ppreds
        
    return scene_cubes

def make_hier_prog(lines):
    all_progs = {}
    root_name = None
        
    cur_name = None
    cur_prog = []
    cur_children = []
        
    for line in lines:
        if len(line) == 0:
            continue
        ls = line.strip().split()

        if ls[0] == 'Assembly':
            cur_name = ls[1]
            if root_name is None:
                root_name = cur_name
                
        elif ls[0] == '}':
            all_progs[cur_name] = (cur_prog, cur_children)
            cur_children = []
            cur_prog = []
            cur_name = None

        else:
            if 'Cuboid' in line:
                if 'Program_' in line:
                    cur_children.append(ls[0])
                else:
                    cur_children.append(None)
                        
            cur_prog.append(line)

    hp = {'name': root_name}
    
    q = [hp]

    seen = set()

    while(len(q)) > 0:
        node = q.pop(0)
        prog, children = all_progs[node['name']]
        node['prog'] = prog
        node['children'] = []
            
        for child in children:
            c = {}
            if child is not None:
                c = {'name': child}

                if child in seen:
                    assert False, 'repeated name'
                seen.add(child)
                
                q.append(c)
            node['children'].append(c)
            
    return hp

# Execute a program 
def run_sa_prog(lines):
    with torch.no_grad():
        hier_prog = make_hier_prog(lines)
        cubes = hier_execute(hier_prog)
    return cubes

def run_sa_prog_ppred(lines):
    with torch.no_grad():
        hier_prog = make_hier_prog(lines)
        cubes,ppred = hier_execute(hier_prog, ret_ppred=True)

    return cubes, ppred
