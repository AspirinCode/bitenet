from .include import *
# from .process import *
# from .clustering import *
from .bitenet import BiteNet
from pymol import cmd, cgo

normals_array = np.array(
        [[-1, 0, 0],
         [1, 0, 0],
         [0, -1, 0],
         [0, 1, 0],
         [0, 0, -1],
         [0, 0, 1]], np.float32)
position_indices_array = np.array([
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[1, 0, 0], [1, 0, 1], [1, 1, 0]],
        [[1, 0, 1], [1, 1, 1], [1, 1, 0]],
        [[0, 0, 0], [0, 0, 1], [1, 0, 0]],
        [[0, 0, 1], [1, 0, 1], [1, 0, 0]],
        [[0, 1, 0], [0, 1, 1], [1, 1, 0]],
        [[0, 1, 1], [1, 1, 1], [1, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [1, 0, 0]],
        [[0, 1, 0], [1, 1, 0], [1, 0, 0]],
        [[0, 0, 1], [0, 1, 1], [1, 0, 1]],
        [[0, 1, 1], [1, 1, 1], [1, 0, 1]]
        ], np.int)
position_indices_lines_array = np.array([
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1],
         [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1],
         [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]]], np.int)

cmd.set("label_position", [2, 2, 2])
cmd.set("transparency_mode", 3)

color_names_all = ["custom_color_{:d}".format(i) for i in range(len(colors_all))]
color_names     = color_names_all[1:]
color_protein        = colors_all[0]
color_name_protein   = color_names_all[0]
color_grey           = colors_all[-1]
color_name_grey      = color_names_all[-1]
for i in range(len(colors_all)):
    cmd.set_color(color_names_all[i], colors_all[i])

global_cgos = {}
def load_cgo(obj, name, state=1, new=True):
    global global_cgos
    name_ = name + "_" + str(state)
    if name_ in global_cgos and not new:
        obj = global_cgos[name_] + obj
    global_cgos[name_] = obj
    if state > 0:
        cmd.load_cgo(obj, name, state)
    else:
        cmd.load_cgo(obj, name)
    return
def add_cgo(obj, name, state=1, new=True):
    global global_cgos
    name_ = name + "_" + str(state)
    if name_ in global_cgos and not new:
        obj = global_cgos[name_] + obj
    global_cgos[name_] = obj
    return obj
def show_cgo(name):
    names = [k for k in global_cgos.keys() if k[:len(name)] == name]
    for n in names:
        name_ = name
        try:
            state = int(n[len(name)+1:])
            if state > 0:
                cmd.load_cgo(global_cgos[n], name_, state)
            else:
                cmd.load_cgo(global_cgos[n], name_)
        except:
            pass
    return
def delete_cgo(name, state=-1):
    global global_cgos
    if state > 0:
        name_ = name + "_" + str(state)
        global_cgos.pop(name_)
    else:
        for name_ in list(global_cgos.keys()):
            if name == name_[:len(name)]:
                try:
                    s = int(name_[len(name)+1:])
                    global_cgos.pop(name_)
                except:
                    pass
    return

def cgo_tetra_planes(position=[0, 0, 0], sizes=[1, 1, 1], 
    color=[1., 1., 1.], alpha=1.):
    x = [position[0] - sizes[0] / 2., position[0] + sizes[0] / 2.]
    y = [position[1] - sizes[1] / 2., position[1] + sizes[1] / 2.]
    z = [position[2] - sizes[2] / 2., position[2] + sizes[2] / 2.]
    obj = [cgo.BEGIN, cgo.TRIANGLES]
    obj += [cgo.ALPHA, alpha]
    obj += [cgo.COLOR, color[0], color[1], color[2]]
    for j1 in range(6):
        for j2 in range(2):
            obj += [cgo.NORMAL, normals_array[j1, 0], normals_array[j1, 1], normals_array[j1, 2]]
            for j3 in range(3):
                obj += [cgo.VERTEX, x[position_indices_array[2 * j1 + j2][j3][0]],
                        y[position_indices_array[2 * j1 + j2][j3][1]],
                        z[position_indices_array[2 * j1 + j2][j3][2]]]

    obj += [cgo.END]
    return obj

def cgo_cube_planes(position=[0, 0, 0], size=1, color=[1., 1., 1.], alpha=1.):
    return cgo_tetra_planes(position, [size, size, size], color=color, alpha=alpha)

position_indices_lines_array = np.array([
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1],
         [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1],
         [0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]]], np.int)

def cgo_tetra_lines(position=[0, 0, 0], sizes=[1, 1, 1], 
                    color=[1., 1., 1.], alpha=1., width=0.5):
    obj = []
    x = [position[0] - sizes[0] / 2., position[0] + sizes[0] / 2.]
    y = [position[1] - sizes[1] / 2., position[1] + sizes[1] / 2.]
    z = [position[2] - sizes[2] / 2., position[2] + sizes[2] / 2.]
    obj += [cgo.BEGIN, cgo.LINES]
    obj += [cgo.ALPHA, alpha]
    obj += [cgo.LINEWIDTH, width]
    obj += [cgo.COLOR, color[0], color[1], color[2]]
    for i in range(3):
        for j in range(8):
            obj += [cgo.VERTEX, x[position_indices_lines_array[i, j, 0]],
                    y[position_indices_lines_array[i, j, 1]],
                    z[position_indices_lines_array[i, j, 2]]]
    obj += [cgo.END]
    return obj

def cgo_cube_lines(position=[0, 0, 0], size=1, 
    color=[1., 1., 1.], alpha=1., width=0.5):
    return cgo_tetra_lines(position, [size, size, size], 
        color=color, alpha=alpha, width=width)

def cgo_sphere(position=[0, 0, 0], radius=1., color=[1., 1., 1.], alpha=1.):
    obj = [cgo.COLOR, color[0], color[1], color[2]]
    obj += [cgo.ALPHA, alpha]
    obj += [cgo.SPHERE, position[0], position[1], position[2], radius]
    return obj

def cgo_triangle_lines(positions=np.zeros((3, 3)), color=[1., 1., 1.], alpha=1., width=0.5):
    obj = []
    obj += [cgo.BEGIN, cgo.LINE_LOOP]
    obj += [cgo.ALPHA, alpha]
    obj += [cgo.LINEWIDTH, width]
    obj += [cgo.COLOR, color[0], color[1], color[2]]
    obj += [cgo.VERTEX, positions[0, 0], positions[0, 1], positions[0, 2]]
    obj += [cgo.VERTEX, positions[1, 0], positions[1, 1], positions[1, 2]]
    obj += [cgo.VERTEX, positions[2, 0], positions[2, 1], positions[2, 2]]
    obj += [cgo.END]
    return obj

def draw_boxes(boxes, name="boxes", state=1, new=True, **args):
    obj = []
    for box in boxes:
        obj += cgo_tetra_lines(box[:3], box[3:6], **args)
    load_cgo(obj, name, state, new=new)
    return obj

def draw_spheres(positions, radius=1, name="spheres", color=[1, 1, 1], 
    alpha=1, state=1, new=True):
    obj = []
    for i in range(len(positions)):
        if type(radius) in [float, int]:
            r = radius
        else:
            r = radius[i]
        pos = positions[i]
        obj += cgo_sphere(pos, r, color=color, alpha=alpha)
    load_cgo(obj, name, state, new=new)
    return obj

class PredictionDrawer:
    def __init__(self, name="predictions",
        radius=1, color=[1, 1, 1], alpha=1, mode=1,
        labels=True):
        self.name   = name
        self.radius = radius
        self.color  = color
        self.alpha  = alpha
        self.mode   = mode
        self.labels = labels
    
    def draw(self, predictions, name="", state=1, new=True, color=[], alpha=None):
        if len(color) == 0:
            color = self.color
        if alpha == None:
            alpha = self.alpha
        if len(name) == 0:
            name = self.name
        if new:
            cmd.delete(name + "_scores")
        obj = []
        for i, p in enumerate(predictions):
            score = p[0]
            pos = p[1:4]
            c = color
            r = self.radius
            a = alpha
            if self.mode in [1, 4]:
                c = (np.ones((3)) - (np.ones(3) - np.array(color)) * score).tolist()
            if self.mode in [2, 4]:
                r = self.radius * score
            if self.mode == 3:
                a = alpha * score
            n = "pred_{:d}".format(i)
            if self.labels:
                cmd.pseudoatom(name + "_scores", name=n+"_label", 
                    pos=pos.tolist(), label="{:.3f}".format(score),
                    state=state, quiet=True)
                cmd.set("label_position", [2, 2, 2])
                    
            obj += cgo_sphere(pos, r, c, a)
        load_cgo(obj, name=name+"_positions", state=state, new=new)
        return
    
    def __call__(self, *args, **kargs):
        self.draw(*args, **kargs)
 
def grid_add_point(grid, grid_floor, position, score, distance_decay=1, distance_cutoff=4, voxel_size=1):
   distance_cutoff_int = int(math.ceil(distance_cutoff / voxel_size))
   p = position - grid_floor
   gi, gj, gk = [int(math.floor(p[i] / voxel_size)) for i in range(3)]
   for i in range(max(0, gi - distance_cutoff_int), min(grid.shape[0], gi + distance_cutoff_int + 1)):
       for j in range(max(0, gj - distance_cutoff_int), min(grid.shape[1], gj + distance_cutoff_int + 1)):
           for k in range(max(0, gk - distance_cutoff_int), min(grid.shape[2], gk + distance_cutoff_int + 1)):
               coord = np.array([i + 0.5, j + 0.5, k + 0.5]) * voxel_size
               d = distance(coord, p)
               if d <= distance_cutoff:
                   grid[i, j, k] += score * np.exp(-0.5 * d / distance_decay)
   return
def get_predictions_density_grid(center, positions, scores,
   distance_decay=1, distance_cutoff=4,
   grid_size=40, voxel_size=1):
   grid = np.zeros((grid_size, grid_size, grid_size))
   grid_floor = center - np.array([grid.shape[0] / 2, grid.shape[1] / 2, grid.shape[2] / 2]) * voxel_size
   for index in range(len(positions)):
       grid_add_point(grid, grid_floor, positions[index], scores[index],
           distance_decay=distance_decay, distance_cutoff=distance_cutoff, voxel_size=voxel_size)
   return grid
 
if True:
   import ctypes
   current_path = os.path.dirname(os.path.abspath(__file__))
   library_path = os.path.join(current_path, "libs", "libmarchingcubes.so")
   lib = ctypes.cdll.LoadLibrary(library_path)
 
   class GridMarching:
       def __init__(self,
           grid=np.zeros((10, 10, 10)),
           grid_floor=[0, 0, 0],
           voxel_size=1):
      
           lib.grid_marching_new.argtypes = [np.ctypeslib.ndpointer(np.float64),
               ctypes.c_int, ctypes.c_int, ctypes.c_int,
               ctypes.c_double, ctypes.c_double, ctypes.c_double,
               ctypes.c_double]
           lib.grid_marching_new.restype = ctypes.c_void_p
 
           lib.calculate_triangles.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_bool]
           lib.calculate_triangles.restype  = ctypes.c_void_p
 
           lib.get_triangles_number.argtypes = [ctypes.c_void_p]
           lib.get_triangles_number.restype  = ctypes.c_int
          
           lib.get_triangles.argtypes = [ctypes.c_void_p]
 
           lib.grid_marching_del.argtypes = [ctypes.c_void_p]
           lib.grid_marching_del.restype  = ctypes.c_void_p
 
           self.grid = grid.astype(np.float64)
           self.grid_floor = grid_floor
           self.voxel_size = voxel_size
           self.obj = lib.grid_marching_new(self.grid,
               self.grid.shape[0], self.grid.shape[1], self.grid.shape[2],
               float(self.grid_floor[0]), float(self.grid_floor[1]), float(self.grid_floor[2]),
               self.voxel_size)
           self.triangles = None
 
       def calculate_triangles(self, isolevel=0.5, bounds=True):
           lib.calculate_triangles(self.obj, isolevel, bounds)
           triangles_num = lib.get_triangles_number(self.obj)
           if triangles_num > 0:
               lib.get_triangles.restype = np.ctypeslib.ndpointer(dtype=np.float64,
                   shape=(triangles_num, 3, 3))
               return lib.get_triangles(self.obj).astype(np.float32)
           else:
               return np.array([])
      
       def __del__(self):
           lib.grid_marching_del(self.obj)
 
   def cgo_marching_cubes_lines(grid, grid_floor=[0, 0, 0], voxel_size=1., isolevel=0.5, bounds=True, **args):
       grid_marching = GridMarching(grid, grid_floor, voxel_size)
       triangles = grid_marching.calculate_triangles(isolevel=isolevel, bounds=bounds)
       obj = []
       for tr in triangles:
           obj += cgo_triangle_lines(tr, **args)
       return obj
 
   def draw_marching_cubes_lines(grid, grid_floor=[0, 0, 0],
       name="isosurface", state=1, new=True, **args):
       obj = cgo_marching_cubes_lines(grid, grid_floor=grid_floor, **args)
       load_cgo(obj, name, state=state, new=new)
       return obj
 
 
class GridDrawer:
    def __init__(self,
        name            = "grid",
        voxel_size      = 1.,
        threshold       = 0.1,
        mode            = 0,
        draw_bounds     = False,
        color           = [1., 1., 1.],
        color_bounds    = [1., 1., 1.],
        spacing         = 0.,
        alpha           = 1.,
        representation  = cgo_cube_planes,
        all_channels    = False, 
        **args,
        ):
        self.name           = name
        self.voxel_size     = voxel_size
        self.threshold      = threshold
        self.mode           = mode
        self.draw_bounds    = draw_bounds
        self.color          = color
        self.color_bounds   = color_bounds
        self.spacing        = spacing
        self.alpha          = alpha
        self.representation = representation
        self.grid_isolevel_bounds = False
        self.all_channels   = all_channels
        self.channel_colors = colors_channels
    
    def draw(self, grid, floor=[0, 0, 0], name="", state=1,
        channel=None, color=[], clear=True, draw=True):
        if len(name) == 0:
            name = self.name
        if len(grid.shape) > 3:
            if channel == None:
                grid = np.max(grid, axis=-1)
            else:
                grid = grid[:, :, :, channel]
        if len(color) == 0:
            color = self.color
        obj = []
        if self.mode not in [3]:
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    for k in range(grid.shape[2]):
                        value = grid[i, j, k]
                        if value >= self.threshold:
                            c = color
                            size = self.voxel_size - self.spacing
                            alpha = self.alpha
                            if self.mode == 0:
                                size = value * self.voxel_size
                            elif self.mode == 1:
                                pass
                            elif self.mode == 2:
                                alpha = value * self.alpha
                            elif self.mode == 4:
                                c = (np.ones((3)) - (np.ones((3)) - np.array(color)) * value).tolist()
                            elif self.mode == 5:
                                alpha = value * self.alpha
                                c = (np.ones((3)) - (np.ones((3)) - np.array(color)) * value).tolist()
                            position = [
                                (i + 0.5) * self.voxel_size + floor[0],
                                (j + 0.5) * self.voxel_size + floor[1],
                                (k + 0.5) * self.voxel_size + floor[2]]
                            obj += self.representation(position, size=size, color=c, alpha=alpha)
        elif self.mode == 3:
            obj += cgo_marching_cubes_lines(grid, floor,
                voxel_size=self.voxel_size, isolevel=self.threshold,
                bounds=self.grid_isolevel_bounds, alpha=self.alpha, color=color)
        
        if self.draw_bounds:
            position = [
                floor[0] + self.voxel_size * grid.shape[0] / 2.,
                floor[1] + self.voxel_size * grid.shape[1] / 2.,
                floor[2] + self.voxel_size * grid.shape[2] / 2.]
            sizes = [
                self.voxel_size * grid.shape[0],
                self.voxel_size * grid.shape[1],
                self.voxel_size * grid.shape[2]]
            obj += cgo_tetra_lines(position, sizes, color=self.color_bounds)
        if draw:
            load_cgo(obj, name, state, new=clear)
        return obj
    
    def __call__(self, *args, **kargs):
        if self.all_channels:
            return self.draw_channels(*args, **kargs)
        else:
            return self.draw(*args, **kargs)

    def heatmap(self, grid, floor=[0, 0, 0], 
        threshold_neg=-0.01,    threshold_pos=0.01,
        max_value_neg=-1.,      max_value_pos=-1,
        color_neg=[0, 0, 1],    color_pos=[1, 0, 0],
        draw=True, name="", state=1, new=True):
        grid = np.clip(grid, max_value_neg, max_value_pos)
        if len(name) == 0:
            name = self.name
        obj = []
        alpha = self.alpha
        size  = self.voxel_size - self.spacing
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                for k in range(grid.shape[2]):
                    value = grid[i, j, k]
                    if value > threshold_neg and value < threshold_pos:
                        continue
                    if value < 0:
                        c = (np.ones((3)) - (np.ones((3)) - np.array(color_neg)) * value / max_value_neg).tolist()
                        v = value / max_value_neg 
                    else:
                        c = (np.ones((3)) - (np.ones((3)) - np.array(color_pos)) * value / max_value_pos).tolist()
                        v = value / max_value_pos
                    if self.mode == 0:
                        size = v * self.voxel_size
                    elif self.mode in [2, 5]:
                        alpha = v * self.alpha
                    position = [
                        (i + 0.5) * self.voxel_size + floor[0],
                        (j + 0.5) * self.voxel_size + floor[1],
                        (k + 0.5) * self.voxel_size + floor[2]]
                    obj += self.representation(position, size=size, color=c, alpha=alpha)
        if draw:
            load_cgo(obj, name=name, state=state, new=new)
        return obj

    def draw_channels(self, grid, floor=[0, 0, 0],
            name="", state=1, new=True, draw=True):
        if len(name) == 0:
            name = self.name
        obj = []
        for channel in range(grid.shape[-1]):
            obj += self.draw(grid, floor, channel=channel, 
                draw=False, color=self.channel_colors[channel])
        if draw:
            load_cgo(obj, name=name, state=state, new=new)
        return obj


class BiteNet_Draw(BiteNet):
    def __init__(self, 
            working_dir     = "./", 
            draw_grid       = False,
            draw_interfaces = False,
        **args):
        super(BiteNet_Draw, self).__init__(**args)
        self.drawer             = GridDrawer(**self.dataloader.get_params())
        self.prediction_drawer  = PredictionDrawer(color=[1, 0, 0])
        self.working_dir        = working_dir
        self.draw_grid          = draw_grid
        self.draw_interfaces    = draw_interfaces

    def predict_draw(self, filename, draw_grid=False, draw_interfaces=False, **args):
        if os.path.isfile(filename):
            name = os.path.basename(filename).replace(".pdb", "")
        elif filename in cmd.get_object_list():
            name = filename
            filename = os.path.join(self.working_dir, "temp.pdb")
            cmd.save(filename, name)
        else:
            print("no such file or object: {:s}".format(filename))
            return [], []

        _, _, _, predictions, residues = self.get_grids_predictions(filename)
        self.prediction_drawer(predictions, name=name + "_pred")
        self.draw(filename, draw_grid=draw_grid, draw_interfaces=draw_interfaces, 
            name=name, **args)

        return predictions, residues

    def draw(self, filename, draw_grid=True, draw_interfaces=False, name="", **args):
        grid, floors, interfaces = self.dataloader.get_single_grid(filename)
        if len(name) == 0:
            name = os.path.basename(filename).replace(".pdb", "")
        if self.draw_grid or draw_grid:
            self.drawer(grid, floors, name=name+"_grid", **args)
        if self.draw_interfaces or draw_interfaces:
            draw_boxes(interfaces, name=name+"_interfaces", color=[0, 0, 1])

    def __call__(self, filename, **args):
        return self.predict_draw(filename, **args)
        

def draw_clusters_predictions(clustering, 
        radius      = 1,
        new         = True,
        name        = "clusters_predictions", 
        state       = 1,
        colors      = colors,
    ):
    if new:
        delete_cgo(name)
    for step, indexes_s in enumerate(clustering.indexes):
        for i, index in enumerate(indexes_s):
            if index >= 0:
                label = clustering.labels_[index]
                if label in clustering.cluster_indexes_:
                    cluster_index = np.where(clustering.cluster_indexes_ == label)[0][0]
                else:
                    cluster_index = -1
                color = colors[min(len(colors) - 1, cluster_index)]
                add_cgo(cgo_sphere(
                    clustering.predictions[step][i, 1:4], 
                    radius * clustering.predictions[step][i, 0],
                    color = color), name=name, state=state, new=False)
    show_cgo(name)
    return

def draw_clusters_isolevel(clustering,
        voxel_size      = 1,
        grid_size       = 20,
        isolevel        = 0.001,
        distance_cutoff = 2,
        distance_decay  = 2,
        all             = True,
        name    = "clusters_isolevel",
        new     = True,
        state   = 1,
        colors  = colors,
    ):
    if new:
        delete_cgo(name)
    for i, cluster_index in enumerate(clustering.cluster_indexes):
        if all:
            predictions = clustering.get_predictions(cluster_index)
            positions_c = predictions[:, 1:4]
            scores_c    = predictions[:, 0]
        else:
            indexes = np.where(clustering.labels_ == cluster_index)[0]
            positions_c = clustering.predictions_[indexes, 1:4]
            scores_c    = clustering.predictions_[indexes, 0]
        color = colors[min(len(colors) - 1, i)]
        grid_center = np.mean(positions_c, axis=0)
        grid_density = get_predictions_density_grid(grid_center,
            positions_c, scores_c, grid_size=grid_size, voxel_size=voxel_size,
            distance_cutoff=distance_cutoff, distance_decay=distance_decay)
        grid_floor = grid_center - \
            np.array([grid_density.shape[0] / 2, grid_density.shape[1] / 2, grid_density.shape[2] / 2]) * voxel_size
        grid_density /= len(clustering.predictions_)
        add_cgo(cgo_marching_cubes_lines(grid_density, grid_floor, 
            voxel_size=voxel_size, isolevel=isolevel, color=color),
            name=name, state=state, new=False)
    show_cgo(name)
    return 

def draw_clusters_density(clustering,
        voxel_size      = 0.5,
        grid_size       = 100,
        distance_cutoff = 2,
        distance_decay  = 2,
        threshold       = 0.02,
        mode            = 2,
        all             = True,
        norm_by         = "max",
        mult_by         = 0.9,
        name    = "clusters_density",
        new     = True,
        state   = 1,
        colors  = colors,
    ):
    if new:
        delete_cgo(name)
    drawer = GridDrawer(name=name, voxel_size=voxel_size,
        threshold=threshold, mode=mode)
    for i, cluster_index in enumerate(clustering.cluster_indexes):
        if all:
            predictions = clustering.get_predictions(cluster_index)
            positions_c = predictions[:, 1:4]
            scores_c    = predictions[:, 0]
        else:
            indexes = np.where(clustering.labels_ == cluster_index)[0]
            positions_c = clustering.predictions_[indexes, 1:4]
            scores_c    = clustering.predictions_[indexes, 0]
        color = colors[min(len(colors) - 1, i)]
        grid_center = np.mean(positions_c, axis=0)
        grid_density = get_predictions_density_grid(grid_center,
            positions_c, scores_c, grid_size=grid_size, voxel_size=voxel_size,
            distance_cutoff=distance_cutoff, distance_decay=distance_decay)
        grid_floor = grid_center - \
            np.array([grid_density.shape[0] / 2, grid_density.shape[1] / 2, grid_density.shape[2] / 2]) * voxel_size
        if norm_by == "max":
            grid_density /= np.max(grid_density) * 1.01
        elif norm_by == "sigmoid":
            grid_density = 2 / (1 + np.exp(-grid_density)) - 1
        grid_density *= mult_by
        add_cgo(drawer(grid_density, grid_floor, color=color, draw=False),
            name=name, state=state, new=False)
    show_cgo(name)
    return

def draw_clusters_centers_steps(clustering,
        norm        = 5,
        radius      = 7.13,
        alpha       = 1,
        threshold   = 0.01,
        name    = "clusters_centers_steps",
        new     = True,
        colors  = colors,
    ):
    if new:
        delete_cgo(name)
    cluster_scores = clustering.cluster_scores_steps
    cluster_centers_steps = clustering.get_cluster_centers_steps()
    if norm > 1:
        cluster_scores = moving_average(cluster_scores, n=(norm-1))
        cluster_centers_steps = moving_average(cluster_centers_steps, n=(norm-1))

    for step in range(cluster_scores.shape[0]):
        for i, cluster_index in enumerate(clustering.cluster_indexes):
            color = colors[min(len(colors) - 1, i)]
            center = cluster_centers_steps[step, cluster_index, :]
            if cluster_scores[step, i] >= threshold:
                add_cgo(cgo_sphere(center, radius * cluster_scores[step, cluster_index],
                    alpha=alpha, color=color),
                name=name, state=step+1, new=False)
    show_cgo(name)
    return

def draw_clusters_predictions_steps(clustering,
        step_decay  = 50,
        radius      = 1,
        name    = "clusters_predictions_steps",
        new     = True,
        colors  = colors,
    ):
    if new:
        delete_cgo(name)
    for step in range(len(clustering.predictions)):
        for dec in range(step_decay):
            if step - dec >= 0:
                for i, p in enumerate(clustering.predictions[step - dec]):
                    cluster_index = np.where(clustering.cluster_indexes_ == \
                        clustering.labels_[clustering.indexes[step - dec][i]])[0][0]
                    color = colors[min(len(colors) - 1, cluster_index)]
                    add_cgo(cgo_sphere(p[1:4], radius * p[0], 
                        color=color, alpha=(step_decay - dec)/step_decay),
                    name=name, state=step+1, new=False)
    show_cgo(name)
    return

def draw_clusters_isolevel_steps(clustering,
        voxel_size      = 2,
        grid_size       = 10,
        distance_decay  = 1,
        distance_cutoff = 4,
        step_decay      = 10,
        isolevel        = 0.01,
        name    = "clusters_isolevel_steps",
        new     = True,
        colors  = colors, 
    ):
    if new:
        delete_cgo(name)
    centers = clustering.cluster_centers_[clustering.cluster_indexes]
    cluster_grids = [np.zeros((grid_size, grid_size, grid_size)) 
        for i in range(len(clustering.cluster_indexes))]
    cluster_grid_floors = [centers[i] - np.array([grid_size / 2, grid_size / 2, grid_size / 2]) * voxel_size 
        for i in range(len(cluster_grids))]
    for step in range(len(clustering.predictions)):
        for g in cluster_grids:
            g *= (step_decay - 1) / step_decay
        for i, index in enumerate(clustering.indexes[step]):
            if index >= 0:
                label = clustering.labels_[index]
                if label in clustering.cluster_indexes:
                    cluster_index = np.where(clustering.cluster_indexes == label)[0][0]
                    grid_add_point(cluster_grids[cluster_index], cluster_grid_floors[cluster_index],
                        clustering.predictions[step][i, 1:4], clustering.predictions[step][i, 0],
                        distance_cutoff=distance_cutoff, distance_decay=distance_decay, voxel_size=voxel_size)
        for index in range(len(cluster_grids)):
            color = colors[min(len(colors) - 1, index)]
            add_cgo(cgo_marching_cubes_lines(cluster_grids[index], cluster_grid_floors[index],
                voxel_size=voxel_size, isolevel=isolevel, color=color),
            name=name, state=step+1, new=False)
    show_cgo(name)
    return

def draw_clusters_density_steps(clustering,
        grid_size       = 20,
        voxel_size      = 1,
        distance_decay  = 1,
        distance_cutoff = 4,
        step_decay      = 10,
        threshold       = 0.01,
        mode            = 2,
        name    = "clusters_density_steps",
        new     = True,
        colors  = colors,
    ):
    if new:
        delete_cgo(name)
    centers = clustering.cluster_centers_[clustering.cluster_indexes]
    cluster_grids = [np.zeros((grid_size, grid_size, grid_size)) 
        for i in range(len(clustering.cluster_indexes))]
    cluster_grid_floors = [centers[i] - np.array([grid_size / 2, grid_size / 2, grid_size / 2]) * voxel_size 
        for i in range(len(cluster_grids))]
    drawer = GridDrawer(name=name, voxel_size=voxel_size, threshold=threshold, mode=mode)
    for step in range(len(clustering.predictions)):
        for g in cluster_grids:
            g *= (step_decay - 1) / step_decay
        for i, index in enumerate(clustering.indexes[step]):
            if index >= 0:
                label = clustering.labels_[index]
                if label in clustering.cluster_indexes:
                    cluster_index = np.where(clustering.cluster_indexes == label)[0][0]
                    grid_add_point(cluster_grids[cluster_index], cluster_grid_floors[cluster_index],
                        clustering.predictions[step][i, 1:4], clustering.predictions[step][i, 0],
                        distance_cutoff=distance_cutoff, distance_decay=distance_decay, voxel_size=voxel_size)
        for index in range(len(cluster_grids)):
            color = colors[min(len(colors) - 1, index)]
            add_cgo(drawer(cluster_grids[index], cluster_grid_floors[index],
                color=color, draw=False), name=name, state=step+1, new=False)
    show_cgo(name)
    return

def color_cluster_residues(clustering, 
        name_object     = "",
        color_protein   = color_name_protein,
        colors          = color_names,
        all             = True,
        score_threshold = 1.,
    ):
    if len(name_object) == 0:
        name_object = cmd.get_object_list()[0]
        name_object = "all"
    cmd.color(color_protein, name_object)
    res_dict = {}
    if all:
        for step, indexes_s in enumerate(clustering.indexes):
            for i, index in enumerate(indexes_s):
                if index >= 0 and clustering.labels_[index] >= 0:
                    res = clustering.residues[step][i]
                    for r in res:
                        if r not in res_dict:
                            res_dict[r] = np.zeros((clustering.n_clusters_))
                        res_dict[r][clustering.labels_[index]] += clustering.predictions[step][i, 0] / len(res)
    else:
        for i, label in enumerate(clustering.labels_):
            if label >= 0:
                res = clustering.residues_[i]
                for r in res:
                    if r not in res_dict:
                        res_dict[r] = np.zeros((clustering.n_clusters_))
                    res_dict[r][label] += clustering.predictions_[i, 0] / len(res)
        
    for r, values in res_dict.items():
        if np.max(values) < score_threshold:
            continue
        selection_string = [name_object, 
            "resi " + r.split("_")[1], "resn " + r.split("_")[2]]
        if len(r.split("_")[0]) > 0:
            selection_string += ["chain " + r.split("_")[0]]
        selection_string = " and ".join(selection_string)
        cluster_index = np.where(clustering.cluster_indexes_ == np.argmax(values))[0][0]
        color_name = colors[min(len(colors) - 1, cluster_index)]
        cmd.color(color_name, selection_string)
    