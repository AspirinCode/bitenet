from .include import *
from multiprocessing import Pool

from . import descriptor

filter_cube_value = 0.0001      # cubes with mean density <= filter_cube_value are not fed into nn

def boxes_to_labels(boxes, cube_size=default_cube_size, cell_size=default_cell_size):
    """
    Converting interface boxes to cell labels.
    Args:
        boxes: np.array of boxes, shape=(None, 6); each box is true interface; 
            box[:3] are coordinates of center, box[4:6] are box sizes.
        cube_size: int; default=64; cubic grid size in voxels.
        cell_size: int; default=8; cell size in voxels.
    Returns:
        np.array of shape (N, N, N, 4), where N is number of cells along axis, N=cube_size//cell_size;
        labels for cells; labels[:, :, :, 0] are confidence (0 if interface box center in cell, 1 otherwise), 
        labels[:, :, :, 1:4] are relative coordinates (in [0, 1] if interface box center in cell, 0 otherwise).
    """
    N = cube_size // cell_size
    labels = np.zeros((N, N, N, 4), dtype=np.float32)
    for box in boxes:
        cell_index = np.array(np.floor_divide(box[0:3], cell_size), dtype=np.int32)
        box_position = np.divide(box[0:3], cell_size)
        labels[cell_index[0], cell_index[1], cell_index[2], 0] = 1.
        labels[cell_index[0], cell_index[1], cell_index[2], 1] = box_position[0]
        labels[cell_index[0], cell_index[1], cell_index[2], 2] = box_position[1]
        labels[cell_index[0], cell_index[1], cell_index[2], 3] = box_position[2]
    return labels

def split_grid_to_cubes(grid, interfaces=[], 
    cube_size   = default_cube_size, 
    cell_size   = default_cell_size,
    stride      = default_stride, 
    train       = False,
    stride_full = True):
    """
    Splitting descriptors output grid of arbitrary size into cubic grids.
    Args:
        grid: np.array of shape (N1, N2, N3, 11).
        interfaces: np.array of shape (None, 6); list of interface boxes;
        cube_size: int; default=64; cubic grid size;
        cell_size: int; default=8; cell size;
        stride: int; default=32; stride for splitting;
        train: bool; default=False; if True, initial strides are random;
        stride_full: bool; default=True: if False, all cube grids will be slices.
    Returns:
        (grid_cubes, target_labels, floors_cubes)
        grid_cubes: list of grids (np.array of shape (cube_size, cube_size, cube_size, 11)).
        target_labels: list of target labels (np.array of shape (N, N, N, 11) where N is number of cells);
        floors_cubes: list of grids origin coordinates;
    """

    grid_shape = grid.shape[0:3]
    grid_shape_new = [max(int(math.ceil(float(shape) / stride)) * stride, cube_size) for shape in grid_shape]

    start_indices = [0, 0, 0]
    if train:
        start_indices[0] = random.randint(0, grid_shape_new[0] - grid_shape[0])
        start_indices[1] = random.randint(0, grid_shape_new[1] - grid_shape[1])
        start_indices[2] = random.randint(0, grid_shape_new[2] - grid_shape[2])
    else:
        start_indices[0] = (grid_shape_new[0] - grid_shape[0]) // 2
        start_indices[1] = (grid_shape_new[1] - grid_shape[1]) // 2
        start_indices[2] = (grid_shape_new[2] - grid_shape[2]) // 2
    
    grids_cubes = []
    target_labels = []
    floors_cubes = []
    # number of cubes strides for each axis
    number_of_cubes = [int((size-(cube_size-stride))/stride) for size in grid_shape_new]
    for i in range(number_of_cubes[0]):
        for j in range(number_of_cubes[1]):
            for k in range(number_of_cubes[2]):
                cube_floor = [-start_indices[0] + stride * i, 
                            -start_indices[1] + stride * j, 
                            -start_indices[2] + stride * k]

                if not train and not stride_full:
                    for index in range(3):
                        if not (cube_floor[index] < 0 and cube_floor[index] + cube_size > grid.shape[index]):
                            cube_floor[index] = max(cube_floor[index], 0)
                            cube_floor[index] = min(cube_floor[index], grid.shape[index] - cube_size)
                
                cube_interfaces = []
                for interface_index in range(len(interfaces)):
                    interface = interfaces[interface_index]
                    cube_inter = interface[:]
                    # moving interface position for cube
                    cube_inter[0] -= cube_floor[0]
                    cube_inter[1] -= cube_floor[1]
                    cube_inter[2] -= cube_floor[2]
                    # if interface is in cube than add to cube interfaces 
                    if (cube_inter[0] >= 0) and (cube_inter[0] < cube_size) and \
                    (cube_inter[1] >= 0) and (cube_inter[1] < cube_size) and \
                    (cube_inter[2] >= 0) and (cube_inter[2] < cube_size):
                        cube_interfaces.append(cube_inter)

                # setting grid for cube
                if cube_floor[0] >= 0 and cube_floor[1] >= 0 and cube_floor[2] >= 0 and \
                    cube_floor[0] + cube_size <= grid.shape[0] and cube_floor[1] + cube_size <= grid.shape[1] and \
                        cube_floor[2] + cube_size <= grid.shape[2]:
                    grid_cube = grid[cube_floor[0] : cube_floor[0] + cube_size, 
                                    cube_floor[1] : cube_floor[1] + cube_size,
                                    cube_floor[2] : cube_floor[2] + cube_size, :]
                else:
                    grid_cube = np.zeros((cube_size, cube_size, cube_size, grid.shape[3]), np.float32)
                    index_cube_floor = [0, 0, 0]
                    index_cube_ceil = [cube_size, cube_size, cube_size]
                    if cube_floor[0] < 0:
                        index_cube_floor[0] = - cube_floor[0]
                    if cube_floor[1] < 0:
                        index_cube_floor[1] = - cube_floor[1]
                    if cube_floor[2] < 0:
                        index_cube_floor[2] = - cube_floor[2]
                    if cube_floor[0] + cube_size > grid.shape[0]:
                        index_cube_ceil[0] = - cube_floor[0] + grid.shape[0]
                    if cube_floor[1] + cube_size > grid.shape[1]:
                        index_cube_ceil[1] = - cube_floor[1] + grid.shape[1]
                    if cube_floor[2] + cube_size > grid.shape[2]:
                        index_cube_ceil[2] = - cube_floor[2] + grid.shape[2]
                    grid_cube[index_cube_floor[0] : index_cube_ceil[0], 
                            index_cube_floor[1] : index_cube_ceil[1], index_cube_floor[2] : index_cube_ceil[2], :] = \
                    grid[max(0, cube_floor[0]) : min(grid.shape[0], cube_floor[0] + cube_size),
                        max(0, cube_floor[1]) : min(grid.shape[1], cube_floor[1] + cube_size),
                        max(0, cube_floor[2]) : min(grid.shape[2], cube_floor[2] + cube_size), :]
                
                # check number of nonzero voxels in grid, if number is too small than grid is ignored
                zero_concentration = float(np.count_nonzero(np.sum(np.abs(grid_cube), axis=-1)))
                if zero_concentration / (cube_size * cube_size * cube_size) < filter_cube_value:
                    continue
                
                grids_cubes.append(grid_cube)
                target_labels.append(boxes_to_labels(cube_interfaces, cube_size, cell_size))
                floors_cubes.append(cube_floor)
                
    return grids_cubes, target_labels, floors_cubes

def get_single_grid(filename, 
        voxel_size      = default_voxel_size,
        density_cutoff  = default_density_cutoff,
        threads         = 8, **args):
    """
    Computes grid and interfaces for protein pdb file.
    Args:
        filename: str; path to pdb file;
        voxel_size: float; voxel size in angstroms;
        density_cutoff: float; threshold for calculating atom density in angstroms
    Returns:
        (grid, grid_floor, interface)
        grid: np.array of shape (cube_size_x, cube_size_y, cube_size_z, 11)
        grid_floor: list [3] of floats, coordinates of grid origins
        interface: np.array of shape (n_interfaces, 6); interface boxes
    """
    grid, grid_floor, interface = descriptor.pdb_grid_interfaces(filename,
        voxel_size=voxel_size, density_cutoff=density_cutoff, threads=threads, **args)
    return grid, grid_floor, interface

def get_interfaces(filename, **args):
    """
    Get interface boxes for pdb file
    Args:
        filename: str; path to pdb file;
    Returns:
        interface: np.array of shape (n_interfaces, 6); interface boxes
    """
    pdb = descriptor.PDBfile(filename, **args)
    pdb.compute_boxes()
    return pdb.export_interfaces()

def get_grids(filename,
        voxel_size      = default_voxel_size,
        density_cutoff  = default_density_cutoff,
        cube_size       = default_cube_size,
        cell_size       = default_cell_size,
        stride          = default_stride,
        train           = False,
        rotation        = False,
        threads         = 8,
        stride_full     = True,
        rotation_angles = [0, 0, 0],
        **args):
    """
    Calculates interfaces, splitted cubic grids and target labels for pdb file.

    Args:
        filename: str;
            path to input pdb file;
        voxel_size: float; default=1.;
            size of voxel (Angstroms);
        density_cutoff: float; default=2.;
            distance threshold for atom density calculation (Angstroms);
        cube_size: int; default=64;
            cubic grid size in voxels;
        cell_size: int; default=8;
            cell size in voxels;
        stride: int; default=32;
            stride for grid splitting to cubes, in voxels;
        train: bool; default=False;
            if True augmentation added during splitting;
        rotation: bool; default=False;
            if True random rotation angles are set, else given rotation angles are used;
        threads: int; default=8;
            number of threads for voxel grid initialization and parametrization;
        stride_full: bool; default=True;
            if True cubic grids are equally strided during splitting, 
            else cubic grids are moved so they are inside initial grid.
            If train is True, this parameter is not used;
        rotation_angles: list of floats; [theta, phi, psi];
            protein is rotated for psi around (cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta));
        **args: additional arguments for PDBfile and Descriptor classes.
    
    Returns: (grids, target_labels, floors, interface):
        grids: list of np.array of shape (cube_size, cube_size, cube_size, n_channels);
            cubic voxel grids;
        target_labels: list of (np.array of shape (N, N, N, 11) where N is number of cells);
            true labels for train;
        floors: list of np.array of shape (3);
            origin coordinates for each cubic grid;
        interface: np.array of shape (n_interfaces, 6);
            interface boxes;
    """

    # rotation angles
    if rotation:
        rotation_angles_ = [
            math.acos(2 * random.random() - 1.),
            2 * math.pi * random.random(),
            2 * math.pi * random.random()]
    else:
        rotation_angles_ = rotation_angles
    # calculating interface boxes and voxel grid
    grid, grid_floor, interface = descriptor.pdb_grid_interfaces(filename, 
        rotation_angles=rotation_angles, threads=threads,
        voxel_size=voxel_size, density_cutoff=density_cutoff, 
        **args)
    # normalizing interface boxes
    interface_normalized = []
    for int_box in interface:
        int_box_norm = [
            (int_box[0] - grid_floor[0]) / voxel_size,
            (int_box[1] - grid_floor[1]) / voxel_size,
            (int_box[2] - grid_floor[2]) / voxel_size,
            int_box[3] / voxel_size,
            int_box[4] / voxel_size,
            int_box[5] / voxel_size
        ]
        interface_normalized.append(int_box_norm)
    # splitting voxel grid
    grids, target_labels, floors = split_grid_to_cubes(grid,
        interface_normalized, cube_size, cell_size, stride, 
        train=train, stride_full=stride_full)
    # normalizing cubic grid floors
    floors = [
        [f[0] * voxel_size + grid_floor[0],
        f[1] * voxel_size + grid_floor[1],
        f[2] * voxel_size + grid_floor[2]] + rotation_angles_ for f in floors]
    return grids, target_labels, floors, interface

def get_grids_p(args):
    """
    get_grids for multiprocessing
    """
    res = get_grids(*args)
    if len(res[0]) > 100:
        # multiprocessing cannot pass too large memory
        return [], [], [], []
    return res

class DataLoader:
    """
    DataLoader

    Class for loading of grid batches

    Args:
        path: str; default="";
            path to folder with pdb files;
            if pdbs is not set, all pdbs from folder will be used;
        pdbs: list of str; 
            names of pdb files;
        voxel_size: float; default=1.;
            voxel size (Angstroms);
        density_cutoff: float; default=2.;
            distance threshold for atom density calculation (Angstroms);
        cube_size: int; default=64;
            cubic grid size in voxels;
        cell_size: int; default=8;
            cell size in voxels;
        stride: int; default=32;
            stride for grid splitting in voxels;
        rotation: bool; default=True;
            if True, random rotations are added during training;
        threads: int; default=1;
            number of threads for multiprocessing;
        shuffle: bool; default=True;
            if True, pdbs are shuffled in the beginning of each epoch;
        random_seed: int; default=0;
            random seed for random;
        train: bool; default=False;
            if True, augmentations are added;
        batch_size: int; default=8;
            number of pdb files in each step;
        threads_desc: int; default=8;
            number of threads for grid initialization and calculation;
        stride_full: bool; default=False;
            if True, cubic grids are equally strided during splitting, 
            else cubic grids are moved so they are inside initial grid;
        rotation_eval: bool; default=False;
            if True, additional rotations for prediction are added.

    """
    def __init__(self, path="", pdbs=[], 
        voxel_size      = default_voxel_size, 
        density_cutoff  = default_density_cutoff,
        cube_size       = default_cube_size,
        cell_size       = default_cell_size,
        stride          = default_stride,
        rotation        = True,
        threads         = 1,
        shuffle         = True,
        random_seed     = 0,
        train           = False,
        batch_size      = 8, 
        threads_desc    = 8,
        stride_full     = False,
        rotation_eval   = False,
        **args):

        self.path = path
        if len(pdbs) == 0:
            self.pdbs = self.get_pdb_list()
        else:
            self.pdbs = [p.replace(".pdb", "") for p in pdbs]
        self.voxel_size     = voxel_size
        self.density_cutoff = density_cutoff
        self.cube_size      = cube_size
        self.cell_size      = cell_size
        self.stride         = stride
        self.rotation       = rotation
        self.pool           = None
        self.threads        = threads
        self.shuffle        = shuffle
        self.train          = train
        self.batch_size     = batch_size
        self.threads_desc   = threads_desc
        self.stride_full    = stride_full
        self.args           = args
        random.seed(random_seed)

        self.rotation_eval          = rotation_eval
        self.rotation_axes          = default_rotation_axes
        self.rotation_axes_num      = -1
        self.rotation_angles_num    = 5
        self.rotation_random        = False
        self.rotation_random_num    = 20
        
    def get_pdb_list(self, sort=True):
        """
        Reading pdb files in folder.
        Args:
            sort: bool; default=True;
                if True, pdb file names are sorted.
        Returns:
            pdbs: list of str; pdb file names in folder;
        """
        pdbs = []
        if os.path.isdir(self.path):
            pdbs = [f.name[:-4] for f in os.scandir(self.path) if f.name[-4:] == ".pdb"]
        if sort:
            try:
                pdbs.sort(key=lambda p: int(p))
            except:
                pdbs.sort()
        return pdbs

    def start(self):
        """
        Opens mulriprocessing pool if self.threads > 1.
        """
        self.close()
        if self.threads != 1:
            self.pool = Pool(processes=self.threads)

    def close(self):
        """
        Closes multiprocessing pool if self.threads > 1.
        """
        if self.pool != None:
            self.pool.close()
            self.pool.join()

    def set_params(self, **args):
        """
        Sets class parameters.
        """
        for name, arg in args.items():
            if name in self.__dict__.keys():
                self.__dict__[name] = arg
        if "path" in args and "pdbs" not in args:
            self.pdbs = self.get_pdb_list()

    def get_params(self):
        """
        Returns class parameters.
        """
        a = {}
        a.update(self.__dict__)
        return a

    def get_filename(self, name):
        """
        Returns path for file with name in folder self.path.
        """
        if name[-4:] != ".pdb":
            name += ".pdb"
        return os.path.join(self.path, name)
    
    def get_filename_list(self, names):
        return [self.get_filename(name) for name in names]
    
    def sample_rotations(self):
        """
        Returns rotation angles if self.rotation_eval is True.
        Each rotation is [theta, phi, psi].
        Proteins are rotated for psi around (cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)).
        If self.rotation_random is True, self.rotation_random_num of random rotations are used;
        else self.rotation_angles_num rotations around self.rotation_axes_num specified axes are used.
        """
        rotations = []
        if not self.rotation_eval:
            return []
        elif self.rotation_random:
            for i in range(self.rotation_random_num):
                rotations.append([
                    math.acos(2 * random.random() - 1.),
                    2 * math.pi * random.random(),
                    2 * math.pi * random.random()])
        else:
            if self.rotation_axes_num <= 0:
                rotation_axes_num = len(self.rotation_axes)
            else:
                rotation_axes_num = self.rotation_axes_num
            for index_ax in range(rotation_axes_num):
                ax = self.rotation_axes[index_ax]
                theta = math.acos(ax[2])
                phi   = math.asin(ax[1] / math.sqrt(1 - ax[2]**2))
                for index_angle in range(self.rotation_angles_num):
                    psi = 2 * math.pi * (index_angle + 1) / (self.rotation_angles_num + 1)
                    rotations.append([theta, phi, psi])
        return rotations

    def get_grids(self, name, train=False, rotation_angles=[0, 0, 0]):
        """
        Calculates grids and interfaces.
        Calls get_grids function.

        Args:
            name: str; pdb name;
            train: bool; default=False; 
                if True augmentations are added;
            rotation_angles: list of floats; [theta, phi, psi]; default=[0, 0, 0];
                protein will be rotated for angles;
        
        Returns: (grids, target_labels, floors, interface):
            grids: list of np.array of shape (cube_size, cube_size, cube_size, n_channels);
                cubic voxel grids;
            target_labels: list of (np.array of shape (N, N, N, 11) where N is number of cells);
                true labels for train;
            floors: list of np.array of shape (3);
                origin coordinates for each cubic grid;
            interface: np.array of shape (n_interfaces, 6);
                interface boxes;
        """
        return get_grids(self.get_filename(name),
            voxel_size  = self.voxel_size,
            cube_size   = self.cube_size,
            cell_size   = self.cell_size,
            stride      = self.stride,
            train       = train, 
            rotation    = self.rotation and train, 
            threads     = self.threads_desc,
            stride_full = self.stride_full,
            rotation_angles = rotation_angles,
            **self.args)

    def get_single_grid(self, name):
        """
        Computes grid and interfaces for protein pdb file.
        Calls get_single_grid function.
        Args:
            name: str; pdb name;
        Returns:
            (grid, grid_floor, interface)
            grid: np.array of shape (cube_size_x, cube_size_y, cube_size_z, 11)
            grid_floor: list [3] of floats, coordinates of grid origins
            interface: np.array of shape (n_interfaces, 6); interface boxes
        """
        return get_single_grid(self.get_filename(name), 
            voxel_size      = self.voxel_size,
            density_cutoff  = self.density_cutoff,
            threads         = self.threads_desc,
            **self.args)
    def get_interfaces(self, name):
        """
        Returns interface boxes for pdb file.
        """
        return get_interfaces(self.get_filename(name), **self.args)

    def get_batch(self, names, train=False):
        """
        Calculates batch of grids and labels.
        
        Args:
            names: list of str; 
                pdb names;
            train: bool; default=False;
                if True augmentations are added;
        Returns: (grids, target_labels, floors, interfaces, names)
            grids: np.array of shape (n_grids, cube_size, cube_size, cube_size, n_channels);
                batch of cubic voxel grids for model input;
            target_labels: np.array of shape (n_grids, N, N, N, 4) where N is number of cells;
                labels for model cost function;
            floors: list of lists of np.array of shape (6); ((n_files, None, 6));
                there is a list of np.array for each input pdb;
                each np.array corresponds to single cubic grids:
                first three values are grid origins and others are rotation angles;
                rotation angles are not zero only if it is not train and self.rotation_eval is True;
            interfaces: list of lists of np.array of shape (6); ((n_files, None, 6));
                interface boxes for each input file;
            names: list of str;
                list of pdb names.
        """
        if type(names) == str:
            names = [names]
        grids, interfaces, target_labels, floors = [], [], [], []

        if self.threads == 1:
            for name in names:
                floors_single_set = []
                rotation_angles = [[0, 0, 0]] + self.sample_rotations()
                for i in range(len(rotation_angles)):
                    grids_single, targets_single, floors_single, interfaces_single = \
                        self.get_grids(name, train=train, rotation_angles=rotation_angles[i])
                    grids += grids_single
                    if np.sum(np.abs(rotation_angles[i])) == 0:
                        interfaces.append(interfaces_single)
                    target_labels += targets_single
                    floors_single_set += floors_single
                floors.append(floors_single_set)
        else:
            args = []
            for name in names:
                rotation_angles = [[0, 0, 0]] + self.sample_rotations()
                for i in range(len(rotation_angles)):
                    args.append([self.get_filename(name), 
                        self.voxel_size, self.density_cutoff, 
                        self.cube_size, self.cell_size,
                        self.stride, train, self.rotation and train, 
                        self.threads_desc, self.stride_full, 
                        rotation_angles[i]])
            last_name = ""
            for i, res in enumerate(self.pool.imap(get_grids_p, args)):
                if len(res[0]) == 0:
                    res = get_grids_p(args[i])
                grids += res[0]
                target_labels += res[1]
                if args[i][0] == last_name:
                    floors[-1] += res[2]
                else:
                    floors.append(res[2])
                if np.sum(np.abs(args[i][-1])) == 0:
                    interfaces.append(np.array(res[3]))
                last_name = args[i][0]
        return np.array(grids), np.array(target_labels), floors, interfaces, names

    def __call__(self, names, **args):
        return self.get_batch(names, **args)

    def __len__(self):
        return len(self.pdbs)

    def __iter__(self):
        self.start()
        if self.shuffle and self.train:
            random.shuffle(self.pdbs)
        for i in range(math.ceil(len(self.pdbs) / self.batch_size)):
            pdb_batch = self.pdbs[i * self.batch_size : (i + 1) * self.batch_size]
            yield self.get_batch(pdb_batch, self.train)
        self.close()
    
    def __str__(self):
        s = self.__class__.__name__ + "\n"
        for arg_name, arg_value in self.__dict__.items():
            if type(arg_value) in [float, int, str, list]:
                s += "{:20s} = ".format(arg_name)
                if type(arg_value) == float:
                    s += "{:5.2f}".format(arg_value)
                elif type(arg_value) == int:
                    s += "{:4d}".format(arg_value)
                elif type(arg_value) == str:
                    s += arg_value
                elif type(arg_value) == list:
                    s += "({:d})".format(len(arg_value))
                s += "\n"
        return s