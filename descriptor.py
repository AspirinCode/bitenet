from ctypes import cdll, c_void_p, c_char_p, c_int, c_uint, c_float, c_double, c_bool, c_size_t
import numpy as np
import os

ATOM_TYPIZATION = "ITSCORE"
PDB_BUFFER = 0.
MIN_ATOMS_IN_LIGAND = 14
MIN_ATOMS_IN_INTERFACE = 20
INTERFACE_CUTOFF = 4.
# MIN_ATOMS_IN_PROTEIN = 400
MIN_ATOMS_IN_PROTEIN = 0
DENSITY_CUTOFF = 2.
VOXEL_SIZE = 1.
GRID_SPACING = 0.
FILTER_LIGANDS = True

PDB_BUFFER += DENSITY_CUTOFF

current_path = os.path.dirname(os.path.abspath(__file__))
library_path = os.path.join(current_path, "libs", "libdescriptor.so")
lib = cdll.LoadLibrary(library_path)

class PDBfile:
	"""
	PDB file

	Class for protein atoms read from pdb file

	Args:
		filename: str; path to pdb file;
		atom_typization: str; default="ITSCORE"; 
			atom typization for further parametrization of grid descriptors,
			there is only "ITSCORE" accepted now;
		min_atoms_in_ligand: int; default=14;
			ligands with number of atoms less than this value are considered nonrelevant;
		min_atoms_in_interface: int; default=20;
			ligands with number of atoms in interface less than this value are considered nonrelevant;
		interface_cutoff: float; default=4;
			distance threshold for protein atoms to be in interface;
		min_atoms_in_protein: int; default=0;
			if number of atoms in protein is less than this value atoms are not stored;
		filter_ligands: bool; default=True;
			if True nonrelevant ligands are filtered and interface boxes are not computed for them;
		pdb_buffer: float; default=2; 
			value to add to protein box sizes;
	"""

	def __init__(self, filename,
		atom_typization 		= ATOM_TYPIZATION,
		min_atoms_in_ligand		= MIN_ATOMS_IN_LIGAND,
		min_atoms_in_interface	= MIN_ATOMS_IN_INTERFACE,
		interface_cutoff		= INTERFACE_CUTOFF,
		min_atoms_in_protein	= MIN_ATOMS_IN_PROTEIN,
		filter_ligands 			= FILTER_LIGANDS,
		pdb_buffer 				= PDB_BUFFER,
		**args):

		self.filename = filename.encode("utf-8")

		# cpp PDBfile class init
		# PDBfile* pdb_new(const char* file_name, 
        #         const char* typization="ITSCORE",
        #         const unsigned int atoms_in_ligand_cutoff=14,
        #         const unsigned int atoms_in_interface_cutoff = 20,
        #         const float interface_radius = 4.0,
        #         const unsigned int atoms_in_protein_cutoff = 400,
        #         const bool filter_ligands = true)
		lib.pdb_new.argtypes = [c_char_p, 
								c_char_p, c_uint, c_uint,
								c_float, c_uint, c_bool]
		lib.pdb_new.restype = c_void_p

		# void rotate(PDBfile* pdb, const double theta, const double phi, const double psi)
		lib.rotate.argtypes = [c_void_p, c_double, c_double, c_double]
		lib.rotate.restype = c_void_p

    	# void center(PDBfile* pdb);
		lib.center.argtypes = [c_void_p]
		lib.center.restype = c_void_p

    	# void compute_boxes(PDBfile* pdb, const float buffer=0.);
		lib.compute_boxes.argtypes = [c_void_p, c_float]
		lib.compute_boxes.restype = c_void_p

		# void write(PDBfile* pdb, const char* filename, 
        # 	const bool flag_write_protein=true, const bool flag_write_ligands=true);
		lib.write.argtypes = [c_void_p, c_char_p, c_bool, c_bool]
		lib.write.restype = c_void_p

		# int get_interface_number(PDBfile* pdb);
		lib.get_interface_number.argtypes = [c_void_p]
		lib.get_interface_number.restype = c_void_p

		# float* export_interfaces(PDBfile* pdb);
		lib.export_interfaces.argtypes = [c_void_p]

		# void pdb_del(PDBfile *pdb);
		lib.pdb_del.argtypes = [c_void_p]
		lib.pdb_del.restype = c_void_p

		self.atom_typization 		= atom_typization.encode("utf-8")
		self.min_atoms_in_ligand 	= min_atoms_in_ligand
		self.min_atoms_in_interface = min_atoms_in_interface
		self.interface_cutoff 		= interface_cutoff
		self.min_atoms_in_protein 	= min_atoms_in_protein
		self.filter_ligands			= filter_ligands
		self.pdb_buffer				= pdb_buffer

		self.obj = lib.pdb_new(self.filename, 
			self.atom_typization, self.min_atoms_in_ligand, self.min_atoms_in_interface, 
			self.interface_cutoff, self.min_atoms_in_protein, self.filter_ligands)

	def __del__(self):
		lib.pdb_del(self.obj)

	def rotate(self, theta, phi, psi):
		"""
		Rotates protein atoms around direction 
		(cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)) for angle psi.
		
		Args:
			theta, phi, psi: float; angles
		"""
		lib.rotate(self.obj, theta, phi, psi)
	
	def center(self):
		"""
		Move protein center to (0, 0, 0) coordinates and rotates protein for orientation along principal axes.
		"""
		lib.center(self.obj)

	def compute_boxes(self):
		"""
		Sets protein box and interface boxes values for current atom coordinates.
		"""
		lib.compute_boxes(self.obj, self.pdb_buffer)

	def write(self, filename, flag_write_protein=True, flag_write_ligands=True):
		"""
		Writes pdb file with.
		Args:
			filename: str; path to new file;
			flag_write_protein: bool; default=True; if True protein atoms will be written;
			flag_write_ligands: bool; default=True; if True ligands will be written;
		"""
		filename = filename.encode("utf-8")
		lib.write(self.obj, filename, flag_write_protein, flag_write_ligands)

	def get_interface_number(self):
		"""
		Returns number of ligand binding interfaces in protein.
		"""
		interface_number = lib.get_interface_number(self.obj)
		if interface_number:
			return interface_number
		else:
			return 0

	def export_interfaces(self):
		"""
		Returns interface boxes (np.array of shape (n_interfaces, 6)).
		"""
		interface_number = self.get_interface_number()
		if interface_number:
			lib.export_interfaces.restype = np.ctypeslib.ndpointer(dtype=np.float32, 
					shape=(interface_number,6))
			return lib.export_interfaces(self.obj).copy()
		else:
			return []
		
class Descriptor:
	"""
	Descriptor

	Class for protein atom density voxel grid calculation.

	Args:
		pdb: PDBfile; 
			protein to voxelize;
		voxel_size: int; default=1.; 
			size of voxel (Angstroms);
		density_cutoff: float; default=2.; 
			distance threshold for atom density calculation (Angstroms);
		grid_spacing: float; default=0.;
			spacing between voxels (Angstroms);
	"""
	def __init__(self, pdb,
		voxel_size 		= VOXEL_SIZE,
		density_cutoff 	= DENSITY_CUTOFF,
		grid_spacing 	= GRID_SPACING, 
		**args):

		self.voxel_size 	= voxel_size
		self.density_cutoff = density_cutoff
		self.grid_spacing 	= grid_spacing
		
		# Descriptor* descriptor_new(PDBfile* pdb);
		lib.descriptor_new.argtypes = [c_void_p]
		lib.descriptor_new.restype = c_void_p

		# void init_grid(Descriptor* descriptor,
		# 	const float voxel_size = 1.,
		# 	const float spacing = 0.0,
		# 	const float r_cutoff=4.,
		# 	const int threads=1);
		lib.init_grid.argtypes = [c_void_p, c_float, c_float, c_float, c_int]
		lib.init_grid.restype = c_void_p

		# int* get_grid_size(Descriptor*);
		lib.get_grid_size.argtypes = [c_void_p]
		lib.get_grid_size.restype = np.ctypeslib.ndpointer(dtype=c_int, shape=(4,))

		# float* get_grid_floor(Descriptor*);
		lib.get_grid_floor.argtypes = [c_void_p]
		lib.get_grid_floor.restype = np.ctypeslib.ndpointer(dtype=c_float, shape=(3,))

		# double* export_grid(Descriptor*, const int threads=1);
		lib.export_grid.argtypes = [c_void_p, c_int]

		# void descriptor_del(Descriptor*);
		lib.descriptor_del.argtypes = [c_void_p]
		lib.descriptor_del.restype = c_void_p
		
		self.pdb = pdb.obj
		self.obj = lib.descriptor_new(self.pdb)

	def __del__(self):
		lib.descriptor_del(self.obj)

	def init_grid(self, threads=8):
		"""
		Initializes voxel grid values.
		Args:
			threads: int; default=8; number of threads;
		"""
		lib.init_grid(self.obj, self.voxel_size, self.grid_spacing, self.density_cutoff, threads)

	def get_grid_size(self):
		"""
		Returns: np.array of shape (4); 
			[grid_size_x, grid_size_y, grid_size_z, n_channels];
			sizes of grid in voxels and number of channels;
		"""
		return np.copy(lib.get_grid_size(self.obj))

	def get_grid_floor(self):
		"""
		Returns: np.array of shape (3);
			[grid_floor_x, grid_floor_y, grid_floor_z];
			grid origin coordinates;
		"""
		return np.copy(lib.get_grid_floor(self.obj))

	def export_grid(self, threads=8):
		"""
		Calculates grid density and returns voxel grid.
		Args:
			threads: int; default=8; number of threads;
		Returns:
			grid: np.array of shape (grid_size_x, grid_size_y, grid_size_z, n_channels);
				voxel grid;
		"""
		size = self.get_grid_size()
		lib.export_grid.restype = np.ctypeslib.ndpointer(dtype=np.float64, 
			shape=(size[0], size[1], size[2], size[3]))
		return lib.export_grid(self.obj, threads).astype(np.float32)


def pdb_grid_interfaces(filename, rotation_angles=[0., 0., 0.], threads=8, **args):
	"""
	Reads pdb file and calculates interface boxes and voxel grid.

	Args:
		filename: str;
			path to pdb file;
		rotation_angles: list of floats; [theta, phi, psi]; default=[0, 0, 0];
			protein will be rotated for angle psi around (cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta));
		threads: int; default=8;
			number of threads for grid initialization and density calculation;
		**args: additional arguments for PDBfile or Descriptor classes;
	Returns: (grid, grid_floor, interfaces):
		grid: np.array of shape (grid_size_x, grid_size_y, grid_size_z, n_channels);
			calculated voxel grid;
		grid_floor: np.array of shape (3); [grid_floor_x, grid_floor_y, grid_floor_z];
			coordinates of grid origin;
		interfaces: np.array of shape (n_interfaces, 6);
			interface boxes;
	"""
	# reading pdb file
	pdb = PDBfile(filename, **args)
	# rotation
	theta, phi, psi = rotation_angles
	if theta != 0 or phi != 0 or psi != 0:
		pdb.rotate(theta, phi, psi)
	# calculating interface boxes
	pdb.compute_boxes()
	interfaces = pdb.export_interfaces()

	# grid initialization
	descriptor = Descriptor(pdb, **args)
	descriptor.init_grid(threads=threads)
	size = descriptor.get_grid_size()
	if size[0] * size[1] * size[2] * size[3] * 8 >= 2**31:
		# to avoid: ValueError: invalid shape in fixed-type tuple: dtype size in bytes must fit into a C int.
		return np.zeros((32, 32, 32, 11), np.float32), np.zeros((3)), []
	# grid calculation
	grid = descriptor.export_grid(threads=threads)
	grid_floor = descriptor.get_grid_floor()

	return grid, grid_floor, interfaces