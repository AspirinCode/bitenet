from .include import *
from .model import Model
from .data import DataLoader
from .process import PredictionProcesser, PredictionLogger

current_path        = os.path.dirname(os.path.abspath(__file__))
path_models         = os.path.join(current_path, "models")
model_default       = "v1"
model_path_default  = os.path.join(path_models, model_default)

class BiteNet:
    """
    BiteNet 

    Binding site predictions

    Args:
        model_path: str; default="v1";
            path to model or model name;
        gpus: str; default="";
            available gpus for session initialization;
        cpu_only: bool; default=False;
            if True, session will be ran on cpu only;
        **args: additional params for DataLoader of PredictionProcesser.

    Attributes:
        model: Model; 
            network class with tensorflow graph;
        dataloader: DataLoader;
            class for loading grid batches;
        prediction_processer: PredictionProcesser;
            class for filtering predictions;
        minibatch_size: int;
            number of grids in batch for single forward pass.
    """
    def __init__(self, 
        model_path  = model_path_default,
        gpus        = "",
        cpu_only    = False,
        **args):

        self.model = Model()
        params = default_params

        if len(model_path) != 0:
            self.load_model(path=model_path, gpus=gpus, cpu_only=cpu_only)
            params.update(self.load_params_(path=model_path))
            self.model_path = model_path
        
        params = update_params_predict(params)
        params.update(args)
        self.dataloader             = DataLoader(**params)
        self.prediction_processer   = PredictionProcesser(**params)
        self.minibatch_size         = params["minibatch_size"]

    def load_model(self, path=model_path_default, gpus="", cpu_only=False):
        """
        Initializing session and loading weights.

        Args:
            path: str; default="v1";
                path to model folder or model name;
            gpus: str; default="";
                available gpus;
            cpu_only: bool; default=False;
                if True, session will be ran on cpu only.
        """
        if not os.path.isdir(path):
            if os.path.isdir(os.path.join(path_models, path)):
                path = os.path.isdir(os.path.join(path_models, path))
            else:
                print("No such folder or model name:", path)
                
        self.model.init_session(gpus=gpus, cpu_only=cpu_only)
        self.model.load(os.path.join(path, "weights", "best"))

    def load_params_(self, path=""):
        """
        Loading model params.
        """
        return load_params(os.path.join(path, "params.json"))

    def get_grids_predictions(self, filename):
        """
        Get grids and predictions for pdb file.

        Args:
            filename: str; path to pdb file.
        
        Returns: (grids, floors, interfaces, predictions, residues):
            grids: np.array of shape (n_grids, cube_size, cube_size, cube_size, n_channels);
                voxel cubic grids for network forward pass;
            floors: np.array of shape (n_grids, 3);
                grids origin coordinates;
            interfaces: np.array of shape (n_interfaces, 6);
                true interface boxes;
            predictions: np.array of shape (n_predictions, 4);
                predictions;
            residues: list of lists of str;
                neighbor residues list for each prediction.
        """
        grids, _, floors, interfaces, _ = self.dataloader.get_batch(filename)
        if len(grids) > 0:
            predictions = self.model(grids, minibatch_size=self.minibatch_size)
            predictions = self.prediction_processer(predictions, floors)
            residues    = self.prediction_processer.get_residues(
                [filename], predictions)
        else:
            predictions, residues = [[]], [[]]
        return grids, floors[0], interfaces[0], predictions[0], residues[0]

    def predict(self, path, out="", show_progress=True, separate=False):
        """
        Predict and logs predictions.

        Args:
            path: str or list of str;
                path to pdb file; path to dataset file; path to folder with pdb files;
                or list of paths to pdb files;
            out: str; default="";
                path to output log file or folder if separate;
            show_progress: bool; default=True;
                if True, progress bar is shown;
            separate: bool; default=True;
                if True, separate output file for each protein will be written.
        Returns: (predictions, residues) if there is only one protein,
        else (names, predictions, residues):
            names: list of str;
                protein names;
            predictions: list of np.array of shape (None, 4);
                predictions for each protein;
            residues: list of lists of str;
                interface residues for each prediction for each protein.
        """
        single_file = False
        # set pdb list to dataloader
        if os.path.isfile(path):
            if path[-4:] == ".pdb":
                self.dataloader.set_params(pdbs=[path])
                single_file = True
            else:
                dataset = read_dataset(path)
                self.dataloader.set_params(
                    path=os.path.dirname(path), pdbs=dataset)
        elif os.path.isdir(path):
            self.dataloader.set_params(path=path)
        elif type(path) == list:
            self.dataloader.set_params(pdbs=path)
        else:
            return []

        if len(out) > 0:
            logger = PredictionLogger(out, separate=separate, single_file=single_file)
        else:
            logger = None
        progress = None
        if not single_file and show_progress:
            progress = tqdm(total=len(self.dataloader), leave=True)
        names_list, predictions_list, residues_list = [], [], []
        # for batches
        for grids, _, floors, _, names in self.dataloader:
            # forward pass
            predictions = self.model(grids, minibatch_size=self.minibatch_size)
            # predictions filtering and rescaling
            predictions = self.prediction_processer(predictions, floors)
            # predictions interface residues
            residues    = self.prediction_processer.get_residues(
                self.dataloader.get_filename_list(names), predictions)
            # write predictions to file
            if logger:
                logger.write(names, predictions, residues)
            names_list          += names
            predictions_list    += predictions
            residues_list       += residues
            if progress:
                progress.update(len(floors))
        if progress:
            progress.close()

        if single_file:
            return predictions_list[0], residues_list[0]
        else:
            return names_list, predictions_list, residues_list
        
    def __call__(self, path, **args):
        return self.predict(path, **args)