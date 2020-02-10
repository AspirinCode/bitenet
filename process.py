from .include import *
from sklearn.metrics.pairwise import euclidean_distances


def sort_residues(res):
    def res_gr(r):
        c, i, a = r.split("_")
        if len(c) == 0:
            c = " "
        v = ord(c) * 10000 + int(i)
        return v
    return sorted(list(res), key=res_gr)

def get_neighbor_residues(filename, positions, distance_threshold=4):
    """
    Getting residues from pdb file that are not further 
    than distance threshold from given coordinates.

    Args:
        filename: str; 
            path to protein pdb file;
        positions: np.array of shape (n_positions, 3);
            predictions coordinates;
        distance_threshold: float; default=4;
            distance threshold; if at least one atom of residue is not further than 
            distance threshold to position, residue is considered to be on interface.
    
    Returns:
        residues: list of lists of str;
            list of interface residues for each given position;
            residue string is composed as "{chain_id}_{residue_id}_{residue_name}".
    """
    residues = [set() for _ in positions]
    with open(filename, "r") as file:
        for line in file:
            if line[:6] == "ATOM  ":
                x = float(line[30:38])  
                y = float(line[38:46])
                z = float(line[46:54])
                chain = line[21].strip()
                res_name = line[17:20].strip()
                res_id = int(line[22:26])
                element = line[76:78].strip()
                res_name_full = "{:s}_{:d}_{:s}".format(chain, res_id, res_name)
                if element not in ["H", "D"]:
                    distances = np.linalg.norm(positions - [x, y, z], axis=1)
                    indexes = np.where(distances <= distance_threshold)[0]
                    for i in indexes:
                        residues[i].add(res_name_full)
    residues = [sort_residues(r) for r in residues]
    return residues

def get_prediction_residues(filenames, predictions, distance_threshold=4):
    """
    Getting interface residues for list of files to corresponding list of predictions.

    Args:
        filenames: list of str;
            list of protein pdb filenames;
        predictions: list of np.array of shape (n_predictions, 4);
            list of predictions for each protein;
        distance_threshold: float; default=4;
            distance threshold for residue to be considered to be in interface.

    Returns:
        residues: list of lists of str;
            list of residues for each protein.
    """
    residues = []
    for i, filename in enumerate(filenames):
        if len(predictions[i]) > 0:
            residues.append(get_neighbor_residues(filename, predictions[i][:, 1:4], 
                distance_threshold=distance_threshold))
        else:
            residues.append([])
    return residues

class PredictionProcesser:
    """
    Prediction Processer

    Class for rescaling prediction coordinates and its filtering.

    Args:
        score_threshold: float; default=0.1;
            predictions with confidence score less than score_threshold are filtered out;
        max_boxes: int; default=-1;
            maximum number of predictions for each protein;
        distance_threshold: float; default=8;
            distance threshold (Angstroms) for non max suppression;
        voxel_size: float; default=1.;
            voxel size (Angstroms), for rescaling predictions coordinates;
        cell_size: int; default=8;
            cell size (voxels), for rescaling predictions coordinates;
        residues: bool; default=True;
            if True, neighbor residues for each predictions are calculated as well;
        distance_residues: float; default=6.;
            distance threshold for residues to be considered to be neighbor.
    """
    def __init__(self,
        score_threshold     = 0.1, 
        max_boxes           = -1, 
        distance_threshold  = 8.,
        voxel_size          = default_voxel_size,
        cell_size           = default_cell_size,
        residues            = True,
        distance_residues   = 6.,
        **kwargs):

        self.score_threshold    = score_threshold
        self.max_boxes          = max_boxes
        self.distance_threshold = distance_threshold
        self.voxel_size         = voxel_size
        self.cell_size          = cell_size
        self.residues           = residues
        self.distance_residues  = distance_residues

    def filter_output(self, model_output, floors):
        """
        Filtering out low score predictions and rescaling predictions coordinates.

        Args:
            model_output: np.array of shape (n_grids, N, N, N, 4), where N is the number of cells;
                outputs of model forward pass;
            floors: list of lists of np.array of shape (3) or (6); (n_proteins, None, 3);
                list of grid origin coordinates for each protein;
                0:3 are grid origin coordinates; 3:6 are rotation angles;
        
        Returns:
            predictions: list of np.array(None, 4); (n_proteins, None, 4);
                list of predictions for each protein.
        """
        # flatten
        shape = model_output.shape
        model_output = np.reshape(model_output, (shape[0], shape[1]*shape[2]*shape[3], shape[4]))
        model_output[:, :, 1:4] *= self.cell_size * self.voxel_size
        predictions = []
        i = 0

        # for each protein
        for f in floors:
            f = np.array(f)
            predictions_single = []
            for j in range(len(f)):
                # filtering out low score predictions
                indexes = np.where(model_output[i][:, 0] >= self.score_threshold)
                values = model_output[i][indexes]
                values[:, 1:4] += f[j, :3]
                # if rotation angles are provided, prediction coordinates are rotated in the opposite direction
                if f.shape[1] > 3:
                    r = f[j, 3:6]
                    for k in range(len(values)):
                        values[k, 1:4] = np.dot(rotation_matrix(r[0], r[1], -r[2]), values[k, 1:4])
                predictions_single += values.tolist()
                i += 1

            predictions.append(np.array(predictions_single))

        return predictions
            
    def non_max_suppression(self, predictions):
        """
        Non max suppression of predictions. 
        Args:
            predictions: np.array of shape (n_predictions, 4).
        Returns:
            predictions: np.array of shape (n_predictions_out, 4);
                filtered with non max suppression predictions.
        """
        if self.distance_threshold <= 0:
            return predictions
        predictions_new = []
        i = 0
        while len(predictions) >= 1 and (self.max_boxes <= 0 or i < self.max_boxes):
            # find top score prediction
            max_index = predictions[:, 0].argmax(axis=0)
            pos_max   = predictions[max_index, 1:4]
            predictions_new.append(predictions[max_index])
            if i == self.max_boxes - 1:
                break
            # filter out all predictions that are closer than distance_threshold to top score prediction
            distances = np.linalg.norm(predictions[:, 1:4] - pos_max, axis=1)
            indexes = np.where(distances >= self.distance_threshold)[0]
            predictions = predictions[indexes]
            i += 1
        return np.array(predictions_new)

    def get_predictions(self, model_output, floors):
        """
        Rescaling and filtering predictions from model forward pass output.
        Args:
            model_output: np.array of shape (n_grids, N, N, N, 4), where N is the number of cells;
                outputs of model forward pass;
            floors: list of lists of np.array of shape (3) or (6); (n_proteins, None, 3);
                list of grid origin coordinates for each protein;
                0:3 are grid origin coordinates; 3:6 are rotation angles.
        Returns: 
            predictions: list of np.array of shape (None, 4);
                rescaled and filtered predictions for each protein.
        """
        predictions_filtered = self.filter_output(model_output, floors)
        predictions_nms = []
        for pred in predictions_filtered:
            predictions_nms.append(self.non_max_suppression(pred))
        return predictions_nms

    def __call__(self, model_output, floors):
        return self.get_predictions(model_output, floors)

    def set_params(self, **args):
        for name, arg in args.items():
            if name in self.__dict__.keys():
                self.__dict__[name] = arg

    def get_residues(self, filenames, predictions):
        """
        Returns residues lists for each protein and corresponding predictions.
        """
        if not self.residues:
            return [[]] * len(filenames)
        else:
            return get_prediction_residues(filenames, predictions, 
            distance_threshold=self.distance_residues)

class PredictionAccuracy:
    """
    Prediction Accuracy

    Class for calculation of model accuracy.

    Args:
        score_threshold: float; default=0.01;
            predictions with confidence score lower than score thresold are not considered;
        distance_threshold: float; default=4.;
            only predictions which are closer than distance threshold to true 
            interface center can be true positive;
        num_boxes: int; default=-1;
            number of predictions for each protein to be considered;
            -1 for top-N; -2 for top-(N+2); -3 for All, 
            where N is the number of true interfaces for current protein;
        single_box: bool; default=True;
            if True only the closest to interface center prediction can be considered as true positive.
    """
    def __init__(self,
        score_threshold     = 0.01,
        distance_threshold  = 4.,
        num_boxes           = -1,
        single_box          = True,
        ):
        self.score_threshold    = score_threshold
        self.distance_threshold = distance_threshold
        self.num_boxes          = num_boxes
        self.single_box         = single_box

    def get_prediction_scores(self, targets, predictions):
        """
        Calculates true positive score for predictions.

        Args:
            targets: np.array of shape (n_interfaces, 6);
                true interface boxes;
            predictions: np.array of shape (n_predictions, 4);
                predictions;

        Returns:
            scores: list of [confidence, score, score_2, score_m];
                where confidence is corresponding prediction confidence scores;
                score is 0/1 for true positive;
                score_2 is 0/1 for true positive, only one TP for each target;
                score_m is least distance to any target.
        """
        # getting top num_boxes predictions
        if self.num_boxes > 0:
            predictions = predictions[:self.num_boxes]
        elif self.num_boxes == -1:
            predictions = predictions[:len(targets)]
        elif self.num_boxes == -2:
            predictions = predictions[:len(targets) + 2]
        if len(predictions) < 1:
            return []
        # filtering out low score predictions
        predictions = predictions[np.where(predictions[:, 0] >= self.score_threshold)]
        if len(predictions) < 1:
            return []
        # getting predictions distances to targets
        if len(targets):
            scores = euclidean_distances(predictions[:, 1:4], targets[:, :3])
        else:
            scores = np.zeros((len(predictions), len(targets)))
        scores_copy = np.copy(scores)

        pred_gt_scores = []
        # if single_box is True, set predictions distance to 1e4,
        # so only single prediction can be true positive for each target.
        if self.single_box:
            
            for j in range(len(targets)):
                max_index = -1
                for i in range(len(predictions)):
                    if ((scores[i, j] <= self.distance_threshold) \
                        and (predictions[i, 0] > predictions[max_index, 0])):
                        max_index = i
                
                if max_index >= 0:
                    for i in range(len(predictions)):
                        if i != max_index:
                            scores[i, j] = 1e4
            
        if len(targets) > 0:
            true_pred = np.zeros((len(targets)), np.int)
        # getting scores for each prediction
        for i in range(len(predictions)):
            # true positive without single_box
            if len(targets) == 0 or np.max(scores[i, :]) < 0:
                score = 0
            else:
                score = int(np.min(scores[i, :]) <= self.distance_threshold)
            # minimum distance
            if len(targets) == 0:
                score_m = -1.
            else:
                score_m = np.min(scores_copy[i, :])
            # true positive with single_box
            score2 = 0
            for j in range(len(targets)):
                if (scores[i, j] <= self.distance_threshold) and (true_pred[j] == 0):
                    true_pred[j] = 1
                    score2 += 1
            pred_gt_scores.append([predictions[i, 0], score, score2, score_m])
        return pred_gt_scores

    def precision_recall_ap(self, targets, predictions):
        """
        Calculates precision, recall and average precision for predictions.

        Args:
            targets: list of np.array of shape (None, 6);
                true interface boxes for each protein;
            predictions: list of np.array of shape (None, 4);
                predictions for each protein.

        Returns: (precision, recall, average_precision):
            precision: float; 
            recall: float;
            average_precision: float.
        """
        # predictions true positive scores
        gt_values, gt_all = [], 0
        for i in range(len(targets)):
            gt_all += len(targets[i])
            gt_values += self.get_prediction_scores(targets[i], predictions[i])
        if len(gt_values) < 1:
            return 0., 0., 0.

        # get precision-recall curve
        gt_values.sort(key=lambda x: -x[0])
        TP, FP, TP2 = 0, 0, 0
        precision_rank, recall_rank = [], []
        for c_gt in gt_values:
            TP += int(c_gt[1])
            FP += 1 - int(c_gt[1])
            TP2 += int(c_gt[2])
            precision_rank.append(float(TP) / max((TP + FP), 1e-9))
            recall_rank.append(float(TP2) / max(gt_all, 1e-9)) 
        if len(precision_rank) < 1:
            return 0.
        # area under precision-recall curve
        AP = precision_rank[0] * recall_rank[0]
        for i in range(1, len(precision_rank)):
            AP += (recall_rank[i] - recall_rank[i-1]) * (precision_rank[i] + precision_rank[i-1]) / 2.
        precision = TP / max((TP + FP), 1e-9) 
        recall = TP2 / max(gt_all, 1e-9)
        return precision, recall, AP                   

    def __call__(self, targets, predictions):
        return self.precision_recall_ap(targets, predictions) 

class PredictionLogger(Logger):
    """
    Prediction Logger

    Class for logging predictions to file.

    Args:
        filename: str;
            path to file;
        clear: bool; default=True;
            if True, file is cleared before opening;
        separate: bool; default=False;
            if True, separate prediction file for each protein will be created;
            file name will be self.filename + name_id + ".log", self.filename should be folder name;
        single_file: bool; default=False;
            set to True, if there is only one protein predictions,
            pdb id will not be written in the beginning of logfile in this case.
    """
    def init(self, separate=False, single_file=False):
        self.filename_ = self.filename
        self.separate = separate
        self.single_file = single_file
        if self.single_file:
            self.separate = True
        if self.separate and not self.single_file:
            os.makedirs(self.filename_, exist_ok=True)

    def write(self, names, predictions, residues=[]):
        """
        Writes predictions to log file.
        Args:
            names: list of str;
                list of pdb names;
            predictions: list of np.array of shape (None, 4);
                list of predictions for each pdb;
            residues: list of lists of str; default=[];
                list of interface residues for each prediction for each protein.
        
        For each protein records are written:
        >{name}
        {pred_0_score} {pred_0_x} {pred_0_y} {pred_0_z} {pred_0_res_0};{pred_0_res_1};...
        {pred_1_score} {pred_1_x} {pred_1_y} {pred_1_z} {pred_1_res_0};{pred_1_res_1};...
        ...
        If self.separate is True, first line is not written.
        """
        if type(names) == str:
            names       = [names]
            predictions = [predictions]
            residues    = [residues]
        for i, name in enumerate(names):
            if self.separate:
                s = ""
                if not self.single_file:
                    self.filename = os.path.join(self.filename_, name + ".log")
            else:
                s = ">{:s}\n".format(name)
            for j, p in enumerate(predictions[i]):
                s += "{:5.3f} {:7.3f} {:7.3f} {:7.3f}".format(
                    p[0], p[1], p[2], p[3])
                if len(residues) == len(predictions) and \
                    len(residues[i]) == len(predictions[i]):
                    s += " " + ";".join(residues[i][j])
                s += "\n"
            mode = "a"
            if self.separate:
                mode = "w"
            super().write(s, end="", mode=mode)

class TargetLogger(Logger):
    def write(self, names, targets):
        if type(names) == str:
            names = [names]
            targets = [targets]
        for i, name in enumerate(names):
            s = ">{:s}\n".format(name)
            for t in targets[i]:
                s += "{:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f}\n".format(
                    t[0], t[1], t[2], t[3], t[4], t[5])
            super().write(s, end="")

def read_interfaces(filename, sort=True):
    """
    Reads interface boxes from file.

    Args:
        filename: str;
            path to input file;
        sort: bool; default=True;
            if True, names and interfaces are sorted by name.

    Returns: (names, interfaces):
        names: list of str;
            pdb names;
        interfaces: list of np.array of shape (None, 6) (or (None, 4) if predictions);
            interface boxes for each name.

    Records should be:
    >{name}
    {value_0_0} {value_0_1} {value_0_2} ...
    {value_1_0} {value_1_1} {value_1_2} ...
    """
    names, predictions = [], []
    with open(filename, "r") as file:
        name, prediction = "", []
        for line in file:
            if line[0] == ">":
                if len(name) > 0:
                    names.append(name)
                    predictions.append(np.array(prediction))
                name = line.rstrip()[1:]
                prediction = []
            else:
                prediction.append(np.array([float(v) for v in line.rstrip().split()]))
        if len(name) > 0:
            names.append(name)
            predictions.append(np.array(prediction))
    if sort:
        # try to sort like if each name is integer
        try:
            indexes = sorted(range(len(names)), key=lambda i: int(names[i].replace(".pdb", "")))
        except:
            indexes = sorted(range(len(names)), key=lambda i: names[i].replace(".pdb", ""))
        names = [names[i] for i in indexes]
        predictions = [predictions[i] for i in indexes]
    return names, predictions

def get_scores_positions(predictions, score_threshold=0., max_boxes=-1):
    """
    Gets score and positions arrays of predictions.

    Args:
        predictions: list of np.array of shape (None, 4);
            list of predictions;
        score_threshold: float; default=0.;
            only predictions with score >= score_threshold are considered;
        max_boxes: int; default=-1;
            maximum number of predictions for each protein.
    
    Returns: (scores, positions):
        scores: list of np.array of shape (None);
            predictions scores;
        positions: list of np.array of shape (None, 3);
            predictions coordinates.
    """
    scores, positions = [], []
    for pred in predictions:
        scores_step, positions_step = [], []
        for i, p in enumerate(pred):
            if max_boxes > 0 and i >= max_boxes:
                break
            if p[0] >= score_threshold:
                scores_step.append(p[0])
                positions_step.append(p[1:4])
        scores.append(scores_step)
        positions.append(positions_step)
    return scores, positions

def read_predictions(filename, sort=True, get_residues=False):
    """
    Reads predictions from file.

    Args:
        filename: str; 
            path to file;
        sort: bool; default=True;
            if True, names and predictions are sorted by name;
        get_residues: bool; default=False;
            if True, neighbor residues for predictions are read and returned as well.

    Returns: (names, predictions, residues) or (names, predictions):
        names: list of str;
            list of pdb names;
        predictions: list of list of np.array of shape (None, 4);
            list of predictions for each pdb;
        residues: list of lists of str;
            list of residues for each pdb for each prediction.

    Records should be:
    >{name}
    {pred_0_score} {pred_0_x} {pred_0_y} {pred_0_z} {pred_0_res_0};{pred_0_res_1};...
    {pred_1_score} {pred_1_x} {pred_1_y} {pred_1_z} {pred_1_res_0};{pred_1_res_1};...
    ...
    """
    names, predictions, residues = [], [], []
    with open(filename, "r") as file:
        name, prediction, residue = "", [], []
        for line in file:
            if line[0] == ">":
                if len(name) > 0:
                    names.append(name)
                    predictions.append(np.array(prediction))
                    residues.append(residue)
                name = line.rstrip()[1:]
                prediction = []
                residue = []
            else:
                l = line.rstrip().split()
                prediction.append(np.array([float(v) for v in l[:4]]))
                if len(l) > 4:
                    residue.append(line.rstrip().split()[4].split(";"))
                else:
                    residue.append([])
        if len(name) > 0:
            names.append(name)
            predictions.append(np.array(prediction))
            residues.append(residue)
    if sort:
        # try to sort like if names are integers
        try:
            indexes = sorted(range(len(names)), key=lambda i: int(names[i].replace(".pdb", "")))
        except:
            indexes = sorted(range(len(names)), key=lambda i: names[i].replace(".pdb", ""))
        names       = [names[i] for i in indexes]
        predictions = [predictions[i] for i in indexes]
        residues    = [residues[i] for i in indexes]
    if get_residues:
        return names, predictions, residues
    else:
        return names, predictions