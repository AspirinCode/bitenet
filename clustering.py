from .include import *
from sklearn.cluster import MeanShift, AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import matplotlib.pyplot as plt

class Clustering:
    """
    Clustering

    Class for clusters scoring.

    Args:
        score_threshold: float; default=0.1;
            input predictions with score lower than score threshold are filtered out;
        cluster_score_threshold: float; default=0.1;
            clusters with step score lower than cluster_score_threshold are not used in the final output;
        clister_score_threshold_step: float; default=0.1;
            only frames where score is greater or equal to cluster_score_threshold_step 
            will be used for calculation of cluster score.
        cluster_score_threshold_mean: float; default=0.001;
            clusters with mean score lower than cluster_score_threshold_mean 
            are not used in the final output;
        filter_residues: bool; default=False;
            if True, predictions with duplicate residues list are not added;
        rank_by: str; default="mean";
            if "mean" clusters are sorted by mean cluster score;
            if "step" clusters are sorted by step cluster score;
            if "max" clusters are sorted by max cluster score;
            if None clusters are not sorted.
    
    Attributes:
        predictions: list of np.array of shape (None, 4);
            input predictions by frames;
        predictions_: np.array of shape (n_predictions_, 4);
            filtered and flattened input predictions;
        residues: list of lists of lists of str;
            input predictions residues by frames;
        residues_: list of lists of str;
            filtered input prediction residues;
        indexes: list of np.array of shape (None);
            indexes of each of prediction from input predictions mapping it
            to filtered predictions_ indexes; if index is -1 that prediction is not added;
        n_clusters_: int;
            number of clusters;
        n_clusters: int;
            number of clusters after filtering by cluster score;
        labels_: np.array of shape (n_predictions_);
            clustered label for each of predictions_ or residues_;
        cluster_indexes_: np.array of shape (n_clusters_);
            cluster indexes for all clusters sorted by score;
        cluster_indexes: np.array of shape (n_clusters);
            cluster indexes for filtered clusters sorted by score;
        cluster_scores: np.array of shape (n_clusters_);
            step cluster scores;
        cluster_scores_mean: np.array of shape (n_clusters_);
            mean cluster scores;
        cluster_scores_steps: np.array of shape (n_frames, n_clusters_);
            cluster scores for each frame;
        cluster_scores_max: np.array of shape (n_clusters_);
            maximum cluster scores;
        cluster_centers_: np.array of shape (n_clusters_, 3);
            cluster centers coordinates.
    """
    def __init__(self,
        score_threshold              = 0.1,
        cluster_score_threshold      = 0.1,
        cluster_score_threshold_step = 0.1,
        cluster_score_threshold_mean = 0.001,
        filter_residues              = False,
        rank_by="mean"):

        self.score_threshold              = score_threshold
        self.cluster_score_threshold      = cluster_score_threshold
        self.cluster_score_threshold_step = cluster_score_threshold_step
        self.cluster_score_threshold_mean = cluster_score_threshold_mean
        self.filter_residues              = filter_residues
        self.rank_by                      = rank_by

        self.predictions         = []
        self.predictions_        = []
        self.residues            = []
        self.residues_           = []
        self.indexes             = []
        self.n_clusters_         = 0
        self.n_clusters          = 0
        self.labels_             = []
        self.cluster_indexes_    = []
        self.cluster_indexes     = []
        self.cluster_scores       = []
        self.cluster_scores_mean  = []
        self.cluster_scores_steps = []
        self.cluster_scores_max   = []
        self.cluster_centers_     = []

    def preprocess(self, predictions, residues=[]):
        """
        Preprocessing input predictions and residues.
        Flattening predictions; filtering out by residues set if filter_residues is True;
        assigning index for each prediction from predictions to index in predictions_.

        Args:
            predictions: list of np.array of shape (None, 4);
                predictions array for each frame;
            residues: list of lists of lists of str; default=[];
                predictions interface residues for each frame for each prediction.

        """
        self.indexes      = []
        self.predictions  = predictions
        self.predictions_ = []
        self.residues     = residues
        self.residues_    = []
        # for each frame
        for step, pred in enumerate(predictions):
            indexes = []
            # for each prediction in frame
            for i, p in enumerate(pred):
                # prediction score threshold
                if p[0] >= self.score_threshold:
                    if not self.filter_residues:
                        # add prediction to predictions_
                        self.predictions_.append(p.copy())
                        # if residues is not empty add residue interface to residues_
                        if len(residues) == len(predictions) and \
                            len(residues[step]) == len(predictions[step]):
                            self.residues_.append(residues[step][i])
                        # add prediction index
                        indexes.append(len(self.predictions_)-1)
                    else:
                        res = residues[step][i]
                        # if prediction with such residue interface is already in list
                        if res in self.residues_:
                            # get index of prediction from list
                            index_res = self.residues_.index(res)
                            # sum prediction score and average coordinates
                            self.predictions_[index_res][1:4] = \
                                (self.predictions_[index_res][1:4] * self.predictions_[index_res][0] + \
                                p[1:4] * p[0]) / (self.predictions_[index_res][0] + p[0])
                            self.predictions_[index_res][0] += p[0]
                            # add index to prediction
                            indexes.append(index_res)
                        else:
                            # add prediction, residues and index
                            index_res = len(self.residues_)
                            self.residues_.append(res)
                            self.predictions_.append(p.copy())
                            indexes.append(index_res)
                else:
                    indexes.append(-1)
            self.indexes.append(np.array(indexes))
        self.predictions_ = np.array(self.predictions_)

    def get_cluster_scores_steps(self):
        """
        Calculating steps cluster scores which are maximum score of predictions 
        from corresponding cluster for each frame.
        """
        cluster_scores_steps = np.zeros((len(self.predictions), self.n_clusters_))
        # for frame
        for step, indexes_s in enumerate(self.indexes):
            # for prediction index in frame
            for i, index in enumerate(indexes_s):
                # if prediction is in labels and clustered
                if index >= 0:
                    label = self.labels_[index]
                    if label >= 0:
                        # max score
                        score = self.predictions[step][i, 0]
                        cluster_scores_steps[step, label] = max(score, cluster_scores_steps[step, label])
        return cluster_scores_steps
    
    def get_cluster_scores_mean(self, cluster_scores_steps=[]):
        """
        Returns mean cluster scores which are mean values of steps cluster scores.
        """
        if len(cluster_scores_steps) == 0:
            return np.mean(self.get_cluster_scores_steps(), axis=0)
        else:
            return np.mean(cluster_scores_steps, axis=0)

    def get_cluster_scores(self, cluster_scores_steps=[]):
        """
        Returns step cluster scores.
        """
        if len(cluster_scores_steps) == 0:
            cluster_scores_steps = self.get_cluster_scores_steps()
        cluster_scores_steps_mask = cluster_scores_steps >= self.cluster_score_threshold_step
        cluster_scores_steps_2 = cluster_scores_steps * cluster_scores_steps_mask
        cluster_scores = np.sum(cluster_scores_steps_2, axis=0) / \
            np.maximum(np.sum(cluster_scores_steps_mask, axis=0), 1e-9)
        return cluster_scores
        
    def rank_clusters(self):
        """
        Calculating cluster scores and sorting clusters.
        """
        if self.n_clusters_ == 0:
            self.n_clusters_ = np.max(self.labels_) + 1
        # Cluster scores
        self.cluster_scores_steps = self.get_cluster_scores_steps()
        self.cluster_scores_mean  = self.get_cluster_scores_mean(self.cluster_scores_steps)
        self.cluster_scores       = self.get_cluster_scores(self.cluster_scores_steps)
        self.cluster_scores_max   = np.max(self.cluster_scores_steps, axis=0)

        # Indexes of all and filtered clusters
        indexes_ = np.where(self.cluster_scores_mean >= 0)[0]
        indexes  = np.where(
            (self.cluster_scores_mean >= self.cluster_score_threshold_mean) * \
            (self.cluster_scores >= self.cluster_score_threshold))[0]
        # Sorting clusters
        if self.rank_by == "mean":
            self.cluster_indexes_ = sorted(indexes_.tolist(), key = lambda i: - self.cluster_scores_mean[i])
            self.cluster_indexes  = sorted(indexes.tolist(),  key = lambda i: - self.cluster_scores_mean[i])
        elif self.rank_by == "step":
            self.cluster_indexes_ = sorted(indexes_.tolist(), key = lambda i: - self.cluster_scores[i])
            self.cluster_indexes  = sorted(indexes.tolist(),  key = lambda i: - self.cluster_scores[i])
        elif self.rank_by == "max":
            self.cluster_indexes_ = sorted(indexes_.tolist(), key = lambda i: - self.cluster_scores_max[i])
            self.cluster_indexes  = sorted(indexes.tolist(),  key = lambda i: - self.cluster_scores_max[i])
        else:
            self.cluster_indexes_ = indexes_
            self.cluster_indexes  = indexes
        self.n_clusters = len(self.cluster_indexes)
        # Cluster centers
        if len(self.cluster_centers_) == 0:
            self.cluster_centers_ = np.zeros((self.n_clusters_, 3))
            for i in range(self.n_clusters_):
                indexes = np.where(self.labels_ == i)[0]
                self.cluster_centers_[i] = np.mean(self.predictions_[indexes, 1:4], axis=0)

    def get_predictions(self, label):
        """
        Getting predictions for specified cluster.
        Args:
            label: int;
                cluster index.
        Returns:
            predictions: np.array of shape (None, 4);
                predictions with label == label.
        """
        predictions = []
        for step, indexes_s in enumerate(self.indexes):
            for i, index in enumerate(indexes_s):
                if index >= 0 and self.labels_[index] == label:
                    predictions.append(self.predictions[step][i])
        return np.array(predictions)

    def get_cluster_centers_steps(self):
        """
        Calculating cluster center for each frame which are average prediction coordinates.
        Returns:
            cluster_centers_steps: np.array of shape (n_frames, n_clusters_, 3);
                center coordinates for each cluster for each frame
        """
        cluster_centers_steps = np.zeros((len(self.predictions), len(self.cluster_centers_), 3))
        for step, indexes_s in enumerate(self.indexes):
            cluster_scores = np.zeros((len(self.cluster_centers_)))
            for i in np.where(indexes_s >= 0)[0]:
                index = indexes_s[i]
                label = self.labels_[index]
                if label >= 0:
                    cluster_scores[label] += self.predictions[step][i, 0]
                    cluster_centers_steps[step, label, :] += self.predictions[step][i, 0] * self.predictions[step][i, 1:4]
            for i in range(len(self.cluster_centers_)):
                if cluster_scores[i] > 0:
                    cluster_centers_steps[step, i, :] /= cluster_scores[i]
                else:
                    if step == 0:
                        cluster_centers_steps[step, i, :] = self.cluster_centers_[i]
                    else:
                        cluster_centers_steps[step, i, :] = cluster_centers_steps[step-1, i, :]
        return cluster_centers_steps

    def __str__(self):
        s = ""
        for n, v in self.__dict__.items():
            if type(v) == int:
                s += "{:30s} : {:<4d}\n".format(n, v)
            elif type(v) == float:
                s += "{:30s} : {:6.3f}\n".format(n, v)
            elif type(v) == bool:
                s += "{:30s} : ".format(n) + str(v) + "\n"
            elif type(v) == str:
                s += "{:30s} : {:s}\n".format(n, v)
            else:
                try:
                    s += "{:30s} : ({:4d})\n".format(n, len(v))
                except:
                    pass
        return s

    def get_summary(self, all=False):
        """
        Cluster scores summary.
        Args:
            all: bool; default=False;
                if True, all clusters are used.
        Returns:
            summary: list of dicts:
                dict with cluster info for each cluster.
        dict keys and values are:
            "index":                int;    index of cluster by score;
            "cluster_index":        int;    index of cluster in self.cluster_indexes;
            "cluster_score":        float;  step cluster score;
            "cluster_score_mean":   float;  mean cluster score;
            "predictions_num":      int;    number of predictions in cluster;
            "steps_num":            int;    number of frames where predictions for this cluster observed;
            "score_max":            float;  max cluster score;
            "score_max_step":       int;    index of frame where cluster has maximum score;
            "cluster_center_x":     float;  x coordinate of cluster center;
            "cluster_center_y":     float;  y coordinate of cluster center;
            "cluster_center_z":     float;  z coordinate of cluster center.
        ]
        """
        if all:
            cluster_indexes = self.cluster_indexes_
        else:
            cluster_indexes = self.cluster_indexes
        summary = []
        for i, index in enumerate(cluster_indexes):
            cluster_score       = self.cluster_scores[index]
            cluster_score_mean  = self.cluster_scores_mean[index]
            predictions_num     = len(self.get_predictions(index))
            steps_num           = len(np.where(self.cluster_scores_steps[:, index] >= self.cluster_score_threshold_step)[0])
            score_max           = np.max(self.cluster_scores_steps[:, index])
            score_max_step      = np.argmax(self.cluster_scores_steps[:, index])
            cluster_center      = self.cluster_centers_[index, :]
            d = {
                "index"             : i,
                "cluster_index"     : index,
                "cluster_score"     : cluster_score,
                "cluster_score_mean": cluster_score_mean,
                "predictions_num"   : predictions_num,
                "steps_num"         : steps_num,
                "score_max"         : score_max,
                "score_max_step"    : score_max_step,
                "cluster_center_x"  : cluster_center[0],
                "cluster_center_y"  : cluster_center[1],
                "cluster_center_z"  : cluster_center[2],
            }
            summary.append(d)
        return summary

    def get_summary_str(self, all=False):
        """
        Returns summary as string.
        """
        summary = self.get_summary(all=all)
        s = ""
        for d in summary:
            s += "{:3d} {:3d} {:.3f} {:.3f} {:5d} {:5d} {:.3f} {:5d} {:7.2f} {:7.2f} {:7.2f}\n".format(
                d["index"], d["cluster_index"], 
                d["cluster_score"], d["cluster_score_mean"],
                d["predictions_num"], d["steps_num"], 
                d["score_max"], d["score_max_step"],
                d["cluster_center_x"], d["cluster_center_y"], d["cluster_center_z"]
            )
        return s

    def export_summary(self, filename, all=False):
        """
        Writes clusters summary csv file.
        """
        summary = self.get_summary(all=all)
        summary = pd.DataFrame(summary, columns=list(summary[0].keys()))
        summary.to_csv(filename, index=False)
        return summary

    def plot(self, filename="", colors=colors):
        """
        Plots cluster scores steps.
        """
        fig, axes = plt.subplots(figsize=(16, 8))
        for i, cluster_index in enumerate(self.cluster_indexes):
            color = colors[min(len(colors)-1, i)]
            axes.plot(np.arange(len(self.cluster_scores_steps)),
                self.cluster_scores_steps[:, cluster_index],
                alpha=0.3, color=color)
        for i, cluster_index in enumerate(self.cluster_indexes):
            color = colors[min(len(colors) - 1, i)]
            axes.plot(np.arange(len(self.cluster_scores_steps)),
                moving_average(self.cluster_scores_steps[:, cluster_index], n=100),
                label=str(i), color=color)
        axes.set_yticks(np.arange(0., 1.01, 0.1))
        plt.ylim(0, 1)
        plt.xlim(0, len(self.cluster_scores_steps))
        plt.xlabel("Frame")
        plt.ylabel("Score")
        plt.grid(True)
        plt.legend()
        if len(filename) > 0:
            plt.savefig(filename)
        return
    
    def get_cluster_residues(self, score_threshold=1., all=True):
        """
        Gets list of residues for each cluster.
        Args:
            score_threshold: float; default=1.;
                score threshold for residue to be in cluster;
            all: bool; default=True;
                if True, residue interfaces from self.residues will be used,
                else residue interfaces from self.residues_.
        Returns:
            result: list of lists of str;
                list of residues for each cluster.
        """
        res_dict = {}
        if all:
            # for frame
            for step, indexes_s in enumerate(self.indexes):
                # for prediction
                for i, index in enumerate(indexes_s):
                    if index >= 0 and self.labels_[index] >= 0:
                        res = self.residues[step][i]
                        # for residue
                        for r in res:
                            # assign residue score
                            if r not in res_dict:
                                res_dict[r] = np.zeros((self.n_clusters_))
                            res_dict[r][self.labels_[index]] += self.predictions[step][i, 0] / len(res)
        else:
            # for label
            for i, label in enumerate(self.labels_):
                if label >= 0:
                    res = self.residues_[i]
                    # for residue
                    for r in res:
                        # assign residue score
                        if r not in res_dict:
                            res_dict[r] = np.zeros((self.n_clusters_))
                        res_dict[r][label] += self.predictions_[i, 0] / len(res)
        
        result = []
        # for cluster
        for i, cluster_index in enumerate(self.cluster_indexes):
            res_list, values_list = [], []
            for r, values in res_dict.items():
                v = values[cluster_index]
                # if residue score for cluster >= score_threshold, add it to list
                if v >= score_threshold:
                    res_list.append(r)
                    values_list.append(v)
            indexes = list(range(len(values_list)))
            indexes = sorted(indexes, key = lambda x : -values_list[x])
            res_list = [res_list[i] for i in indexes]
            result.append(res_list)
        return result

    def write_cluster_residues(self, filename, **args):
        """
        Writes residue interfaces for each cluster to file.
        """
        res = self.get_cluster_residues(**args)
        with open(filename, "w") as file:
            for i, r in enumerate(res):
                file.write("{:2d} {:s}\n".format(i, ";".join(r)))
        return res
        

class Clustering_MeanShift(Clustering, MeanShift):
    """
    Mean Shift clustering.

    Class for predictions clustering using Mean Shift algorithm.
    Inherited from Clustering and sklearn.MeanShift classes.

    Args:
        distance_merge: float; default=5.;
            bandwidth parameter for sklearn.MeanShift; 
            clusters with distances less than distance_merge are merged;
        cluster_all: bool; default=True;
            cluster_all parameter for sklearn.MeanShift;
            if True, labels for all predictions are assigned;
        **args: arguments for Clustering.
    """
    def __init__(self, 
        distance_merge     = 5.,
        cluster_all        = True,
        **args):
        self.distance_merge     = distance_merge
        Clustering.__init__(self, **args)
        MeanShift.__init__(self, 
            bandwidth   = self.distance_merge,
            bin_seeding = True,
            cluster_all = cluster_all,
        )

    def cluster(self, predictions, residues=[]):
        self.preprocess(predictions, residues)
        self.fit(self.predictions_[:, 1:4])
        self.rank_clusters()


def assign_leaf_labels(children, labels, i=0, label=0):
    if i >= len(children) + 1:
        i -= len(children) + 1
    for c in children[i]:
        if c < len(children) + 1:
            labels[c] = label
        else:
            assign_leaf_labels(children, labels, c, label)


class Clustering_Agglomerative(Clustering, AgglomerativeClustering):
    """
    Agglomerative Clustering.

    Class for predictions clustering for coordinates based on agglomerative algorithm.

    Args:
        parameters for sklearn.Agglomerative:
            n_clusters: int; default=10;
                number of clusters;
            distance_threshold: float; default=None;
                distance threshold for clusters merging;
            linkage: str; default="ward";
            affinity: str; default="euclidean";
            compute_full_tree: bool; default=False.
        **args: arguments for Clustering.
    """
    def __init__(self,
        n_clusters          = 10,
        distance_threshold  = None,
        linkage             = "ward",
        affinity            = "euclidean",
        compute_full_tree   = False,
        **args):
        if n_clusters != None:
            distance_threshold = None
        Clustering.__init__(self, **args)
        AgglomerativeClustering.__init__(self,
            n_clusters          = n_clusters,
            distance_threshold  = distance_threshold,
            linkage             = linkage,
            affinity            = affinity,
            compute_full_tree   = compute_full_tree,
        )

    def cluster(self, predictions, residues=[]):
        self.preprocess(predictions, residues)
        self.fit(self.predictions_[:, 1:4])
        self.rank_clusters()

    def refit(self, n_clusters=10):
        """
        Getting another specified number of clusters based on precomputed tree.
        """
        labels = np.zeros_like(self.labels_)
        for i in range(len(self.children_) - 1, len(self.children_) - n_clusters, -1):
            for c in self.children_[i]:
                if c >= len(self.children_) + 1:
                    assign_leaf_labels(self.children_, labels, c, c)
        for i, label in enumerate(np.unique(labels)):
            indexes = np.where(labels == label)
            labels[indexes] = i
        self.labels_ = labels

        self.n_clusters_ = n_clusters
        self.cluster_centers_ = []
        self.rank_clusters()

class Clustering_Agglomerative_Residues(Clustering_Agglomerative):
    """
    Agglomerative Residues Clustering.

    Class for predictions clustering for residues based on agglomerative algorithm.

    Args:
        linkage: str; default="average";
            param for sklearn.Agglomerative;
        distance_norm: str; default="max";
            if "min", residue sets similarity is intersection / (min len two residue sets);
            if "max", residue sets similarity is intersection / (max len two residue sets);
        filter_residues: bool; default=True;
            filter_residues param for Clustering;
        **args: arguments for Clustering.
    """
    def __init__(self, 
            linkage         = "average",
            distance_norm   = "max",
            filter_residues = True,
            **args):
        self.distance_norm = distance_norm
        Clustering_Agglomerative.__init__(self, 
            linkage         = linkage,
            filter_residues = filter_residues,
            affinity        = "precomputed",
            **args)

    def get_distance_matrix(self):
        """
        Calculates distance matrix based on distances between residues sets.
        """
        residues_unique = []
        residues_unique_indexes = []
        # get unique residue indexes
        for i, res_list in enumerate(self.residues_):
            res_i = []
            for res in res_list:
                if res in residues_unique:
                    res_i.append(residues_unique.index(res))
                else:
                    res_i.append(len(residues_unique))
                    residues_unique.append(res)
            residues_unique_indexes.append(res_i)
        # map residue sets to bool matrix
        residues_codes = np.zeros((len(self.residues_), len(residues_unique)), np.bool)
        for i, res_list in enumerate(residues_unique_indexes):
            for res in res_list:
                residues_codes[i, res] = 1

        # get intersection
        self.distance_matrix = np.dot(residues_codes, residues_codes.T).astype(np.float32)
        residues_lens = np.sum(residues_codes, axis=1)
        # divide by min or max residue set len
        if self.distance_norm == "min":
            norm_matrix = np.zeros_like(self.distance_matrix)
            for i in range(norm_matrix.shape[0]):
                norm_matrix[i, :] = np.minimum(residues_lens, residues_lens[i])
        elif self.distance_norm == "max":
            norm_matrix = np.zeros_like(self.distance_matrix)
            for i in range(norm_matrix.shape[0]):
                norm_matrix[i, :] = np.maximum(residues_lens, residues_lens[i])
        else:
            norm_matrix = np.ones_like(self.distance_matrix)
        self.distance_matrix /= norm_matrix
        self.distance_matrix = 1. - self.distance_matrix
        return self.distance_matrix

    def cluster(self, predictions, residues):
        self.preprocess(predictions, residues)
        self.get_distance_matrix()
        self.fit(self.distance_matrix)
        self.rank_clusters()

class Clustering_DBSCAN(Clustering, DBSCAN):
    """
    DBSCAN clustering.

    Class for predictions clustering based on sklearn.DBSCAN.

    Args:
        params for sklearn.DBSCAN:
            eps: float; default=0.5;
            min_samples: int; default=5;
            algorithm: str; default="auto";
            leaf_size: int; default=30;
        **args: arguments for Clustering.
    """
    def __init__(self, 
        eps         = 0.5,
        min_samples = 5,
        algorithm   = "auto",
        leaf_size   = 30,
        **args):
        Clustering.__init__(self, **args)
        DBSCAN.__init__(self,
            eps         = eps,
            min_samples = min_samples,
            algorithm   = algorithm,
            leaf_size   = leaf_size,
        )

    def cluster(self, predictions, residues=[]):
        self.preprocess(predictions, residues)
        self.labels_ = self.fit_predict(self.predictions_[:, 1:4], 
            sample_weight=self.predictions_[:, 0])
        self.rank_clusters()
