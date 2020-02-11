import os
import argparse
import numpy as np
from bitenet.process import read_predictions
from bitenet.clustering import Clustering_MeanShift, Clustering_DBSCAN, \
    Clustering_Agglomerative, Clustering_Agglomerative_Residues

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to predictions file")
    parser.add_argument("out",  help="path to folder to write outputs")
    
    parser.add_argument("--method", help="clustering method to use",
        choices=["meanshift", "dbscan", "agglomerative", "agglomerative_residues"],
        default="agglomerative_residues")
    parser.add_argument("--score_threshold", 
        help="score threshold for predictions", 
        type=float, default=0.1)
    parser.add_argument("--cluster_score_threshold_mean", 
        help="score threshold for filtering of clusters based on mean trajectory score",
        type=float, default=0.0001)
    parser.add_argument("--distance_merge", 
        help="bandwidth parameter for meanshift",
        type=float, default=5.)
    parser.add_argument("--eps", 
        help="eps parameter for dbscan",
        type=float, default=0.5)
    parser.add_argument("--n_clusters", 
        help="n_clusters for agglomerative clustering",
        type=int, default=10)

    args = parser.parse_args()

    # read predictions from file
    _, predictions, residues = read_predictions(args.path, get_residues=True)

    # init clustering class
    if args.method == "meanshift":
        clustering = Clustering_MeanShift(
            distance_merge=args.distance_merge)
    elif args.method == "dbscan":
        clustering = Clustering_DBSCAN(eps=args.eps)
    elif args.method == "agglomerative":
        clustering = Clustering_Agglomerative(
            n_clusters=args.n_clusters)
    elif args.method == "agglomerative_residues":
        clustering = Clustering_Agglomerative_Residues(
            n_clusters=args.n_clusters)

    clustering.score_threshold = args.score_threshold
    clustering.cluster_score_threshold_mean = args.cluster_score_threshold_mean
    
    # clustering fit
    clustering.cluster(predictions, residues)

    os.makedirs(args.out, exist_ok=True)
    # print summary
    print(clustering.get_summary_str(all=False))
    # write summary
    with open(os.path.join(args.out, "output.log"), "w") as file:
        file.write(clustering.get_summary_str())
    # write csv summary file
    clustering.export_summary(os.path.join(args.out, "clusters.csv"), all=False)
    # write csv summary file for all clusters
    clustering.export_summary(os.path.join(args.out, "clusters_all.csv"), all=True)
    # plot step cluster scores and save image to file
    clustering.plot(os.path.join(args.out, "plot.png"))

if __name__ == "__main__":
    main()