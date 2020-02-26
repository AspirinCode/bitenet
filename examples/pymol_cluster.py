from bitenet import *
from bitenet.clustering import *
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("traj", help="path to trajectory folder")
parser.add_argument("preds", help="path to predictions file")
parser.add_argument("out", help="path to output image")
parser.add_argument("--method", help="clusterization algorithm", 
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

parser.add_argument("--density", help="if set, predictions densities will be shown", 
    action="store_true")

args = parser.parse_args()

# read predictions from file
_, predictions, residues = read_predictions(args.preds, get_residues=True)

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

# fit clusterization
clustering.cluster(predictions, residues)


cmd.load(os.path.join(args.traj, "0.pdb"), object="protein")
cmd.color(color_names_all[0], "protein and polymer.protein and name C*")
cmd.set("ray_shadows", 0)
cmd.set("cartoon_oval_length", 0.8)
cmd.set("cartoon_rect_length", 0.8)
cmd.set("sphere_scale", 0.7)
cmd.set("orthoscopic", True)
cmd.bg_color("white")
cmd.set("opaque_background", 0)
cmd.set("transparency_mode", 1)
view = \
    [-0.305338711,   -0.047106944,    0.951076210,\
     0.952173471,   -0.027010430,    0.304352403,\
     0.011352286,    0.998524666,    0.053101886,\
    -0.000005900,   -0.000001758, -240.384826660,\
    -4.131359100,    0.517762542,   -5.264386177,\
  -207.693450928,  688.463378906,   20.000000000]


if not args.density:
    # draw all predictions with spheres
    draw_clusters_predictions(clustering, name="predictions")
else:
    # draw cluster predictions densities
    draw_clusters_density(clustering, name="predictions") 
cmd.set_view(view)
cmd.png(args.out, width=720, height=720, dpi=-1, ray=1, quiet=1)

cmd.quit()