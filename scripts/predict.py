from bitenet import BiteNet
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to input pdb file, dataset file or folder")
    parser.add_argument("out",  help="path to output predictions file or folder")
    parser.add_argument("--separate", help="if set, folder will be created and separate output predictions file for each input will be written", 
        action="store_true")
    parser.add_argument("--model", help="path to model folder", default="")
    parser.add_argument("--gpus", help="visible gpu devices", default="")
    parser.add_argument("--cpu_only", help="if set, model will run on cpu", action="store_true")

    parser.add_argument("--minibatch_size", 
        help="minibatch size for cnn", 
        type=int, default=-1)
    parser.add_argument("--score_threshold", 
        help="score threshold for predictions", 
        type=float, default=-1)
    parser.add_argument("--distance_threshold", 
        help="distance threshold for predictions non max suppression", 
        type=float, default=-1)
    parser.add_argument("--distance_residues", 
        help="distance threshold for residues to be considered on interface", 
        type=float, default=-1)
    
    args = parser.parse_args()

    params = {"gpus": args.gpus, "cpu_only" : args.cpu_only}
    if len(args.model) > 0:
        params["model_path"]         = args.model
    if args.minibatch_size > 0:
        params["minibatch_size"]     = args.minibatch_size
    if args.score_threshold >= 0:
        params["score_threshold"]    = args.score_threshold
    if args.distance_threshold >= 0:
        params["distance_threshold"] = args.distance_threshold
    if args.distance_residues >= 0:
        params["distance_residues"]  = args.distance_residues

    bitenet = BiteNet(**params)
    bitenet(args.path, out=args.out, separate=args.separate)

if __name__ == "__main__":
    main()