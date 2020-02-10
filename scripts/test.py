import os
import argparse
from bitenet.include import read_dataset, Timer
from bitenet.process import PredictionAccuracy, PredictionLogger
from bitenet import BiteNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path",     help="path to input dataset file or folder")
    parser.add_argument("--out",    help="path to output prediction file", default="")
    parser.add_argument("--model",  help="path to model folder", default="")
    parser.add_argument("--gpus",   help="visible gpu devices", default="")
    parser.add_argument("--cpu_only", help="if set, model will run on cpu", action="store_true")

    parser.add_argument("--minibatch_size", 
        help="minibatch size for cnn", 
        type=int, default=-1)
    parser.add_argument("--score_threshold",
        help="score threshold for predictions",
        type=float, default=-1)
    parser.add_argument("--distance_threshold_nms",
        help="distance threshold for predictions non max suppresssion",
        type=float, default=-1)
    
    parser.add_argument("--distance_threshold", 
        help="distance threshold for predictions to be considered true positive",
        type=float, default=-1)
    parser.add_argument("--num_boxes", 
        help="number of top predictions for accuracy calculation: >0 for number of boxes; -1 for top-N; -2 for top-(N+2); -3 for all.",
        type=int, default=-4)
    
    args = parser.parse_args()
    params = {"gpus": args.gpus, "cpu_only" : args.cpu_only}
    if len(args.model) > 0:
        params["model_path"]         = args.model
    if args.minibatch_size > 0:
        params["minibatch_size"]     = args.minibatch_size
    if args.score_threshold >= 0:
        params["score_threshold"]    = args.score_threshold
    if args.distance_threshold_nms >= 0:
        params["distance_threshold"] = args.distance_threshold_nms
    
    timer = Timer()
    # init bitenet
    bitenet = BiteNet(**params)
    bitenet.prediction_processer.residues = False
    prediction_accuracy = PredictionAccuracy(score_threshold=0.)
    if args.distance_threshold >= 0:
        prediction_accuracy.distance_threshold = args.distance_threshold
    if args.num_boxes >= -3:
        prediction_accuracy.num_boxes = args.num_boxes
    # load pdbs dataset
    if os.path.isdir(args.path):
        bitenet.dataloader.set_params(path=args.path)
    elif os.path.isfile(args.path):
        dataset = read_dataset(args.path)
        bitenet.dataloader.set_params(
            path=os.path.dirname(args.path), pdbs=dataset)
    else:
        print("no such file or folder")
        exit()
    
    logger = None
    if len(args.out) > 0:
        logger = PredictionLogger(args.out)
    step_index = 0
    predictions_accum, interfaces_accum = [], []
    # for batch
    for grids, _, floors, interfaces, names in bitenet.dataloader:
        # forward pass
        predictions = bitenet.model(grids, minibatch_size=bitenet.minibatch_size)
        # get predictions
        predictions = bitenet.prediction_processer(predictions, floors)

        predictions_accum += predictions
        interfaces_accum  += interfaces
        # get accuracy
        acc = prediction_accuracy(interfaces, predictions)
        step_index += len(floors)
        # print
        print("{:10s} {:5d}/{:5d} {:5.3f} {:5.3f} {:5.3f}".format(
                str(timer), step_index, len(bitenet.dataloader),
                acc[0], acc[1], acc[2]), flush=True, end="\r")
        # log
        if logger:
            logger.write(names, predictions)
    # overall accuracy
    acc = prediction_accuracy(interfaces_accum, predictions_accum)
    print("{:10s} {:5s} {:5d} {:5.3f} {:5.3f} {:5.3f}".format(
            str(timer), "", len(bitenet.dataloader),
            acc[0], acc[1], acc[2]))

if __name__ == "__main__":
    main()