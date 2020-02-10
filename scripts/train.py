from bitenet.include import *
from bitenet.data import DataLoader
from bitenet.model import *
from bitenet.process import *
import argparse

class LearningRateScheduler:
    def __init__(self, **args):
        self.step = 0
    def call(self):
        return 1e-3
    def __call__(self):
        value = self.call()
        self.step += 1
        return value

class LearningRate_constant(LearningRateScheduler):
    def __init__(self, learning_rate=1e-3):
        super(LearningRate_constant, self).__init__()
        self.learning_rate = 1e-3
    def call(self):
        return self.learning_rate

class LearningRate_decay(LearningRateScheduler):
    def __init__(self, initial_learning_rate=1e-3, final_learning_rate=1e-5, steps=100000):
        super(LearningRate_decay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.final_learning_rate = final_learning_rate
        self.steps = steps
    def call(self):
        decay = math.pow(self.final_learning_rate / self.initial_learning_rate, 1 / self.steps)
        return self.initial_learning_rate * math.pow(decay, self.step)

class LearningRate_decay2(LearningRateScheduler):
    def __init__(self, initial_learning_rate=1e-4, max_learning_rate=1e-3, final_learning_rate=1e-5,
        steps_up=10000, steps_down=100000):
        super(LearningRate_decay2, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.max_learning_rate = max_learning_rate
        self.final_learning_rate = final_learning_rate
        self.steps_up = steps_up
        self.steps_down = steps_down

    def call(self):
        if self.step <= self.steps_up:
            decay = math.pow(self.max_learning_rate / self.initial_learning_rate, 1 / self.steps_up)
            lr = self.initial_learning_rate * math.pow(decay, self.step)
        else:
            decay = math.pow(self.final_learning_rate / self.max_learning_rate, 1 / self.steps_down)
            lr = self.max_learning_rate * math.pow(decay, (self.step - self.steps_up))
        return lr


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to model folder")
    parser.add_argument("--data", help="path to pdbs folder or pdb list file", default="./data/")
    parser.add_argument("--gpus", help="visible gpu devices", default="")
    parser.add_argument("--restore", help="if set, model will be restored", action="store_true")
    parser.add_argument("--params", help="path to params json file", default="")
    parser.add_argument("-v", "--verbosity", help="verbosity level", type=int, default=1)

    parser.add_argument("--voxel_size", help="grid voxel size", type=float, default=-1)
    parser.add_argument("--density_cutoff", help="grid density cutoff", type=float, default=-1)
    parser.add_argument("--cube_size", help="model input grid cube size", type=int, default=-1)
    parser.add_argument("--cell_size", help="model output cell size", type=int, default=-1)
    parser.add_argument("--cube_stride", help="stride of grid cube", type=int, default=-1)

    args = parser.parse_args()

    timer = Timer()
    
    # create folders
    train_dir       = os.path.join(args.path, "train")
    logs_dir        = os.path.join(train_dir, "logs")
    chekpoint_dir   = os.path.join(train_dir, "checkpoints")
    weights_dir     = os.path.join(args.path, "weights")
    model_name      = os.path.basename(args.path)
    if len(model_name) == 0:
        model_name = os.path.basename(os.path.dirname(args.path))

    if args.restore:
        # load params
        params           = load_params(os.path.join(args.path, "params.json"))
        # load train dataloader
        dataloader_train = load(os.path.join(train_dir, "dataloader_train.pkl"))
        # load test dataloader
        dataloader_test  = load(os.path.join(train_dir, "dataloader_test.pkl"))
        # init session
        model = Model()
        model.init_session(gpus=args.gpus)
        # get last checkpoint index
        checkpoints = [int(f.name[6:-5]) for f in os.scandir(chekpoint_dir) if "train-" in f.name and ".meta" in f.name]
        last_epoch = max(checkpoints)
        # load model weights
        model.load(os.path.join(chekpoint_dir, "train-" + str(last_epoch)), full=True)
        start_epoch = last_epoch + 1
        # load learning rate 
        learning_rate = load(os.path.join(train_dir, "learning_rate.pkl"))
    else:
        # init params 
        params = default_params
        if os.path.isfile(args.params):
            params.update(load_params(args.params))
        if args.voxel_size > 0:
            params["voxel_size"] = args.voxel_size
        if args.density_cutoff > 0:
            params["density_cutoff"] = args.density_cutoff
        if args.cube_size > 0:
            params["cube_size"] = args.cube_size
        if args.cell_size > 0:
            params["cell_size"] = args.cell_size
        if args.cube_stride > 0:
            params["cube_stride"] = args.cube_stride

        random.seed(params["seed"])
        tf.set_random_seed(params["seed"])

        # read pdb dataset
        if os.path.isfile(args.data):
            path, train_list, test_list = read_set(args.data)
        elif os.path.isdir(args.data):
            pdb_list = [f.name[:-4] for f in os.scandir(args.data) if f.name[-4:] == ".pdb"]
            path = args.data
            train_list, test_list = get_simple_splits(pdb_list, seed=params["seed"])
        else:
            path, train_list, test_list = "", [], []

        if args.verbosity >= 2:
            print("Dataset:", path, "Train/test size:", len(train_list), len(test_list))
        # init dataloader
        dataloader_train = DataLoader(path, train_list, train=True, **params)
        dataloader_test  = DataLoader(path, test_list, **params)

        # create folders and write dataloader and params
        os.makedirs(args.path, exist_ok=True)
        os.makedirs(train_dir, exist_ok=True)
        write_set(path, train_list, test_list, os.path.join(train_dir, "dataset.log"))
        save(dataloader_train, os.path.join(train_dir, "dataloader_train.pkl"))
        save(dataloader_test,  os.path.join(train_dir, "dataloader_test.pkl"))
        save_params(params, os.path.join(args.path, "params.json"))

        # build model graph and init session
        cube_size, cell_size, channel_num = params["cube_size"], params["cell_size"], params["channel_num"]
        grid_shape = (None, cube_size, cube_size, cube_size, channel_num)
        model = Model(grid_shape, cell_size)
        model.build(cost_lambda=params["cost_lambda"], cost_gamma=params["cost_gamma"])
        model.init_session(gpus=args.gpus)

        start_epoch = 0
        learning_rate = LearningRate_decay(1e-3, 1e-5, 
            len(dataloader_train) * params["epoch_num"] / params["batch_size"])

    # create folders
    os.makedirs(logs_dir,       exist_ok=True)
    os.makedirs(chekpoint_dir,  exist_ok=True)
    os.makedirs(weights_dir,    exist_ok=True)

    # predictions processer
    prediction_processer = PredictionProcesser(
        score_threshold=0.1, max_boxes=5, residues=False, **params)
    prediction_accuracy  = PredictionAccuracy(score_threshold=0.1)

    # logging
    logger       = Logger(os.path.join(logs_dir, "epochs.log"), clear=not args.restore)
    logger_train = Logger(os.path.join(logs_dir, "train.log"),  clear=not args.restore)
    logger_test  = Logger(os.path.join(logs_dir, "test.log"),   clear=not args.restore)

    best_score = 0
    # for epoch
    for epoch in range(start_epoch, params["epoch_num"]):
        
        step_index = 0
        losses_epoch = []
        prediction_accum, interfaces_accum = [], []
        # epoch steps logging
        logger_predictions = PredictionLogger(os.path.join(logs_dir, "train_predictions_{:d}.log".format(epoch)), clear=True)
        logger_interfaces  = TargetLogger(os.path.join(logs_dir, "train_interfaces_{:d}.log".format(epoch)))
        # for batch in train
        for grids, targets, floors, interfaces, names in dataloader_train:
            # train on batch
            predictions, losses = model.train_step(grids, targets, 
                learning_rate=learning_rate(), minibatch_size=params["minibatch_size"])
            step_index += len(floors)
            losses_epoch.append(losses)
            # get predictions
            predictions = prediction_processer(predictions, floors)
            # get accuracy
            acc = prediction_accuracy(interfaces, predictions)
            prediction_accum += predictions
            interfaces_accum += interfaces

            # print and log
            if args.verbosity >= 2:
                print("{:<20s} {:10s} {:3d}t{:5d}/{:5d}|{:7.3f}{:8.3f}{:8.3f}|{:5.3f} {:5.3f} {:5.3f}".format(
                    model_name, str(timer), epoch, step_index, len(dataloader_train), 
                    losses[0], losses[1], losses[2], 
                    acc[0], acc[1], acc[2]), flush=True, end="\r")
            logger_train.write([str(timer), epoch, step_index] + losses + list(acc))
            logger_predictions.write(names, predictions)
            logger_interfaces.write(names, interfaces)

        # print accuracy on train
        acc = prediction_accuracy(interfaces_accum, prediction_accum)
        losses = np.mean(np.array(losses_epoch), axis=0).tolist()
        if args.verbosity >= 1:
            print("{:<20s} {:10s} {:3d}t{:5s} {:5s}|{:7.3f}{:8.3f}{:8.3f}|{:5.3f} {:5.3f} {:5.3f}".format(
                model_name, str(timer), epoch, "", "", 
                losses[0], losses[1], losses[2],
                acc[0], acc[1], acc[2]), flush=True, end="\r")
        # save
        model.save(os.path.join(chekpoint_dir, "train"), step=epoch)
        save(learning_rate, os.path.join(train_dir, "learning_rate.pkl"))

        step_index = 0
        prediction_accum, interfaces_accum = [], []
        logger_predictions = PredictionLogger(os.path.join(logs_dir, "test_predictions_{:d}.log".format(epoch)), clear=True)
        # for batch in test
        for grids, targets, floors, interfaces, names in dataloader_test:
            # forward pass
            predictions = model(grids, minibatch_size=params["minibatch_size"])
            # get predictions
            predictions = prediction_processer(predictions, floors)
            prediction_accum += predictions
            interfaces_accum += interfaces
            # get accuracy
            acc = prediction_accuracy(interfaces, predictions)
            step_index += len(floors)
            # print and log
            if args.verbosity >= 2:
                print("{:<20s} {:10s} {:3d}v{:5d}/{:5d}|{:7.3f}{:8.3f}{:8.3f}|{:5.3f} {:5.3f} {:5.3f}".format(
                    model_name, str(timer), epoch, step_index, len(dataloader_test), 
                    losses[0], losses[1], losses[2], 
                    acc[0], acc[1], acc[2]), flush=True, end="\r")
            logger_test.write([str(timer), epoch, step_index] + list(acc))
            logger_predictions.write(names, predictions)

        # print accuracy on test
        acc = prediction_accuracy(interfaces_accum, prediction_accum)
        if args.verbosity >= 1:
            print("{:<20s} {:10s} {:3d} {:5s} {:5s}|{:7.3f}{:8.3f}{:8.3f}|{:5.3f} {:5.3f} {:5.3f}".format(
                model_name, str(timer), epoch, "", "", 
                losses[0], losses[1], losses[2], 
                acc[0], acc[1], acc[2]))
        # log 
        logger.write([str(timer), epoch] + losses + list(acc))
        if epoch == 0:
            logger_interfaces = TargetLogger(os.path.join(logs_dir, "test_interfaces.log"))
            logger_interfaces.write(dataloader_test.pdbs, interfaces_accum)
        # save best
        if acc[2] > best_score:
            best_score = acc[2]
            model.save(os.path.join(weights_dir, "best"))
        
    model.save(os.path.join(weights_dir, "final"), full=False)


if __name__ == "__main__":
    main()