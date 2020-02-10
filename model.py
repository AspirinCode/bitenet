from .include import *

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

class Conv3d:
    """
    3d convolution layer.
    """
    def __init__(self, filters=32, kernel_size=3, stride=1, 
        padding="SAME", name="Conv3d",
        activation=tf.nn.relu, 
        regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
        use_batch_norm=True):
        self.name = name
        self.filters     = filters
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding
        self.activation  = activation
        self.regularizer = regularizer
        self.use_batch_norm = use_batch_norm

    def forward(self, input, training=False, **args):
        with tf.variable_scope(self.name):
            weights = tf.get_variable(name="w", 
                shape=(self.kernel_size, self.kernel_size, self.kernel_size, input.shape[-1], self.filters),
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), 
                regularizer=self.regularizer)
            x = tf.nn.conv3d(input, weights, strides=[1, self.stride, self.stride, self.stride, 1],
                padding=self.padding, name="conv")
            if self.use_batch_norm:
                x = tf.layers.batch_normalization(x, name="bn", training=training)
            if self.activation != None:
                x = self.activation(x)
        return x

    def __call__(self, input, training=False, **args):
        return self.forward(input, training=training, **args)

class Pool3d:
    def __init__(self, pool_size=2, stride=2, padding="SAME", name="Pool3d", **args):
        self.name = name
        self.pool_size = pool_size
        self.stride    = stride
        self.padding   = padding

    def forward(self, input, **args):
        with tf.variable_scope(self.name):
            pool = tf.layers.max_pooling3d(input, 
                pool_size=[self.pool_size, self.pool_size, self.pool_size],
                strides=[self.stride, self.stride, self.stride],
                padding=self.padding, name="pool")
        return pool

    def __call__(self, input, **args):
        return self.forward(input, **args)

class ConvBlock:
    """
    Convolution block layer: (Conv3d, Conv3d, Conv3dPool).
    """
    def __init__(self, filters=32, name="ConvBlock", **args):
        self.name = name
        self.filters = filters
        self.conv1   = Conv3d(filters=self.filters, name="conv1", **args)
        self.conv2   = Conv3d(filters=self.filters, name="conv2", **args)
        self.pool    = Conv3d(filters=self.filters, stride=2, name="pool", **args)

    def forward(self, input, **args):
        with tf.variable_scope(self.name):
            x = self.conv1(input, **args)
            x = self.conv2(x,     **args)
            x = self.pool(x,      **args)
        return x

    def __call__(self, input, **args):
        return self.forward(input, **args)

class PredictionActivation:
    """
    Final activation layer: 
    sigmoid function is applied to [:, :, :, :, 0] confidence of model output;
    prediction coordinates are renormalized from cell local coordinates to absolute:
    sigmoid function is applied and then added to cell indexes.
    """
    def __init__(self, name="PredictionActivation"):
        self.name = name

    def forward(self, input, name="prediction", **args):
        with tf.variable_scope(self.name):
            confidences = input[:, :, :, :, 0]
            centers     = input[:, :, :, :, 1:4]
            confidences = tf.reshape(confidences, 
                [-1, input.shape[1], input.shape[2], input.shape[3], 1])
            confidences = tf.sigmoid(confidences, name="confidences")
            indices = np.zeros((input.shape[1], input.shape[2], input.shape[3], 3))
            for i in range(input.shape[1]):
                for j in range(input.shape[2]):
                    for k in range(input.shape[3]):
                        indices[i, j, k, :] = [i, j, k]
            centers = tf.add(tf.nn.sigmoid(centers), indices, name="centers")
        out = tf.concat([confidences, centers], axis=-1, name=name)
        return out

    def __call__(self, input, **args):
        return self.forward(input, **args)

class Loss:
    """
    Cost function.
    """
    def __init__(self, cost_lambda=5.):
        self.cost_lambda = cost_lambda

    def position_loss(self, target, prediction, sample_weight=1.):
        centers_target      = target[:, :, :, :, 1:4]
        confidence_target   = target[:, :, :, :, 0]
        centers_prediction  = prediction[:, :, :, :, 1:4]

        pos_cost = tf.reduce_sum(tf.square(centers_target - centers_prediction), axis=-1)
        pos_cost = tf.multiply(pos_cost, confidence_target)
        pos_cost = tf.reduce_sum(pos_cost, axis=[1, 2, 3])

        pos_cost = tf.scalar_mul(self.cost_lambda, pos_cost)
        pos_cost = tf.multiply(pos_cost, sample_weight)

        return tf.reduce_mean(pos_cost, name="position_cost")

    def confidence_loss(self, target, prediction, sample_weight=1.):
        confidence_target     = target[:, :, :, :, 0]
        confidence_prediction = prediction[:, :, :, :, 0]
        confidence_cost = tf.square(confidence_target - confidence_prediction)
        confidence_cost = tf.reduce_sum(confidence_cost, axis=[1, 2, 3])

        confidence_cost = tf.multiply(confidence_cost, sample_weight)
        return tf.reduce_mean(confidence_cost, name="confidence_cost")

    def __call__(self, target, prediction, sample_weight=1.):
        cost = tf.add(
            self.position_loss(target, prediction, sample_weight),
            self.confidence_loss(target, prediction, sample_weight),
            name="prediction_cost")
        return cost

class Model:
    def __init__(self, 
        input_shape=(None, default_cube_size, default_cube_size, default_cube_size, default_channel_num),
        cell_size=default_cell_size):
        
        self.input_shape    = input_shape
        self.cell_size      = cell_size
        self.output_shape   = (None, 
            input_shape[1] // cell_size, input_shape[2] // cell_size, input_shape[3] // cell_size, 4)

        # number of size downsampling depends on cube_size/cell_size ratio
        pooling_num = int(math.log2(cell_size))
        args = {}

        # if input grid size is:
        # 64x64x64x11
        self.layers = [
            Conv3d(32, name="Conv1", **args),
            Conv3d(32, stride=2, name="Pool1", **args)]
        # 64x64x64x32
        # 32x32x32x32

        if pooling_num - 1 == 1:
            self.layers += [ConvBlock(64, **args)]
        elif pooling_num - 1 == 2:
            self.layers += [
                ConvBlock(32, name="ConvBlock1", **args), 
                ConvBlock(64, name="ConvBlock2", **args)]
            # 16x16x16x32
            # 8x8x8x64
        elif pooling_num - 1 == 3:
            self.layers += [
                ConvBlock(32, name="ConvBlock1", **args),
                ConvBlock(64, name="ConvBlock2", **args),
                ConvBlock(64, name="ConvBlock3", **args)]
        
        self.layers += [
            Conv3d(128, name="ConvFinal", **args),
            Conv3d(4, activation=None, use_batch_norm=False, **args),
        ]
        # 8x8x8x128
        # 8x8x8x4
        self.prediction_activation = PredictionActivation()

    def build(self, optimizer=tf.train.AdamOptimizer, 
        cost_lambda=default_params["cost_lambda"],
        cost_gamma=default_params["cost_gamma"]):
        """
        Builds computation graph.
        Args:
            optimizer: tf.train.Optimizer; default=tf.train.AdamOptimizer;
            cost_lambda: float; default=5.;
                parameter lambda for cost function;
            cost_gamma: float; default=1e-5;
                parameter gamma for regularization;
        """

        # placeholder for input grids
        self.X = tf.placeholder(tf.float32, shape=self.input_shape, name="input_grid")
        # placeholder for target labels
        self.Y = tf.placeholder(tf.float32, shape=self.output_shape, name="target")
        # placeholder for boolean for batch norm 
        self.training = tf.placeholder(tf.bool, (), name="training")
        
        x = self.X
        # forward
        for l in self.layers:
            x = l(x, training=self.training)
        # output activation function
        self.output = self.prediction_activation(x, name="output")

        # placeholder for learning_rate
        self.learning_rate = tf.placeholder(tf.float32, (), name="lrate_placeholder")
        # not used
        self.sample_weight = tf.placeholder(tf.float32, shape=(None), name="sample_weight")

        # loss function
        self.loss = Loss(cost_lambda=cost_lambda)
        # regularization terms
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss      = cost_gamma * tf.reduce_sum(reg_variables, name="reg_cost")
        # losses for prediction coordinates and confidence
        self.pos_loss      = self.loss.position_loss(self.Y, self.output, self.sample_weight)
        self.conf_loss     = self.loss.confidence_loss(self.Y, self.output, self.sample_weight)
        self.total_loss    = tf.add_n([reg_loss, self.pos_loss, self.conf_loss], name="total_cost")
        
        # optimizer
        self.optimizer = optimizer(learning_rate=self.learning_rate, name="optimizer")
        # applying gradients
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.total_loss)
        # weights saver
        self.saver = tf.train.Saver()
        self.saver_short = None
        self.saver_none  = tf.train.Saver(max_to_keep=None)


    def save(self, path, full=True, step=None):
        """
        Saves weights.
        Args:
            path: str; 
                path to output file;
            full: bool; default=True;
                if True gradients are saved;
            step: int; default=None;
                global_step for saver;
        """
        if full:
            if step == None:
                self.saver_none.save(self.sess, path)
            else:
                self.saver.save(self.sess, path, global_step=step)
        else:
            if self.saver_short == None:
                var_list = []
                for v in tf.global_variables():
                    if "Adam" not in v.name:
                        var_list.append(v)
                self.saver_short = tf.train.Saver(var_list)
            self.saver_short.save(self.sess, path, global_step=step)

    def load(self, path, full=False):
        """
        Loads graph and weights from file.
        Args:
            path: str;
                filename prefix for .meta file;
            full: bool; default=False;
                if True, gradients and cost tensors are loaded as well.
        """
        self.saver = tf.train.import_meta_graph(path + ".meta")
        self.saver.restore(self.sess, path)
        self.saver_short = None
        self.saver_none  = tf.train.Saver(max_to_keep=None)
        graph = tf.get_default_graph()
        self.X          = graph.get_tensor_by_name("input_grid:0")
        self.Y          = graph.get_tensor_by_name("target:0")
        self.training   = graph.get_tensor_by_name("training:0")
        self.output     = graph.get_tensor_by_name("output:0")
        if full:
            self.pos_loss   = graph.get_tensor_by_name("position_cost:0")
            self.conf_loss  = graph.get_tensor_by_name("confidence_cost:0")
            self.total_loss = graph.get_tensor_by_name("total_cost:0")
            self.learning_rate = graph.get_tensor_by_name("lrate_placeholder:0")
            self.sample_weight = graph.get_tensor_by_name("sample_weight:0")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = graph.get_operation_by_name("optimizer")
        

    def init_session(self, gpus="", cpu_only=False):
        """
        Initializes tensorflow session.
        Args:
            gpus: str; default="";
                available gpus;
            cpu_only: bool; default=False;
                if True, session will be ran on cpu only.
        """
        if cpu_only:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            gpu_options = tf.GPUOptions(visible_device_list=gpus)
            config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train_step(self, grids, targets, sample_weight=[], 
        minibatch_size=default_params["minibatch_size"], learning_rate=1e-3):
        """
        Single train step of model.

        Args:
            grids: np.array of shape (n_grids, cube_size, cube_size, cube_size, n_channels);
                input cubic grids for model forward pass;
            targets: np.array of shape (n_grids, N, N, N, 4) where N is the number of cells;
                true target labels for loss calculation;
            sample_weight: not used;
            minibatch_size: int; default=32;
                minibatch size, number of grids in single forward pass; 
                n_grids will be splitted to these minibatches;
            learning_rate: float; default=1e-3;
                optimizer learning rate.

        Returns: (predictions, [pos_loss, conf_loss, total_loss]):
            predictions: np.array of shape (n_grids, N, N, N, 4);
                model output for each grid;
            pos_loss: float; 
                loss value for predictions coordinates;
            conf_loss: float;
                loss value for predictions confidence values;
            total_loss: float;
                total loss value = pos_loss + conf_loss + reg_loss.
        """

        if len(sample_weight) == 0:
            sample_weight = np.ones((len(targets)))
        prediction_list, pos_loss, conf_loss, total_loss = [], 0., 0., 0.
        for minibatch_index in range(math.ceil(len(grids) / minibatch_size)):
            i_start = minibatch_index * minibatch_size
            sw = sample_weight[i_start : i_start + minibatch_size]

            res = self.sess.run([self.train_op, self.output, self.pos_loss, self.conf_loss, self.total_loss],
                feed_dict={self.X : grids[i_start : i_start + minibatch_size],
                    self.Y : targets[i_start : i_start + minibatch_size],
                    self.sample_weight : sw, self.learning_rate : learning_rate,
                    self.training : True})
            prediction_list.append(res[1])
            pos_loss    += res[2] * np.sum(sw)
            conf_loss   += res[3] * np.sum(sw)
            total_loss  += res[4] * np.sum(sw)
        pos_loss    /= np.sum(sample_weight)
        conf_loss   /= np.sum(sample_weight)
        total_loss  /= np.sum(sample_weight)
        return np.concatenate(prediction_list), [pos_loss, conf_loss, total_loss]

    def predict(self, grids, minibatch_size=default_params["minibatch_size"]):
        """
        Retrieving predictions for input grids.

        Args:
            grids: np.array of shape (n_grids, cube_size, cube_size, cube_size, n_channels);
                input cubic grids;
            minibatch_size: int; default=32;
                minibatch size, number of grids in single forward pass; 
                n_grids will be splitted to these minibatches.
        
        Returns:
            predictions: np.array of shape (n_grids, N, N, N, 4);
                model output for each grid.
        """
        prediction_list = []
        for minibatch_index in range(math.ceil(len(grids) / minibatch_size)):
            i_start = minibatch_index * minibatch_size

            predictions = self.sess.run(self.output, 
                feed_dict={
                    self.X : grids[i_start : i_start + minibatch_size],
                    self.training : False})
            prediction_list.append(predictions)
        if len(prediction_list) > 0:
            return np.concatenate(prediction_list)
        else:
            return np.array([])

    def __call__(self, grids, **args):
        return self.predict(grids, **args)