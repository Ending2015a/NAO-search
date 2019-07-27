import os
import sys
import time
import logging

import numpy as np
import tensorflow as tf

from encoder import encoder
from decoder import decoder

from copy import deepcopy
from collections import OrderedDict

from . import tf_util
from .tf_util import TensorboardWriter

class BaseModel:
    def __init__(self,
                encoder_num_layers = 1,
                encoder_hidden_size = 96,
                encoder_emb_size = 32,
                mlp_num_layers = 0,
                mlp_hidden_size = 32,
                mlp_dropout = 0.5,
                decoder_num_layers = 1,
                decoder_hidden_size = 32,
                source_length = 60,       # input source length
                encoder_length = 60,      # encoder length, the input source will be reshaped to match this encoder length: (batch_size, encoder_length, source_length// encoder_length)
                decoder_length = 60,      # decoder length
                encoder_dropout = 0.0,
                decoder_dropout = 0.0,
                weight_decay = 1e-4,
                encoder_vocab_size = 21,
                decoder_vocab_size = 21,
                trade_off = 0.5,
                batch_size = 128,
                learning_rate = 1.0,
                optimizer = 'adam',
                start_decay_step = 100,
                decay_steps = 1000,
                decay_factor = 0.9,
                attention = False,
                max_gradient_norm = 5.0,
                beam_width = 0,
                time_major = False,

                num_cpu = 0,
                full_tensorboard_log = False,
                _init_setup_model=False):

        '''
        :param encoder_num_layers: (int) The number of hidden layers of the encoder
        :param encoder_hidden_size: (int) The hidden layer size of the encoder
        :param encoder_emb_size: (int) The encoder embedding size
        :param mlp_num_layers: (int) The number of layers of the multilayer perceptron behind the encoder
        :param mlp_hidden_size:
        :param mlp_dropout:
        :param decoder_num_layers:
        :param decoder_hidden_size:
        :param source_length: (int) The input sequence length
        :param encoder_length: (int) The encoder length. The input source will be folded to match this encoder length.
        :param decoder_length: (int)
        :param encoder_dropout:
        :param decoder_dropout:
        :param weight_decay:
        :param encoder_vocab_size:
        :param decoder_vocab_size:
        :param trade_off:
        :param batch_size:
        :param learning_rate:
        :param optimizer: (str)
        :param start_decay_step:
        :param decay_steps:
        :param decay_factor:
        :param attention:
        :param max_gradient_norm:
        :param beam_width:
        :param time_major:
        :param num_cpu:
        '''
    
        self.encoder_num_layers = encoder_num_layers
        self.encoder_hidden_size = encoder_hidden_size 
        self.encoder_emb_size = encoder_emb_size
        self.mlp_num_layers = mlp_num_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_dropout = mlp_dropout
        self.decoder_num_layers = decoder_num_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.source_length = source_length
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.weight_decay = weight_decay
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.trade_off = trade_off
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.start_decay_step = start_decay_step
        self.decay_steps = decay_steps
        self.decay_factor = decay_factor
        self.attention = attention
        self.max_gradient_norm = max_gradient_norm
        self.beam_width = beam_width
        self.time_major = time_major


        self.num_cpu = num_cpu
        self.full_tensorboard_log = full_tensorboard_log
        self.graph = None
        self.sess = None
        self.encoder_ph = None
        self.decoder_ph = None
        self.target_ph = None

        # private
        self._param_load_ops = None


        self.LOG = logging.getLogger()

        if _init_setup_model:
            self._setup_model()

    def _data_preprocess(self, X, y):

    def _data_postprocess(self, X, y):


    def learn(self, X, y, epochs=10, callback=None, log_interval=1, tb_log_name="epd", reset_num_timesteps=True):
        X_bak = np.array(X, copy=True, dtype=np.int32)
        y_bak = np.array(y, copy=True, dtype=np.float32)

        assert np.all(X_bak >= 0 and X_bak < encoder_vocab_size), ValueError("Each element of input 'X' must be in the range of the vocab size")
        assert np.all(y_bak >= 0.0 and y_bak <= 1.0), ValueError("Each element of input 'y' must be normalized between 0 ~ 1")
        assert len(X_bak) == len(y_bak), ValueError("The size of input 'X' and 'y' must be equal")
        assert len(X_bak) > 10, "The number of training samples 'X' must be greater than 10"

        new_tb_log = self._initialize_global_step(reset_num_timesteps)

        with TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) as writer:

            # calculate the total number of items
            num_items = len(y_bak)

            # shuffled indices
            index_shuffle = [idx for idx in range(len(num_items))]
            np.random.shuffle(index_shuffle)

            # shuffled samples
            X_shuffle = X_bak[index_shuffle]
            y_shuffle = y_bak[index_shuffle]

            # calculate training data size
            train_num = int(num_items * 0.7)
            eval_num = num_items - train_num

            # create dataset
            X_train = X_shuffle[:train_num]
            y_train = y_shuffle[:train_num]

            X_eval = X_shuffle[train_num:]
            y_eval = y_shuffle[train_num:]

            # total training steps of each epoch
            n_updates = train_num // self.batch_size

            t_first_start = time.time()

            for update in range(1, n_updates+1):

                t_start = time.time()
                




    def predict(self, seeds, lambdas=[10, 20, 30]):

    @classmethod
    def load(cls, load_path):
        data, params = cls._load_model(load_path)
        
        model = cls(_init_setup_model=False)
        model.__dict__.update(data)
        model._setup_model()
        
        model._load_parameters(params)
        
        return model


    def _build_param_dict(self):

        data = {
            'encoder_num_layers':  self.encoder_num_layers,
            'encoder_hidden_size': self.encoder_hidden_size,
            'encoder_emb_size':    self.encoder_emb_size,
            'mlp_num_layers':      self.mlp_num_layers,
            'mlp_hidden_size':     self.mlp_hidden_size,
            'mlp_dropout':         self.mlp_dropout,
            'decoder_num_layers':  self.decoder_num_layers,
            'decoder_hidden_size': self.decoder_hidden_size,
            'source_length':       self.source_length,
            'encoder_length':      self.encoder_length,
            'decoder_length':      self.decoder_length,
            'encoder_dropout':     self.encoder_dropout,
            'decoder_dropout':     self.decoder_dropout,
            'weight_decay':        self.weight_decay,
            'encoder_vocab_size':  self.encoder_vocab_size,
            'decoder_vocab_size':  self.decoder_vocab_size,
            'trade_off':           self.trade_off,
            'batch_size':          self.batch_size,
            'learning_rate':       self.learning_rate,
            'optimizer':           self.optimizer,
            'start_decay_step':    self.start_decay_step,
            'decay_steps':         self.decay_steps,
            'decay_factor':        self.decay_factor,
            'attention':           self.attention,
            'max_gradient_norm':   self.max_gradient_norm,
            'beam_width':          self.beam_width,
            'time_major':          self.time_major
        }
        
        return data

    def save(self, save_path):
        
        data = self._build_param_dict()

        params = self._get_parameters()
        
        self._save_model(save_path, data, params)

        
    def _setup_model(self):

        self.params = self._build_param_dict()

        # get cpu count
        n_cpu = self.num_cpu if self.num_cpu > 0 else multiprocessing.cpu_count()
        self.num_cpu = n_cpu

        # create graph
        self.graph = tf.Graph()
        with self.graph.as_default():

            # create session
            self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

            # placeholder
            self.encoder_ph = tf.placeholder(shape=(None, self.source_length), tf.int32)
            self.decoder_ph = tf.placeholder(shape=(None, self.source_length), tf.int32)
            self.decoder_input_ph = tf.placeholder(shape=(None, self.source_length), tf.int32)

            self.target_ph = tf.placeholder(shape=(None, ), tf.float32)


            with tf.variable_scope('model'):
            
                # === train ===
                # encoder
                self.train_encoder = encoder.Model(self.encoder_ph, self.target_ph, self.params, mode=tf.estimator.ModeKeys.TRAIN, scope='Encoder', reuse=False)

                encoder_outputs = train_encoder.encoder_outputs
                encoder_state = train_encoder.arch_emb
                encoder_state.set_shape([None, self.decoder_hidden_size])
                encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
                encoder_state = (encoder_state,) * self.decoder_num_layers

                #decoder_input_pad = tf.pad(self.decoder_ph, [[0, 0], [1, 0]], "CONSTANT", constant_values=0)
                #decoder_input = tf.slice(decoder_input_pad, [0, 0], [None, -1])

                # decoder
                self.train_decoder = decoder.Model(encoder_outputs, encoder_state, self.decoder_input_ph, self.decoder_ph, self.params, mode=tf.estimator.ModeKeys.TRAIN, scope='Decoder')

                # get loss
                encoder_loss = self.train_encoder.loss
                decoder_loss = self.train_decoder.loss

                # set reuse variables
                tf.get_variable_scope().reuse_variables()

                # === predict ===

                # encode old arch
                self.eval_encoder = encoder.Model(self.encoder_ph, None, self.params, mode=tf.estimator.ModeKeys.PREDICT, scope='Encoder', reuse=True)
                encoder_outputs = self.eval_encoder.encoder_outputs
                encoder_state = self.eval_encoder.arch_emb
                encoder_state.set_shape([None, self.decoder_hidden_size])
                encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
                encoder_state = (encoder_state,) * self.decoder_num_layers

                self.eval_decoder = decoder.Model(encoder_outputs, encoder_state, None, None, pself.params, mode=tf.estimator.ModeKeys.PREDICT, scope='Decoder')
                
                # predict new arch embedding
                res = self.eval_encoder.infer()
                predict_value = res['predict_value']
                arch_emb = res['arch_emb']
                new_arch_emb = res['new_arch_emb']
                new_arch_outputs = res['new_arch_outputs']

                # decode old arch (evaluate)
                res = self.eval_decoder.decode()
                sample_id = res['sample_id']

                encoder_state = new_arch_emb
                encoder_state.set_shape([None, self.decoder_hidden_size])
                encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
                encoder_state = (encoder_state,) * self.decoder_num_layers

                # decode new arch
                self.pred_decoder = decoder.Model(new_arch_outputs, encoder_state, None, None, self.params, mode=tf.estimator.ModeKeys.PREDICT, 'Decoder')
                res = my_decoder.decode()
                new_sample_id = res['sample_id']


            # compute loss
            with tf.variable_scope('loss'):
                encoder_loss = tf.identity(encoder_loss, 'encoder_loss')
                decoder_loss = tf.identity(decoder_loss, 'decoder_loss')
                decay_loss = self.weight_decay * tf.add_n( [tf.nn.l2_loss(v) for v in tf.trainable_variables()] )
                total_loss = self.trade_off * encoder_loss + (1. - self.trade_off) * decoder_loss + decay_loss

                tf.summary.scalar('encoder_loss', encoder_loss)
                tf.summary.scalar('decoder_loss', decoder_loss)
                tf.summary.scalar('decay_loss', decay_loss)
                tf.summary.scalar('total_loss', total_loss)

            # global step
            global_step = tf.train.get_or_create_global_step()
            # learning rate
            learning_rate = tf.constant(self.learning_rate)

            # optimizer
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                gradients, variables = zip(*opt.compute_gradients(total_loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

                # training operations
                train_op = opt.apply_gradients(
                        zip(clipped_gradients, variables), global_step=global_step)

            # summary
            with tf.variable_scope('model'):
                self.parameters = tf.trainable_variables()
                if self.full_tensorboard_log:
                    for var in self.parameters:
                        tf.summary.histogram(var.name, var)


            with tf.variable_scope('info'):
                learning_rate = tf.identity(opt._lr, 'learning_rate')
                tf.summary.scalar('learning_rate', learning_rate)

            
            self.encoder_loss = encoder_loss
            self.decoder_loss = decoder_loss
            self.decay_loss = decay_loss
            self.total_loss = total_loss
            self.global_step = global_step
            self.lr = learning_rate

            # training op
            self.train_op = train_op

            # prediction op
            self.predict_ops = {
                    'arch': decoder_target,
                    'ground_truth_value': encoder_target, # ground truth score
                    'predict_value': predict_value,       # predicted score
                    'sample_id': sample_id,               # old arch (evaluate)
                    'new_sample_id': new_sample_id        # new arch
                }

            tf.global_variables_initializer().run(session=self.sess)

            # summary op
            self.summary_op = tf.summary.merge_all()


    def _train_step(self):
        pass

    def _initialize_global_step(reset_num_timesteps):

        if reset_num_timesteps:
            self.sess.run(
                    [self.global_step.assign(0)]
                    )

        return self.sess.run( [self.global_step] ) == 0

    @staticmethod
    def _save_model(save_path, data, params=None):
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == '':
                save_path += '.pkl'
                
            with open(save_path, 'wb') as file_:
                cloudpickle.dump((data, params), file_)
        else:
            cloudpickle.dump((data, params), save_path)



    @staticmethod
    def _load_model(load_path):
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + '.pkl'):
                    load_path += '.pkl'
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))
                    
            with open(load_path, 'rb') as file_:
                data, params = cloudpickle.load(file_)
                
        else:
            data, params = cloudpickle.load(load_path)
            
        return data, params
    
    def _setup_load_operations(self):
        
        if self._param_load_ops is not None:
            raise RuntimeError("Parameter load operations have already been created")

        loadable_parameters = self._get_parameter_list()
    
        self._param_load_ops = OrderedDict()
        with self.graph.as_default():
            for param in loadable_parameters:
                placeholder = tf.placeholder(dtype=param.dtype, shape=param.shape)
                self._param_load_ops[param.name] = (placeholder, param.assign(placeholder))

    def _load_parameters(self, load_dict):
        
        if self._param_load_ops is None:
            self._setup_load_operations()

        params = None
        if isinstance(load_dict, dict):
            params = load_dict
        else:
            raise ValueError("Unknown type of load_dict, the load_doct mustbe dict type")

        feed_dict = {}
        param_update_ops = []
        not_updated_variables = set(self._param_load_ops.keys())
        
        for param_name, param_value in params.items():
            placeholder, assign_op = self._param_load_ops[param_name]
            feed_dict[placeholder] = param_value

            param_update_ops.append(assign_op)

            not_updated_variables.remove(param_name)

            self.LOG.info('variable: {} loaded'.format(param_name))

        if len(not_updated_variables) > 0:
            for var in not_updated_variables:
                self.LOG.warning('missing variable: {}'.format(var))

        self.sess.run(param_update_ops, feed_dict=feed_dict)

    
    def _get_parameter_list(self):
        return self.parameters

    def _get_parameters(self):
        parameters = self._get_parameter_list()
        parameter_values = self.sess.run(parameters)

        return_dictionary = OrderedDict((param.name, value) for param, value in zip(parameters, parameter_values))

        return return_dictionary

