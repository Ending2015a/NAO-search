import os
import sys
import time
import logging

import tensorflow as tf

from encoder import encoder
from decoder import decoder


from . import tf_util


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



        if _init_setup_model:
            self._setup_model()


    def fit(self, skills, scores):

    def predict(self, seeds, lambdas=[10, 20, 30]):

    @classmethod
    def load(cls, load_path):
        data, params = cls._load_model(load_path)
        
        model = cls(_init_setup_model=False)
        model.__dict__.update(data)
        model._setup_model()
        
        model._load_parameters(params) #TODO
        
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

        params = self._get_parameters() #TODO
        
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
            self.target_ph = tf.placeholder(shape=(None, ), tf.float32)


            with tf.variable_scope('model'):
            
                # encoder
                self.my_encoder = encoder.Model(self.encoder_ph, self.target_ph, self.params, mode=tf.estimator.ModeKeys.TRAIN, scope='Encoder', reuse=False)

                encoder_outputs = self.my_encoder.encoder_outputs
                encoder_state = self.my_encoder.arch_emb
                encoder_state.set_shape([None, self.decoder_hidden_size])
                encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
                encoder_state = (encoder_state,) * self.decoder_num_layers

                decoder_input_pad = tf.pad(self.decoder_ph, [[0, 0], [1, 0]], "CONSTANT", constant_values=0)
                self.decoder_input = tf.slice(decoder_input_pad, [0, 0], [None, -1])

                #decoder
                self.my_decoder = decoder.Model(encoder_outputs, encoder_state, self.decoder_input, self.decoder_ph, self.params, mode=tf.estimator.ModeKeys.TRAIN, scope='Decoder')

                # get loss
                self.encoder_loss = self.my_encoder.loss
                self.decoder_loss = self.my_decoder.loss

            # compute loss
            with tf.variable_scope('loss'):
                self.decay_loss = self.weight_decay * tf.add_n( [tf.nn.l2_loss(v) for v in tf.trainable_variables()] )
                self.total_loss = self.trade_off * self.encoder_loss + (1 - self.trade_off) * decoder_loss + decay_loss

                tf.summary.scalar('encoder_loss', self.encoder_loss)
                tf.summary.scalar('decoder_loss', self.decoder_loss)
                tf.summary.scalar('decay_loss', self.decay_loss)
                tf.summary.scalar('total_loss', self.total_loss)

            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.constant(self.learning_rate)

            opt = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                gradients, variables = zip(*opt.compute_gradients(total_loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

                # training operations
                self.train_op = opt.apply_gradients(
                        zip(clipped_gradients, variables), global_step=global_step)

            # summary
            with tf.variable_scope('model'):
                self.parameters = tf.trainable_variables()
                if self.full_tensorboard_log:
                    for var in self.parameters:
                        tf.summary.histogram(var.name, var)


            learning_rate = tf.identity(learning_rate, 'learning_rate')
            tf.summary.scalar('learning_rate', learning_rate)




    def _train_step(self)

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
    
    
    def _load_parameters(self, params):
        # TODO
    
    def _get_parameters(self):
        # TODO


