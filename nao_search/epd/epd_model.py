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
                encoder_num_layers:  int = 1,
                encoder_hidden_size: int = 96,
                encoder_emb_size:    int = 32,
                mlp_num_layers:      int = 0,
                mlp_hidden_size:     int = 100,
                mlp_dropout:       float = 0.5,
                decoder_num_layers:  int = 1,
                decoder_hidden_size: int = 96,
                source_length:       int = 60,
                encoder_length:      int = 20,
                decoder_length:      int = 60,
                encoder_dropout:   float = 0.1,
                decoder_dropout:   float = 0.0,
                weight_decay:      float = 1e-4,
                encoder_vocab_size:  int = 21,
                decoder_vocab_size:  int = 21,
                trade_off:         float = 0.8,
                batch_size:          int = 128,
                learning_rate:     float = 0.001,
                optimizer:           str = 'adam',  # DO NOT MODIFY
                start_decay_step:    int = 100,     # X
                decay_steps:         int = 1000,    # X
                decay_factor:      float = 0.9,     # X
                attention:          bool = True,    # DO NOT MODIFY
                max_gradient_norm: float = 5.0,
                beam_width:          int = 0,
                time_major:         bool = True,    # DO NOT MODIFY

                num_cpu:               int = 0,
                full_tensorboard_log: bool = False,
                _init_setup_model:    bool = False): # DO NOT MODIFY

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
        :param encoder_length: (int) The encoder input length. The source sequence will be folded to match this encoder length. (batch_size, encoder_length, source_length//encoder_length)
        :param decoder_length: (int) The decoder output length.
        :param encoder_dropout:
        :param decoder_dropout:
        :param weight_decay:
        :param encoder_vocab_size:
        :param decoder_vocab_size:
        :param trade_off: (float) An alpha value that trade off the portion between the encoder loss and decoder loss. total_loss = trade_off * encoder_loss + (1.0-trade_off) * decoder_loss
        :param batch_size:
        :param learning_rate:
        :param optimizer: (str) ONLY 'adam' IS AVAILABLE.
        :param start_decay_step: (int) The starting step to decay the learning rate. (for 'sgd' optimizer)
        :param decay_steps: (int) The decay steps of the learning rate. (for 'sgd' optimizer)
        :param decay_factor: (float) The decay factor of the learning rate. (for 'sgd' optimizer)
        :param attention:
        :param max_gradient_norm:
        :param beam_width:
        :param time_major:
        :param num_cpu:
        :param full_tensorboard_log: (bool)
        '''
    

        # === hyperparameters ===

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

        # === misc ===

        self.num_cpu = num_cpu
        self.full_tensorboard_log = full_tensorboard_log
        self.graph = None
        self.sess = None
        self.encoder_ph = None
        self.decoder_ph = None
        self.decoder_intput_ph = None
        self.target_ph = None

        # === private ===
        self._param_load_ops = None


        self.LOG = logging.getLogger()

        if _init_setup_model:
            self._setup_model()


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

                def _build_encoder_state(encoder):
                    encoder_outputs = encoder.encoder_outputs
                    encoder_state = encoder.arch_emb
                    encoder_state.set_shape([None, self.decoder_hidden_size])
                    encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
                    encoder_state = (encoder_state,) * self.decoder_num_layers

                    return encoder_outputs, encoder_state
            
                # === train ===
                # encoder
                self.train_encoder = encoder.Model(self.encoder_ph, self.target_ph, self.params, mode=tf.estimator.ModeKeys.TRAIN, scope='Encoder', reuse=False)

                encoder_outputs, encoder_state = _build_encoder_state(self.train_encoder)

                #encoder_outputs = train_encoder.encoder_outputs
                #encoder_state = train_encoder.arch_emb
                #encoder_state.set_shape([None, self.decoder_hidden_size])
                #encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
                #encoder_state = (encoder_state,) * self.decoder_num_layers

                #decoder_input_pad = tf.pad(self.decoder_ph, [[0, 0], [1, 0]], "CONSTANT", constant_values=0)
                #decoder_input = tf.slice(decoder_input_pad, [0, 0], [None, -1])

                # decoder
                self.train_decoder = decoder.Model(encoder_outputs, encoder_state, self.decoder_input_ph, self.decoder_ph, self.params, mode=tf.estimator.ModeKeys.TRAIN, scope='Decoder')

                # get loss
                train_encoder_loss = self.train_encoder.loss # Encoder/square_error
                train_decoder_loss = self.train_decoder.loss # Decoder/cross_entropy

                # set reuse variables
                tf.get_variable_scope().reuse_variables()


                # === eval ===
                
                # encoder
                self.eval_encoder = encoder.Model(self.encoder_ph, self.target_ph, self.params, mode=tf.estimator.ModeKeys.EVAL, scope='Encoder', reuse=True)
                
                encoder_outputs, encoder_state = _build_encoder_state(self.eval_encoder)

                # decoder
                self.eval_decoder = decoder.Model(encoder_outputs, encoder_state, self.decoder_intput_ph, self.decoder_ph, self.params,mode=tf.estimator.ModeKeys.EVAL, scope='Decoder')

                eval_encoder_loss = self.eval_encoder.loss # Encoder/square_error
                eval_decoder_loss = self.eval_decoder.loss # Decoder/cross_entropy

                # === predict ===

                # encoder
                self.pred_encoder = encoder.Model(self.encoder_ph, None, self.params, mode=tf.estimator.ModeKeys.PREDICT, scope='Encoder', reuse=True)
                
                # encode old arch
                encoder_outputs, encoder_state = _build_encoder_state(self.pred_encoder)
                #encoder_outputs = self.eval_encoder.encoder_outputs
                #encoder_state = self.eval_encoder.arch_emb
                #encoder_state.set_shape([None, self.decoder_hidden_size])
                #encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
                #encoder_state = (encoder_state,) * self.decoder_num_layers

                # tmp decoder
                tmp_decoder = decoder.Model(encoder_outputs, encoder_state, None, None, pself.params, mode=tf.estimator.ModeKeys.PREDICT, scope='Decoder')
                
                # predict new arch embedding
                res = self.pred_encoder.infer()
                predict_value = res['predict_value']
                arch_emb = res['arch_emb']
                new_arch_emb = res['new_arch_emb']
                new_arch_outputs = res['new_arch_outputs']

                # decode old arch (evaluate)
                res = tmp_decoder.decode()
                sample_id = res['sample_id']

                encoder_state = new_arch_emb
                encoder_state.set_shape([None, self.decoder_hidden_size])
                encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
                encoder_state = (encoder_state,) * self.decoder_num_layers

                # decode new arch
                self.pred_decoder = decoder.Model(new_arch_outputs, encoder_state, None, None, self.params, mode=tf.estimator.ModeKeys.PREDICT, 'Decoder')
                res = my_decoder.decode()
                new_sample_id = res['sample_id']


            # compute training loss
            with tf.variable_scope('train_loss'):
                encoder_loss = tf.identity(train_encoder_loss, 'encoder_loss')
                decoder_loss = tf.identity(train_decoder_loss, 'decoder_loss')
                decay_loss = self.weight_decay * tf.add_n( [tf.nn.l2_loss(v) for v in tf.trainable_variables()] )
                model_loss = self.trade_off * encoder_loss + (1. - self.trade_off) * decoder_loss
                total_loss = model_loss + decay_loss

                tf.summary.scalar('encoder_loss', encoder_loss)
                tf.summary.scalar('decoder_loss', decoder_loss)
                tf.summary.scalar('decay_loss', decay_loss)
                tf.summary.scalar('model_loss', model_loss)
                tf.summary.scalar('total_loss', total_loss)

                train_encoder_loss = encoder_loss
                train_decoder_loss = decoder_loss
                train_decay_loss = decay_loss
                train_model_loss = model_loss
                train_total_loss = total_loss

                self.train_summary_op = tf.summary.merge_all(scope=tf.get_variable_scope())


            # compute eval loss
            with tf.variable_scope('eval_loss'):
                encoder_loss = tf.identity(eval_encoder_loss, 'encoder_loss')
                decoder_loss = tf.identity(eval_decoder_loss, 'decoder_loss')
                decay_loss = self.weight_decay * tf.add_n( [tf.nn.l2_loss(v) for v in tf.trainable_variables()] )
                model_loss = self.trade_off * encoder_loss + (1. - self.trade_off) * decoder_loss
                total_loss = model_loss + decay_loss

                tf.summary.scalar('encoder_loss', encoder_loss)
                tf.summary.scalar('decoder_loss', decoder_loss)
                tf.summary.scalar('decay_loss', decay_loss)
                tf.summary.scalar('model_loss', model_loss)
                tf.summary.scalar('total_loss', total_loss)

                eval_encoder_loss = encoder_loss
                eval_decoder_loss = decoder_loss
                eval_decay_loss = decay_loss
                eval_model_loss = model_loss
                eval_total_loss = total_loss

                self.eval_summary_op = tf.summary.merge_all(scope=tf.get_variable_scope())


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
 
            self.global_step = global_step
            self.lr = learning_rate

            # training op
            self.train_ops = {
                    'train_op': train_op,
                    'encoder_loss': train_encoder_loss,
                    'decoder_loss': train_decoder_loss,
                    'decay_loss': train_decay_loss,
                    'model_loss': train_model_loss,
                    'total_loss': train_total_loss
                }


            # eval op
            self.eval_ops = {
                    'encoder_loss': eval_encoder_loss,
                    'decoder_loss': eval_decoder_loss,
                    'decay_loss': eval_decay_loss,
                    'model_loss': eval_model_loss,
                    'total_loss': eval_total_loss
                }


            # prediction op
            self.predict_ops = {
                    'arch': decoder_target,
                    'ground_truth_value': encoder_target, # ground truth score
                    'predict_value': predict_value,       # predicted score
                    'sample_id': sample_id,               # old arch (evaluate)
                    'new_sample_id': new_sample_id        # new arch
                }

            # initialize all variables
            tf.global_variables_initializer().run(session=self.sess)

            # summary op
            self.summary_op = tf.summary.merge_all()





    def _data_preprocessing(self, X, y):
        X_ = np.array(X, copy=True, dtype=np.int32)
        y_ = np.array(y, copy=True, dtype=np.float32)

        X_ = X_ + 1

        return X_, y_

    def _data_postprocessing(self, X, y):
        X_ = X - 1
        y_ = y

        return X_, y_


    def _train_step(self, X, X_feed, y, writer, epoch):

        train_ops_name = list(self.train_ops.keys())
        train_ops = [self.train_ops[name] for name in train_ops_name]

        feed_dict = {
                self.encoder_ph: X,
                self.decoder_ph: X,
                self.decoder_input_ph: X_feed,
                self.target_ph: y,
            }

        # get ops list
        ops_name = ['global_step', 'learing_rate'] + train_ops_name
        ops = [self.global_step, self.lr] + train_ops

        if writer is not None:

            # add summary op to ops list
            ops_name = ['summary'] + ops_name
            ops = [self.train_summary_op] + ops
            
            if self.full_tensorboard_log:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                outputs = self.sess.run(
                        ops,
                        feed_dict,
                        options=run_options,
                        run_metadata=run_metadata)

                output_dict = dict(zip(ops_name, outputs))

                writer.add_run_metadata(run_metadata, 'step%d' % output_dict['global_step'])

            else:
                outputs = self.sess.run(
                        ops,
                        feed_dict)

                output_dict = dict(zip(ops_name, outputs))

            writer.add_summary(output_dict['summary'], output_dict['global_step'])

        else:
            outputs = self.sess.run(
                    ops, 
                    feed_dict)

            output_dict = dict(zip(ops_name, outputs))


        return_ops_name = train_ops_name
        return_ops_output = [output_dict[name] for name in return_ops_name]


        return dict(zip(return_ops_name, return_ops_output))



    def _eval_step(self, X, X_feed, y, writer, epoch):
        
        eval_ops_name = list(self.eval_ops.keys())
        eval_ops = [self.eval_ops[name] for name in eval_ops_name]

        feed_dict = {
                self.encoder_ph: X,
                self.decoder_ph: X,
                self.decoder_input_ph: X_feed,
                self.target_ph: y,
            }

        # get ops list
        ops_name = eval_ops_name
        ops = eval_ops

        if writer is not None:

            # add summary op to ops list
            ops_name = ['summary'] + ops_name
            ops = [self.eval_summary_op] + ops
            
            outputs = self.sess.run(
                    ops,
                    feed_dict)

            output_dict = dict(zip(ops_name, outputs))

            writer.add_summary(output_dict['summary'], epoch)

        else:
            outputs = self.sess.run(
                    ops,
                    feed_dict)

            output_dict = dict(zip(ops_name, outputs))


        return_ops_name = eval_ops_name
        return_ops_output = [output_dict[name] for name in return_ops_name]


        return dict(zip(return_ops_name, return_ops_output))


    def learn(self, 
              X, 
              y, 
              epochs:               int = 1000,
              eval_every_n_epochs:  int = 50,
              log_every_n_epochs:   int = 1,
              callback                  = None, 
              log_interval:         int = 1, 
              tb_log_name:          str = "epd", 
              reset_num_timesteps: bool = True):

        SOS = 0

        # preprocessing, since 0 is defined as the start of sequence (SOS), each element in X must add 1
        X_bak, y_bak = self._data_preprocessing(X, y)

        # === check ===
        assert X_bak.ndim == 2, ValueError("The dimension of input 'X' must equal to 2")
        assert np.all(X_bak >= 0 and X_bak < encoder_vocab_size), ValueError("Each element of input 'X' must be in the range of the vocab size")
        assert np.all(y_bak >= 0.0 and y_bak <= 1.0), ValueError("Each element of input 'y' must be normalized between 0 ~ 1")
        assert len(X_bak) == len(y_bak), ValueError("The size of input 'X' and 'y' must be equal")
        assert len(X_bak) > 10, "The number of training samples 'X' must be greater than 10"

        # === initialize global step ===
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

            # split to training set and eval set
            X_train = X_shuffle[:train_num]
            y_train = y_shuffle[:train_num]

            X_eval = X_shuffle[train_num:]
            y_eval = y_shuffle[train_num:]

            # for training LSTM, insert 0 in the beginning and crop the last element in each row
            X_train_feed = np.pad(np.copy(X_train), ((0, 0), (1, 0)), 'constant', constant_values=SOS)[..., :-1]
            X_eval_feed = np.pad(np.copy(X_eval), ((0, 0), (1, 0)), 'constant', constant_values=SOS)[..., :-1]


            # total training/eval steps of each epoch
            n_updates = train_num // self.batch_size + 1
            n_eval = eval_num // self.batch_size + 1
            index_shuffle = [idx for idx in range(len(train_num))]

            t_first_start = time.time()

            for epoch in range(1, epochs+1):

                # shuffle 
                np.random.shuffle(index_shuffle)
                
                X_ = X_train[index_shuffle]
                y_ = y_train[index_shuffle]
                X_feed = X_train_feed[index_shuffle]

                # === some containers ===
                epoch_train_loss_vals = []
                epoch_eval_loss_vals = []

                # === start epoch ===

                # === start training ===
                t_start = time.time()

                for update in range(1, n_updates+1):

                    start_ind = (update-1) * self.batch_size
                    end_ind = start_ind + self.batch_size

                    # clamp
                    end_ind = end_ind if end_ind <= train_num else train_num

                    # get slice
                    X_slice = X_[start_ind:end_ind]
                    y_slice = y_[start_ind:end_ind]
                    X_feed_slice = X_feed[start_ind:end_ind]

                    # train step
                    epoch_train_loss_vals.append(
                            self._train_step(X_slice, y_slice, X_feed_slice, writer=writer, epoch=epoch-1)
                            )                

                t_end = time.time()

                t_training_time = t_end - t_start

                # === start eval ===
                if epoch % eval_every_n_epochs == 0:

                    t_eval_start = t_end

                    for update in range(1, n_eval+1):

                        start_ind = (update-1) * self.batch_size
                        end_ind = start_ind + self.batch_size

                        # clamp
                        end_ind = end_ind if end_ind <= eval_num else eval_num

                        # get slice
                        X_slice = X_eval[start_ind:end_ind]
                        y_slice = y_eval[start_ind:end_ind]
                        X_feed_slice = X_eval_feed[start_ind:end_ind]

                        # eval step
                        epoch_eval_loss_vals.append(
                                self._eval_step(X_slice, y_slice, X_feed_slice, writer=writer, epoch=epoch-1)
                                )
                    
                    t_eval_end = time.time()
                    t_eval_time = t_eval_end - t_eval_start

                    t_end = t_eval_end

                else:
                    t_eval_time = None

                
                t_epoch_time = t_end - t_start


                if epoch % log_every_n_epochs == 0 or epoch % eval_every_n_epochs == 0:


                    avg_loss = {
                            'encoder_loss': 0.0,
                            'decoder_loss': 0.0,
                            'decay_loss': 0.0,
                            'model_loss': 0.0,
                            'total_loss': 0.0
                        }

                    # sum
                    for d in epoch_train_loss_vals:
                        for name in avg_loss.keys():
                            avg_loss[name] += d[name]

                    # average
                    total_vals_num = len(epoch_train_loss_vals)
                    for name in avg_loss.keys():
                        avg_loss[name] /= total_vals_num


                    log_kvpair = {
                            'epochs': '{}/{}'.format(epoch, epochs+1),
                            'epoch_time': t_epoch_time,
                            'training_time': t_training_time,
                            'training_encoder_loss': avg_loss['encoder_loss']
                            'training_decoder_loss': avg_loss['decoder_loss']
                            'training_decay_loss': avg_loss['decay_loss']
                            'training_model_loss': avg_loss['model_loss']
                            'training_total_loss': avg_loss['total_loss']
                        }

                    if epoch % eval_every_n_epochs == 0:

                        
                        avg_loss = {
                                'encoder_loss': 0.0,
                                'decoder_loss': 0.0,
                                'decay_loss': 0.0,
                                'model_loss': 0.0,
                                'total_loss': 0.0
                            }

                        # sum
                        for d in epoch_eval_loss_vals:
                            for name in avg_loss.keys():
                                avg_loss[name] += d[name]

                        # average
                        total_vals_num = len(epoch_eval_loss_vals)
                        for name in avg_loss.keys():
                            avg_loss[name] /= total_vals_num

                        log_kvpair['eval'] = True
                        log_kvpair['eval_time'] = t_eval_time
                        log_kvpair['eval_encoder_loss'] = avg_loss['encoder_loss']
                        log_kvpair['eval_decoder_loss'] = avg_loss['decoder_loss']
                        log_kvpair['eval_decay_loss'] = avg_loss['decay_loss']
                        log_kvpair['eval_model_loss'] = avg_loss['model_loss']
                        log_kvpair['eval_total_loss'] = avg_loss['total_loss']

                    
                    log_kvpair #TODO: output to log


            if callback is not None:
                if callback(locals(), globals()) is False:
                    break



    def predict(self, seeds, lambdas=[10, 20, 30]):
        pass #TODO


    @classmethod
    def load(cls, load_path):
        data, params = cls._load_model(load_path)
        
        model = cls(_init_setup_model=False)
        model.__dict__.update(data)
        model._setup_model()
        
        model._load_parameters(params)
        
        return model


    def save(self, save_path):
        
        data = self._build_param_dict()

        params = self._get_parameters()
        
        self._save_model(save_path, data, params)


    def _initialize_global_step(reset_num_timesteps):

        # if reset_num_timesteps == True: assign 0 to global_step
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

