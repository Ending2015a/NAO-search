import os
import sys
import time
import logging
import multiprocessing

import cloudpickle

import numpy as np
import tensorflow as tf

from copy import deepcopy
from collections import OrderedDict

from .encoder import encoder
from .decoder import decoder

from . import tf_util
from .tf_util import TensorboardWriter

from nao_search.common.logger import Logger
from nao_search.common.utils import pairwise_accuracy

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
                batch_size:          int = 50,
                learning_rate:     float = 0.001,
                optimizer:           str = 'adam',  # DO NOT MODIFY
                start_decay_step:    int = 100,     # X
                decay_steps:         int = 1000,    # X
                decay_factor:      float = 0.9,     # X
                attention:          bool = True,    # DO NOT MODIFY
                max_gradient_norm: float = 5.0,
                beam_width:          int = 0,
                time_major:         bool = True,    # DO NOT MODIFY
                predict_beam_width:  int = 0,

                num_cpu:               int = 0,
                tensorboard_log:       str = None,
                full_tensorboard_log: bool = False,
                input_processing:     bool = True,
                _init_setup_model:    bool = True): # DO NOT MODIFY

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
        :param tensorboard_log: (str) log path
        :param full_tensorboard_log: (bool)
        :param input_processing: (bool) automatically preprocess/postprocess input. Set this param to True will automatically convert your input sequences from 0-based to 1-based (since 0 = <SOS>), also your scores will be reversed, since the native Neural Architecture Search is using error rates as their scores of the sequences.
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
        self.predict_beam_width = predict_beam_width

        # === misc ===

        self.num_cpu = num_cpu
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log
        self.input_processing = input_processing
        self.graph = None
        self.sess = None
        self.encoder_input_ph = None
        self.decoder_target_ph = None
        self.decoder_intput_ph = None
        self.encoder_target_ph = None

        # === private ===
        self._param_load_ops = None
        self._SOS = 0

        self.LOG = Logger()

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
            'time_major':          self.time_major,
            'predict_beam_width':  self.predict_beam_width
        }
        
        return data


    def _setup_model(self):

        # === DEBUG LOG ===
        self.LOG.set_header('DEBUG_LOG: Setup Model')
        # =================

        self.params = self._build_param_dict()

        # get cpu count
        n_cpu = self.num_cpu if self.num_cpu > 0 else multiprocessing.cpu_count()
        self.num_cpu = n_cpu

        # === DEBUG LOG ===
        self.LOG.switch_group('Hyperparameters')
        for key, value in self.params.items():
            fmt=None
            if isinstance(value, float):
                fmt = '{key}: {value:.6f}'
            self.LOG.add_pair(key, value, fmt)
        # =================

        # create graph
        self.graph = tf.Graph()
        with self.graph.as_default():

            # create session
            self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

            # placeholder
            self.encoder_input_ph = tf.placeholder(shape=(None, self.source_length), dtype=tf.int32, name='encoder_input_ph')
            self.decoder_target_ph = tf.placeholder(shape=(None, self.source_length), dtype=tf.int32, name='decoder_target_ph')
            self.decoder_input_ph = tf.placeholder(shape=(None, self.source_length), dtype=tf.int32, name='decoder_input_ph')

            self.encoder_target_ph = tf.placeholder(shape=(None, ), dtype=tf.float32, name='encoder_target_ph')
            self.predict_lambda_ph = tf.placeholder(shape=(1,), dtype=tf.float32, name='predict_lambda_ph')


            reshaped_target_ph = tf.reshape(self.encoder_target_ph, shape=(-1, 1))

            # dummy decoder input for predicting
            dummy_batch_size = tf.shape(self.encoder_input_ph)[0]
            dummy_decoder_input = tf.fill(value=self._SOS, dims=(dummy_batch_size, 1), name='dummy_decoder_input')

            # === DEBUG LOG ===
            self.LOG.switch_group('placeholder')
            
            self.LOG.add_pair('{} shape'.format(self.encoder_input_ph.name),   self.encoder_input_ph.get_shape())
            self.LOG.add_pair('{} shape'.format(self.decoder_target_ph.name),  self.decoder_target_ph.get_shape())
            self.LOG.add_pair('{} shape'.format(self.decoder_input_ph.name),   self.decoder_input_ph.get_shape())
            self.LOG.add_pair('{} shape'.format(self.encoder_target_ph.name),  self.encoder_target_ph.get_shape())
            self.LOG.add_pair('{} shape'.format(self.predict_lambda_ph.name),  self.predict_lambda_ph.get_shape())
            self.LOG.add_pair('{} shape'.format(dummy_decoder_input.name),     dummy_decoder_input.get_shape())
            # =================

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
                self.train_encoder = encoder.Model(self.encoder_input_ph, reshaped_target_ph, self.params, mode=tf.estimator.ModeKeys.TRAIN, scope='Encoder', reuse=False)

                encoder_outputs, encoder_state = _build_encoder_state(self.train_encoder)

                #encoder_outputs = train_encoder.encoder_outputs
                #encoder_state = train_encoder.arch_emb
                #encoder_state.set_shape([None, self.decoder_hidden_size])
                #encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
                #encoder_state = (encoder_state,) * self.decoder_num_layers

                #decoder_input_pad = tf.pad(self.decoder_ph, [[0, 0], [1, 0]], "CONSTANT", constant_values=0)
                #decoder_input = tf.slice(decoder_input_pad, [0, 0], [None, -1])

                # decoder
                self.train_decoder = decoder.Model(encoder_outputs, encoder_state, self.decoder_input_ph, self.decoder_target_ph, self.params, mode=tf.estimator.ModeKeys.TRAIN, scope='Decoder')

                # get loss
                train_encoder_loss = self.train_encoder.loss # Encoder/square_error
                train_decoder_loss = self.train_decoder.loss # Decoder/cross_entropy

                debug_print = [ tf.print( tf.shape(self.train_decoder.target), output_stream=sys.stdout), 
                                tf.print( tf.shape(self.train_decoder.logits), output_stream=sys.stdout),
                                tf.print( self.train_decoder.loss, output_stream=sys.stdout)]

                # set reuse variables
                tf.get_variable_scope().reuse_variables()


                # === eval ===
                
                # encoder
                self.eval_encoder = encoder.Model(self.encoder_input_ph, reshaped_target_ph, self.params, mode=tf.estimator.ModeKeys.EVAL, scope='Encoder', reuse=True)
                
                encoder_outputs, encoder_state = _build_encoder_state(self.eval_encoder)

                # decoder
                self.eval_decoder = decoder.Model(encoder_outputs, encoder_state, self.decoder_input_ph, self.decoder_target_ph, self.params,mode=tf.estimator.ModeKeys.EVAL, scope='Decoder')

                eval_encoder_loss = self.eval_encoder.loss # Encoder/square_error
                eval_decoder_loss = self.eval_decoder.loss # Decoder/cross_entropy

                # === predict ===

                # encoder
                self.pred_encoder = encoder.Model(self.encoder_input_ph, None, self.params, mode=tf.estimator.ModeKeys.PREDICT, scope='Encoder', reuse=True)
                
                # encode old arch
                encoder_outputs, encoder_state = _build_encoder_state(self.pred_encoder)
                #encoder_outputs = self.eval_encoder.encoder_outputs
                #encoder_state = self.eval_encoder.arch_emb
                #encoder_state.set_shape([None, self.decoder_hidden_size])
                #encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
                #encoder_state = (encoder_state,) * self.decoder_num_layers

                # tmp decoder
                tmp_decoder = decoder.Model(encoder_outputs, encoder_state, dummy_decoder_input, None, self.params, mode=tf.estimator.ModeKeys.PREDICT, scope='Decoder')
                
                # predict new arch embedding
                res = self.pred_encoder.infer(self.predict_lambda_ph)
                predict_value = res['predict_value']
                arch_emb = res['arch_emb']
                new_arch_emb = res['new_arch_emb']
                new_arch_outputs = res['new_arch_outputs']


                # decode old arch (evaluate)
                res = tmp_decoder.decode()
                predict_sample_id = tf.reshape(res['sample_id'], shape=(-1, self.decoder_length))

                encoder_state = new_arch_emb
                encoder_state.set_shape([None, self.decoder_hidden_size])
                encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
                encoder_state = (encoder_state,) * self.decoder_num_layers

                # decode new arch
                self.pred_decoder = decoder.Model(new_arch_outputs, encoder_state, dummy_decoder_input, None, self.params, mode=tf.estimator.ModeKeys.PREDICT, scope='Decoder')
                res = self.pred_decoder.decode()
                predict_new_sample_id = tf.reshape(res['sample_id'], shape=(-1, self.decoder_length))


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

                self.train_summary_op = tf.summary.merge_all(scope=tf.get_variable_scope().name)



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

                self.eval_summary_op = tf.summary.merge_all(scope=tf.get_variable_scope().name)



            # global step
            global_step = tf.train.get_or_create_global_step()
            # learning rate
            learning_rate = tf.constant(self.learning_rate)

            # optimizer
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                gradients, variables = zip(*opt.compute_gradients(train_total_loss))
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
                learning_rate = opt._lr
                tf.summary.scalar('learning_rate', learning_rate)
                tf.summary.scalar('global_step', global_step)

                tf.summary.merge_all(scope=tf.get_variable_scope().name)
 
            self.global_step = global_step
            self.lr = learning_rate

            # training op
            self.train_ops = {
                    'train_op': train_op,
                    #'debug_print': debug_print,
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
                    'ground_truth_seq': self.decoder_target_ph,
                    'ground_truth_value': self.encoder_target_ph, # ground truth score
                    'predict_value': predict_value,               # predicted score
                    'predict_old_seq': predict_sample_id,         # old arch (evaluate)
                    'predict_new_seq': predict_new_sample_id      # new arch
                }

            # initialize all variables
            tf.global_variables_initializer().run(session=self.sess)

            # summary op
            self.summary_op = tf.summary.merge_all()
    
            # === DEBUG LOG ===
            self.LOG.switch_group('train')

            self.LOG.add_pair('encoder_loss shape', train_encoder_loss.get_shape())
            self.LOG.add_pair('decoder_loss shape', train_decoder_loss.get_shape())
            self.LOG.add_pair('decay_loss shape',   train_decay_loss.get_shape())
            self.LOG.add_pair('model_loss shape',   train_model_loss.get_shape())
            self.LOG.add_pair('total_loss shape',   train_total_loss.get_shape())

            self.LOG.switch_group('eval')

            self.LOG.add_pair('encoder_loss shape', eval_encoder_loss.get_shape())
            self.LOG.add_pair('decoder_loss shape', eval_decoder_loss.get_shape())
            self.LOG.add_pair('decay_loss shape',   eval_decay_loss.get_shape())
            self.LOG.add_pair('model_loss shape',   eval_model_loss.get_shape())
            self.LOG.add_pair('total_loss shape',   eval_total_loss.get_shape())

            self.LOG.switch_group('predict')

            self.LOG.add_pair('ground_truth_seq shape',   self.decoder_target_ph.get_shape())
            self.LOG.add_pair('ground_truth_value shape', self.encoder_target_ph.get_shape())
            self.LOG.add_pair('predict_value shape',      predict_value.get_shape())
            self.LOG.add_pair('predict_old_seq shape',    predict_sample_id.get_shape())
            self.LOG.add_pair('predict_new_seq shape',    predict_new_sample_id.get_shape())
            # =================

        self.LOG.dump_to_log(level=logging.DEBUG)


    def _data_preprocessing(self, X=None, y=None):

        # processing
        if not X is None:
            X_ = np.array(X, copy=True, dtype=np.int32)
            X_ = X_ + 1

        if not y is None:    
            y_ = np.array(y, copy=True, dtype=np.float32)
            y_ = 1. - y_

        # return
        if (not X is None) and (not y is None):
            return X_, y_
        elif not X is None:
            return X_
        elif not y is None:
            return y_

        return

    def _data_postprocessing(self, X=None, y=None):
        
        # processing
        if not X is None:
            X_ = X - 1

        if not y is None:
            y_ = 1. - y

        # return
        if (not X is None) and (not y is None):
            return X_, y_
        elif not X is None:
            return X_
        elif not y is None:
            return y_

        return 


    def _train_step(self, X, y, X_feed, writer, epoch):

        train_ops_name = list(self.train_ops.keys())
        train_ops = [self.train_ops[name] for name in train_ops_name]

        feed_dict = {
                self.encoder_input_ph: X,
                self.decoder_target_ph: X,
                self.decoder_input_ph: X_feed,
                self.encoder_target_ph: y,
            }


        # get ops list
        ops_name = ['global_step', 'learning_rate'] + train_ops_name
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

        return_ops_name = ops_name
        return_ops_output = [output_dict[name] for name in return_ops_name]


        return dict(zip(return_ops_name, return_ops_output))



    def _eval_step(self, X, y, X_feed, writer=None, epoch=None):
        
        eval_ops_name = list(self.eval_ops.keys())
        eval_ops = [self.eval_ops[name] for name in eval_ops_name]

        feed_dict = {
                self.encoder_input_ph: X,
                self.decoder_target_ph: X,
                self.decoder_input_ph: X_feed,
                self.encoder_target_ph: y,
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


    def _predict_step(self, X, ld, y=None):

        feed_dict = {
                self.encoder_input_ph: X,
                self.decoder_target_ph: X,
                self.predict_lambda_ph: [ld]
            }


        pred_ops_name = list(self.predict_ops.keys())

        # if ground truth value provided
        if y is not None:
            feed_dict[self.encoder_target_ph] = y
        else:
            if 'ground_truth_value' in pred_ops_name:
                pred_ops_name.remove('ground_truth_value')

        pred_ops = [self.predict_ops[name] for name in pred_ops_name]

        # get ops list
        ops_name = pred_ops_name
        ops = pred_ops

        # predict step
        outputs = self.sess.run(
                ops,
                feed_dict)

        output_dict = dict(zip(ops_name, outputs))


        return_ops_name = pred_ops_name
        return_ops_output = [output_dict[name] for name in return_ops_name]


        return dict(zip(return_ops_name, return_ops_output))

    def learn(self, 
              X, 
              y, 
              epochs:               int = 1000,
              callback                  = None,
              eval_interval:        int = 50,
              log_interval:         int = 1, 
              tb_log_name:          str = "epd",
              input_processing          = None,
              reset_num_timesteps: bool = True):

        '''
        :param X: sequences, (1-based), since 0 = <SOS>, if your sequences are 0-based, your should set input_processing to True to automatically convert your sequences to 1-based.
        :param y: scores, (error rate), if your scores do not representing the error rates of the sequences, you should set input_processing to True to automatically reverse your scores.
        :param epochs: (int) training epochs
        :param callback: callback function, signature: callback(local, global)
        :param eval_interval: (int) do evaluation every n epochs
        :param log_interval: (int) log every n epochs
        :param input_processing: (bool) this will overwrite default input_processing setting
        :param reset_num_timesteps: (bool) reset global_step
        '''
        
        input_processing = input_processing if not input_processing is None else self.input_processing

        # preprocessing, since 0 is defined as the start of sequence (SOS), each element in X must add 1
        if input_processing:
            X_bak, y_bak = self._data_preprocessing(X, y)
        else:
            X_bak = np.array(X, copy=True, dtype=np.int32)
            y_bak = np.array(y, copy=True, dtype=np.float32)

        # === check ===
        assert X_bak.ndim == 2, ValueError("The dimension of input 'X' must equal to 2")
        assert X_bak.shape[-1] == self.source_length, ValueError("The last dimension of input 'X' must match to 'source_length'")
        assert np.all( (X_bak >= 0) & (X_bak <= self.encoder_vocab_size) ), ValueError("Each element of input 'X' must be in the range of the vocab size")
        assert np.all( (y_bak >= 0.0) & (y_bak <= 1.0) ), ValueError("Each element of input 'y' must be normalized between 0 ~ 1")
        assert len(X_bak) == len(y_bak), ValueError("The size of input 'X' and 'y' must be equal")
        assert len(X_bak) > 10, "The number of training samples 'X' must be greater than 10"

        # === initialize global step ===
        new_tb_log = self._initialize_global_step(reset_num_timesteps)


        with TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) as writer:

            # calculate the total number of items
            num_items = len(y_bak)

            # shuffled indices
            index_shuffle = [idx for idx in range(num_items)]
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
            X_train_feed = np.pad(np.copy(X_train), ((0, 0), (1, 0)), 'constant', constant_values=self._SOS)[..., :-1]
            X_eval_feed = np.pad(np.copy(X_eval), ((0, 0), (1, 0)), 'constant', constant_values=self._SOS)[..., :-1]


            # total training/eval steps of each epoch
            n_updates = train_num // self.batch_size
            n_eval = eval_num // self.batch_size

            if train_num % self.batch_size > 0:
                n_updates += 1

            if eval_num % self.batch_size > 0:
                n_eval += 1
 
            index_shuffle = [idx for idx in range(train_num)]

            
            # === DEBUG LOG ===
            self.LOG.set_header('DEBUG LOG: Learn')
            self.LOG.switch_group('Training info')
            self.LOG.add_pair('input_processing',         input_processing)
            self.LOG.add_pair('batch_size',               self.batch_size)
            self.LOG.add_pair('eval_interval',            eval_interval)
            self.LOG.add_pair('training steps per epoch', n_updates)
            self.LOG.add_pair('eval steps per epoch',     n_eval)

            self.LOG.switch_group('Log info')
            self.LOG.add_pair('log_interval',         log_interval)
            self.LOG.add_pair('tensorboard logdir',   self.tensorboard_log)
            self.LOG.add_pair('tensorboard log name', tb_log_name)

            self.LOG.switch_group('Dataset')
            self.LOG.add_pair('training samples', train_num)
            self.LOG.add_pair('eval samples',     eval_num)

            self.LOG.switch_group('Dataset shape')
            self.LOG.add_pair('train X',      X_train.shape)
            self.LOG.add_pair('train y',      y_train.shape)
            self.LOG.add_pair('train X_feed', X_train_feed.shape)
            self.LOG.add_pair('eval X',       X_eval.shape)
            self.LOG.add_pair('eval y',       y_eval.shape)
            self.LOG.add_pair('eval X_feed',  X_eval_feed.shape)

            self.LOG.dump_to_log(level=logging.DEBUG)
            # =================



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
                            [self._train_step(X_slice, y_slice, X_feed_slice, writer=writer, epoch=epoch-1), end_ind - start_ind]
                            )

                t_end = time.time()

                t_training_time = t_end - t_start

                # === start eval ===
                if epoch % eval_interval == 0:

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
                                    [self._eval_step(X_slice, y_slice, X_feed_slice, writer=writer, epoch=epoch-1), 
                                        end_ind - start_ind,
                                        self._predict_step(X_slice, ld=1., y=y_slice)]
                                )
                    
                    t_eval_end = time.time()
                    t_eval_time = t_eval_end - t_eval_start

                    t_end = t_eval_end

                else:
                    t_eval_time = None

                
                t_epoch_time = t_end - t_start


                # logging
                if epoch % log_interval == 0 or epoch % eval_interval == 0:


                    avg_loss = {
                            'encoder_loss': 0.0,
                            'decoder_loss': 0.0,
                            'decay_loss': 0.0,
                            'model_loss': 0.0,
                            'total_loss': 0.0
                        }

                    # sum
                    for d, batch_sz in epoch_train_loss_vals:
                        for name in avg_loss.keys():
                            avg_loss[name] += d[name] * batch_sz

                    # average
                    for name in avg_loss.keys():
                        avg_loss[name] = avg_loss[name] / train_num

                    # get timestep
                    last_timestep = epoch_train_loss_vals[-1][0]['global_step']


                    # === LOG ===
                    self.LOG.set_header('Epoch {}/{}'.format(epoch, epochs))
                    self.LOG.switch_group()

                    fmt = '{key}: {value:.6f}'

                    self.LOG.add_pair('total_timestep', last_timestep)
                    self.LOG.add_pair('epoch_time', t_epoch_time, fmt=fmt + ' sec')

                    # === train info ===
                    self.LOG.switch_group('Train')

                    self.LOG.add_pair('elapsed_time', t_training_time, fmt=fmt + ' sec')
                    self.LOG.add_pair('encoder_loss (mse)', avg_loss['encoder_loss'], fmt)
                    self.LOG.add_pair('decoder_loss  (ce)', avg_loss['decoder_loss'], fmt)
                    self.LOG.add_pair('decay_loss', avg_loss['decay_loss'], fmt)
                    self.LOG.add_pair('model_loss', avg_loss['model_loss'], fmt)
                    self.LOG.add_pair('total_loss', avg_loss['total_loss'], fmt)

                    if epoch % eval_interval == 0:

                        
                        avg_loss = {
                                'encoder_loss': 0.0,
                                'decoder_loss': 0.0,
                                'decay_loss': 0.0,
                                'model_loss': 0.0,
                                'total_loss': 0.0
                            }

                        ground_truth_value = []
                        predict_value = []

                        # sum
                        for d, batch_sz, _ in epoch_eval_loss_vals:
                            for name in avg_loss.keys():
                                avg_loss[name] += d[name] * batch_sz

                        # average
                        for name in avg_loss.keys():
                            avg_loss[name]  = avg_loss[name] / eval_num

                        # accuracy
                        for _, _, pred in epoch_eval_loss_vals:
                            # flattening to 1D list
                            ground_truth_value.extend(pred['ground_truth_value'].flatten().tolist())
                            predict_value.extend(pred['predict_value'].flatten().tolist())

                        # converting list to np.array
                        ground_truth_value = np.array(ground_truth_value)
                        predict_value = np.array(predict_value)

                        # calculate pairwise accuracy
                        mse = ((predict_value - ground_truth_value)**2).mean(axis=0)
                        pairwise_acc = pairwise_accuracy(ground_truth_value, predict_value)

                        # === eval info ===
                        self.LOG.switch_group('Eval')

                        self.LOG.add_pair('elapsed_time', t_eval_time, fmt=fmt + ' sec')
                        self.LOG.add_pair('encoder_loss (mse)', avg_loss['encoder_loss'], fmt)
                        self.LOG.add_pair('decoder_loss  (ce)', avg_loss['decoder_loss'], fmt)
                        self.LOG.add_pair('decay_loss', avg_loss['decay_loss'], fmt)
                        self.LOG.add_pair('model_loss', avg_loss['model_loss'], fmt)
                        self.LOG.add_pair('total_loss', avg_loss['total_loss'], fmt)

                        self.LOG.switch_group('Predict')
                        self.LOG.add_pair('pairwise_accuracy ', pairwise_acc, fmt)
                        self.LOG.add_pair('value_loss   (mse)', mse, fmt)


                    # output to log
                    self.LOG.dump_to_log()


                if callback is not None:
                    if callback(locals(), globals()) is False:
                        break




    def eval(self, X, 
                   y,
                   input_processing=None):

        input_processing = input_processing if not input_processing is None else self.input_processing

        if input_processing:
            X_bak, y_bak = self._data_preprocessing(X, y)
        else:
            X_bak = np.array(X, copy=True, dtype=np.int32)
            y_bak = np.array(y, copy=True, dtype=np.float32)


        # === check ===
        assert X_bak.ndim == 2, ValueError("The dimension of input 'X' must equal to 2")
        assert X_bak.shape[-1] == self.source_length, ValueError("The last dimension of input 'X' must match to 'source_length'")
        assert np.all( (X_bak >= 0) & (X_bak <= self.encoder_vocab_size) ), ValueError("Each element of input 'X' must be in the range of the vocab size")
        assert np.all( (y_bak >= 0.0) & (y_bak <= 1.0) ), ValueError("Each element of input 'y' must be normalized between 0 ~ 1")
        assert len(X_bak) == len(y_bak), ValueError("The size of input 'X' and 'y' must be equal")
        # =============

        num_items = len(y_bak)
        eval_num = num_items

        n_eval = eval_num // self.batch_size
        
        if eval_num % self.batch_size > 0:
            n_eval += 1

        # === DEBUG LOG ===
        self.LOG.set_header('DEBUG LOG: Eval')
        self.LOG.switch_group('Eval info')
        self.LOG.add_pair('batch_size', self.batch_size)
        self.LOG.add_pair('eval steps', n_eval)
        self.LOG.dump_to_log(level=logging.DEBUG)
        # =================

        X_eval = X_bak
        y_eval = y_bak
        X_eval_feed = np.pad(np.copy(X_eval), ((0, 0), (1, 0)), 'constant', constant_values=self._SOS)[..., :-1]

        # === some containers ===
        eval_loss_vals = []


        # === start evaluate ===
        t_eval_start = time.time()

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
            eval_loss_vals.append(
                    [self._eval_step(X_slice, y_slice, X_feed_slice), end_ind - start_ind]
                    )

        t_eval_end = time.time()
        t_eval_time = t_eval_end - t_eval_start

        avg_loss = {
                'encoder_loss': 0.0,
                'decoder_loss': 0.0,
                'decay_loss': 0.0,
                'model_loss': 0.0,
                'total_loss': 0.0
            }

        # sum & average
        for d, batch_sz in eval_loss_vals:
            for name in avg_loss.keys():
                avg_loss[name] += d[name] * (batch_sz / eval_num)  # alleviate numerical error

        
        # === eval info ===
        fmt = '{key}: {value:.6f}'
        self.LOG.set_header('Eval results')

        self.LOG.add_pair('elapsed_time', t_eval_time, fmt=fmt + ' sec')
        self.LOG.add_pair('encoder_loss (mse)', avg_loss['encoder_loss'], fmt)
        self.LOG.add_pair('decoder_loss  (ce)', avg_loss['decoder_loss'], fmt)
        self.LOG.add_pair('decay_loss', avg_loss['decay_loss'], fmt)
        self.LOG.add_pair('model_loss', avg_loss['model_loss'], fmt)
        self.LOG.add_pair('total_loss', avg_loss['total_loss'], fmt)

        self.LOG.dump_to_log()



    def predict(self, seeds, 
                      lambdas=[10, 20, 30],
                      input_processing=None):
 

        input_processing = input_processing if not input_processing is None else self.input_processing

        # === preprocess ===
        if not isinstance(lambdas, list):
            lambdas = list(lambdas)

        lambdas = np.array(lambdas, dtype=np.float32).flatten()
        
        if input_processing:
            seeds_bak = self._data_preprocessing(X=seeds)
        else:
            seeds_bak = np.array(seeds, copy=True, dtype=np.int32)
        # =========================


        # === check ===
        assert seeds_bak.ndim == 2, ValueError("The dimension of input 'seeds' must be 2")
        assert seeds_bak.shape[-1], ValueError("The last dimension of input 'seeds' must match to source_length")
        assert seeds_bak.shape[-1] == self.source_length, ValueError("The last dimension of input 'X' must match to 'source_length'")
        assert np.all( (seeds_bak >= 0) & (seeds_bak <= self.encoder_vocab_size) ), ValueError("Each element of input 'X' must be in the range of the vocab size")
        # =============


        # calculate the total number of items
        num_items = len(seeds_bak)
        pred_num = num_items


        # predicting steps of for each epoch
        n_pred = pred_num // self.batch_size

        if pred_num % self.batch_size > 0:
            n_pred += 1

        # === DEBUG LOG ===
        self.LOG.set_header('DEBUG LOG: Predict')
        self.LOG.switch_group('Predicting info')
        self.LOG.add_pair('batch_size', self.batch_size)
        self.LOG.add_pair('total pred steps', n_pred)
        self.LOG.add_pair('lambdas', lambdas.tolist())
        self.LOG.add_pair('seeds num', len(seeds_bak))
        self.LOG.add_pair('expected results num', len(seeds_bak) * len(lambdas))
        self.LOG.dump_to_log(level=logging.DEBUG)
        # =================

        # === some containers ===
        results_vals = []

        t_start = time.time()

        for _lambda in lambdas.tolist():


            for pred in range(1, n_pred+1):

                start_ind = (pred-1) * self.batch_size
                end_ind = start_ind + self.batch_size

                # clamp
                end_ind = end_ind if end_ind <= pred_num else pred_num

                # get slice
                seeds_slice = seeds_bak[start_ind:end_ind]

                results_vals.append(
                        self._predict_step(seeds_slice, _lambda)
                        )

        t_end = time.time()


        result_list = {
                'ground_truth_seq': [],
                'predict_value': [],
                'predict_old_seq': [],
                'predict_new_seq': []
                }


        # for each batch
        for results_batch in results_vals:

            results_batch = zip(results_batch['ground_truth_seq'],
                                results_batch['predict_old_seq'],
                                results_batch['predict_new_seq'],
                                results_batch['predict_value'])

            # for each item
            for result in results_batch:
                
                ground_truth_seq = result[0].flatten()
                predict_old_seq = result[1].flatten()
                predict_new_seq = result[2].flatten()
                predict_value = np.asscalar(result[3])

                
                if input_processing:
                    ground_truth_seq = self._data_postprocessing(X=ground_truth_seq)
                    predict_old_seq  = self._data_postprocessing(X=predict_old_seq)
                    predict_new_seq  = self._data_postprocessing(X=predict_new_seq)
                    predict_value    = self._data_postprocessing(y=predict_value)

                result_list['ground_truth_seq'].append(ground_truth_seq.tolist())
                result_list['predict_old_seq'].append(predict_old_seq.tolist())
                result_list['predict_new_seq'].append(predict_new_seq.tolist())
                result_list['predict_value'].append(predict_value)



        _info = {
                'ground_truth_seq': result_list['ground_truth_seq'],
                'predict_old_seq': result_list['predict_old_seq'],
                'predict_value': result_list['predict_value']
                }

        return result_list['predict_new_seq'], _info


    
    def predict_scores(self, seeds,
                            input_processing=None):

        _, _info = self.predict(seeds,
                                lambdas=[0.],
                                input_processing=input_processing)

        return _info['predict_value']



    @classmethod
    def load(cls, load_path,
                  num_cpu:               int = 0,
                  tensorboard_log:       str = None,
                  full_tensorboard_log: bool = False,
                  input_processing:     bool = True):

        data, params = cls._load_model(load_path)
        
        model = cls(
                num_cpu=num_cpu,
                tensorboard_log=tensorboard_log,
                full_tensorboard_log=full_tensorboard_log,
                input_processing=input_processing,
                _init_setup_model=False)

        model.__dict__.update(data)
        model._setup_model()
        
        model._load_parameters(params)
        
        return model


    def save(self, save_path):
        
        data = self._build_param_dict()

        params = self._get_parameters()
        
        self._save_model(save_path, data, params)


    def _initialize_global_step(self, reset_num_timesteps):

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

        # === LOG ===
        self.LOG.set_header('DEBUG LOG: Load model')
        self.LOG.switch_group('loaded variables')
        # ===========

        for param_name, param_value in params.items():
            placeholder, assign_op = self._param_load_ops[param_name]
            feed_dict[placeholder] = param_value

            param_update_ops.append(assign_op)

            not_updated_variables.remove(param_name)

            # === LOG ===
            self.LOG.add_pair(param_name, '', fmt='{key}')
            # ===========

        # === LOG ===
        if len(not_updated_variables) > 0:
            self.LOG.switch_group('missing variables')

            for var in not_updated_variables:
                self.LOG.add_pair(var, '', fmt='{key}')

            self.LOG.set_header('WARNING (Missing vairables): Load model')
            self.LOG.dump_to_log(level=logging.WARNING)
        else:
            self.LOG.dump_to_log(level=logging.DEBUG)
        # ===========

        self.sess.run(param_update_ops, feed_dict=feed_dict)

    
    def _get_parameter_list(self):
        return self.parameters

    def _get_parameters(self):
        parameters = self._get_parameter_list()
        parameter_values = self.sess.run(parameters)

        return_dictionary = OrderedDict()

        # === LOG ===
        self.LOG.set_header('DEBUG LOG: Save model')
        self.LOG.switch_group('variables')
        # ===========

        for param, value in zip(parameters, parameter_values):
            # === LOG ===
            self.LOG.add_pair(param.name, '', fmt='{key}')
            # ===========

            return_dictionary[param.name] = value

        #return_dictionary = OrderedDict((param.name, value) for param, value in zip(parameters, parameter_values))

        # === LOG ===
        self.LOG.dump_to_log(level=logging.DEBUG)
        # ===========

        return return_dictionary

    def __del__(self):
        
        tf.reset_default_graph()

        self.sess.close()

