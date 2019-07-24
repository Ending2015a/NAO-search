import os
import sys
import time
import logging

import tensorflow as tf

from encoder import encoder
from decoder import decoder




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
                source_length = 60,
                encoder_length = 60,
                decoder_length = 60,
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
                _init_setup_model=False):
    
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
        
        model._load_parameters(params)
        
        return model

    def save(self, save_path):
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
        
        params = self._get_parameters()
        
        self._save_model(save_path, data, params)
        
        
    def _setup_model(self):
        

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
    
    
    def _load_parameters(self):
    
    def _get_parameters(self):
