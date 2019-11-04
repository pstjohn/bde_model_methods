import sys

runid = int(sys.argv[1])
num_messages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10][runid]
model_name = 'layer_runs2/n_message_{}'.format(num_messages)
print(model_name)

atom_features = 128
# num_messages = 8
lr = 1E-3

batch_size = 128
decay = 1E-5

import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import shutil

import gzip
import pickle
import warnings

import keras
import keras.backend as K

from keras.layers import (
    Input, Embedding, Dense, BatchNormalization, Dropout,
    Reshape, Lambda, Activation, Add, Concatenate, Multiply)

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from keras.models import Model
from keras.engine import Layer

from nfp.layers import (GRUStep, ReduceAtomOrBondToMol, Embedding2D, Squeeze,
                        MessageLayer, GatherAtomToBond, ReduceBondToAtom,
                        ReduceAtomOrBondToMol, GatherMolToAtomOrBond)
from nfp.models import GraphModel, masked_mean_absolute_error

from preprocessor_utils import ConcatGraphSequence

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)

with gzip.open('processed_inputs_190531.p.gz', 'rb') as f:
    inputs = pickle.load(f)
    
preprocessor = inputs['preprocessor']

inputs_train = inputs['train'][0]
y_train = inputs['train'][1]

inputs_valid = inputs['valid'][0]
y_valid = inputs['valid'][1]


# Construct input sequences
train_sequence = ConcatGraphSequence(inputs_train, y_train, batch_size, final_batch=False)
valid_sequence = ConcatGraphSequence(inputs_valid, y_valid, batch_size, final_batch=False)


# Raw (integer) graph inputs
mol_type = Input(shape=(1,), name='n_atom', dtype='int32')
node_graph_indices = Input(shape=(1,), name='node_graph_indices', dtype='int32')
bond_graph_indices = Input(shape=(1,), name='bond_graph_indices', dtype='int32')
atom_types = Input(shape=(1,), name='atom', dtype='int32')
bond_types = Input(shape=(1,), name='bond', dtype='int32')
connectivity = Input(shape=(2,), name='connectivity', dtype='int32')

squeeze = Squeeze()

snode_graph_indices = squeeze(node_graph_indices)
sbond_graph_indices = squeeze(bond_graph_indices)
smol_type = squeeze(mol_type)
satom_types = squeeze(atom_types)
sbond_types = squeeze(bond_types)

# Initialize RNN and MessageLayer instances

# Initialize the atom states
atom_state = Embedding(
    preprocessor.atom_classes,
    atom_features, name='atom_embedding')(satom_types)

# Initialize the bond states
bond_state = Embedding(
    preprocessor.bond_classes,
    atom_features, name='bond_embedding')(sbond_types)

# Initialize the bond states
bond_mean = Embedding(
    preprocessor.bond_classes,
    1, name='bond_mean')(sbond_types)


def message_block(original_atom_state, original_bond_state, connectivity, i):
    
    atom_state = BatchNormalization()(original_atom_state)
    bond_state = BatchNormalization()(original_bond_state)
    
    source_atom_gather = GatherAtomToBond(1)
    target_atom_gather = GatherAtomToBond(0)

    source_atom = source_atom_gather([atom_state, connectivity])
    target_atom = target_atom_gather([atom_state, connectivity])

    # Edge update network
    new_bond_state = Concatenate(name='concat_{}'.format(i))([
        source_atom, target_atom, bond_state])
    new_bond_state = Dense(
        2*atom_features, activation='relu')(new_bond_state)
    new_bond_state = Dense(atom_features)(new_bond_state)

    bond_state = Add()([original_bond_state, new_bond_state])

    # message function
    source_atom = Dense(atom_features)(source_atom)    
    messages = Multiply()([source_atom, bond_state])
    messages = ReduceBondToAtom(reducer='sum')([messages, connectivity])
    
    # state transition function
    messages = Dense(atom_features, activation='relu')(messages)
    messages = Dense(atom_features)(messages)
    
    atom_state = Add()([original_atom_state, messages])
    
    return atom_state, bond_state

for i in range(num_messages):
    atom_state, bond_state = message_block(atom_state, bond_state, connectivity, i)

bond_state = Dense(1)(bond_state)
bond_state = Add()([bond_state, bond_mean])

symb_inputs = [mol_type, node_graph_indices, bond_graph_indices,
               atom_types, bond_types, connectivity]

model = GraphModel(symb_inputs, [bond_state])

epochs = 500

model.compile(optimizer=keras.optimizers.Adam(lr=lr, decay=decay),
              loss=masked_mean_absolute_error)

if not os.path.exists(model_name):
    os.makedirs(model_name)

# Make a backup of the job submission script
shutil.copy(__file__, model_name)
    
filepath = model_name + "/best_model.hdf5"
checkpoint = ModelCheckpoint(filepath, save_best_only=True, period=10,
                             verbose=0)
csv_logger = CSVLogger(model_name + '/log.csv')

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    hist = model.fit_generator(train_sequence, validation_data=valid_sequence,
                               epochs=epochs, verbose=1,
                               callbacks=[checkpoint, csv_logger])
