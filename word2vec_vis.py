# -*- coding: utf-8 -*-
"""
# Created on Mon July 14 12:31:10 2019
# Copyright  Yasir Hussain (yaxirhuxxain@yahoo.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()


from tensorflow import keras
import os, re, time, sys,csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)
#keras.backend.set_session(sess)



class Word2Vec_Visualize(object):

    def __init__(self,context_size = 20,
        n_dim = 300,
        n_epochs = 2 ** 32 - 1,
        n_batch = 32,
        n_drop = 0.25,
        learner = "RNN",
        optimizer = "adam",
        learn_rate = 0.001,
        activation = "softmax",
        loss = "sparse_categorical_crossentropy",
        patience = 10):

        self.context_size = context_size
        self.n_dim = n_dim
        self.n_epochs = n_epochs  # Yes, 2**32 is technically infinity
        self.n_batch = n_batch
        self.n_drop = n_drop
        self.vocab_size = None
        self.learn_rate = learn_rate
        self.learner = learner
        self.optimizer = optimizer
        self.activation = activation
        self.loss = loss
        self.patience = patience
        self.word_idx = None


    
    def build_model(self):
        model = keras.Sequential()
        model.add(keras.layers.Embedding(self.vocab_size, self.n_dim, input_length=self.context_size))


        if self.learner == 'RNN':
           model.add(keras.layers.SimpleRNN(self.n_dim))

        elif self.learner == 'LSTM':
            model.add(keras.layers.LSTM(self.n_dim))

        elif self.learner == 'GRU':
            model.add(keras.layers.GRU(self.n_dim))
        


        elif self.learner == 'BiRNN':
            model.add(keras.layers.Bidirectional(keras.layers.SimpleRNN(int(self.n_dim/2))))

        elif self.learner == 'BiLSTM':
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(int(self.n_dim/2))))

        elif self.learner == 'BiGRU':
            model.add(keras.layers.Bidirectional(keras.layers.GRU(int(self.n_dim/2))))
			


        elif self.learner == 'BNBiRNN':
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Bidirectional(keras.layers.SimpleRNN(int(self.n_dim/2))))
            model.add(keras.layers.BatchNormalization())

        elif self.learner == 'BNBiLSTM':
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(int(self.n_dim/2))))
            model.add(keras.layers.BatchNormalization())

        elif self.learner == 'BNBiGRU':
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Bidirectional(keras.layers.GRU(int(self.n_dim/2))))
            model.add(keras.layers.BatchNormalization())



        elif self.learner == 'CNN':
            model.add(keras.layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.GlobalMaxPooling1D())
        
        elif self.learner == 'CNN-RNN':
            model.add(keras.layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.MaxPooling1D())
            model.add(keras.layers.SimpleRNN(self.n_dim))

        elif self.learner == 'CNN-LSTM':
            model.add(keras.layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.MaxPooling1D())
            model.add(keras.layers.LSTM(self.n_dim))
            
        elif self.learner == 'CNN-GRU':
            model.add(keras.layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.MaxPooling1D())
            model.add(keras.layers.GRU(self.n_dim))

        elif self.learner == 'CNN-BiRNN':
            model.add(keras.layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.MaxPooling1D())
            model.add(keras.layers.Bidirectional(keras.layers.SimpleRNN(self.n_dim)))

            
        elif self.learner == 'CNN-BiLSTM':
            model.add(keras.layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.MaxPooling1D())
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(self.n_dim)))
            
        elif self.learner == 'CNN-BiGRU':
            model.add(keras.layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.MaxPooling1D())
            model.add(keras.layers.Bidirectional(keras.layers.GRU(self.n_dim)))
        
        elif self.learner == 'BNCNN':
            model.add(keras.layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Flatten())
            model.add(keras.layers.BatchNormalization())

        
        elif self.learner == 'BNCNN-RNN':
            model.add(keras.layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.SimpleRNN(self.n_dim))
            model.add(keras.layers.BatchNormalization())

        
        elif self.learner == 'BNCNN-LSTM':
            model.add(keras.layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.LSTM(self.n_dim))
            model.add(keras.layers.BatchNormalization())


        elif self.learner == 'BNCNN-GRU':
            model.add(keras.layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.GRU(self.n_dim))
            model.add(keras.layers.BatchNormalization())

        
        elif self.learner == 'BNCNN-BiRNN':
            model.add(keras.layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Bidirectional(keras.layers.SimpleRNN(self.n_dim)))
            model.add(keras.layers.BatchNormalization())

        
        elif self.learner == 'BNCNN-BiLSTM':
            model.add(keras.layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(self.n_dim)))
            model.add(keras.layers.BatchNormalization())

        
        elif self.learner == 'BNCNN-BiGRU':
            model.add(keras.layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Bidirectional(keras.layers.GRU(self.n_dim)))
            model.add(keras.layers.BatchNormalization())




        
        elif self.learner == 'CNN1':
            model.add(keras.layers.Conv1D(filters=200, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.GlobalMaxPooling1D())

            
        elif self.learner == 'CNN2':
            model.add(keras.layers.Conv1D(filters=200, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.MaxPooling1D())
            model.add(keras.layers.Conv1D(filters=200, kernel_size=4, activation='relu', padding='same'))
            model.add(keras.layers.GlobalMaxPooling1D())
        
        elif self.learner == 'CNN2-LSTM':
            model.add(keras.layers.Conv1D(filters=200, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.MaxPooling1D())
            model.add(keras.layers.Conv1D(filters=200, kernel_size=4, activation='relu', padding='same'))
            model.add(keras.layers.MaxPooling1D())
            model.add(keras.layers.LSTM(self.n_dim))
        
        elif self.learner == 'BNCNN2-LSTM':
            model.add(keras.layers.Conv1D(filters=200, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Conv1D(filters=200, kernel_size=4, activation='relu', padding='same'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.LSTM(self.n_dim))
            model.add(keras.layers.BatchNormalization())

        else:
            sys.exit("Invalid Learner")

        if self.n_drop:
            model.add(keras.layers.Dropout(self.n_drop))


        if self.activation == 'softmax':
            model.add(keras.layers.Dense(self.vocab_size, activation='softmax'))
        elif self.activation == 'sigmoid':
            model.add(keras.layers.Dense(self.vocab_size, activation='sigmoid'))
        else:
            print("Unknown activation: %r", self.learner)
            sys.exit(2)


        model.summary()

        return model


    def load_model(self,model, model_folder):
        model.load_weights(os.path.join(model_folder,'model.hdf5'))
        print("Saved Model loaded")


    def compile_model(self,model):

        if self.optimizer == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(lr=self.learn_rate)
        elif self.optimizer == 'adam':
            optimizer = keras.optimizers.Adam(lr=self.learn_rate)
        else:
            print("Unknown optimizer: %r", self.optimizer)
            sys.exit(2)

        model.compile(loss=self.loss,
                      optimizer=optimizer,
                      metrics=['acc'])

        print(" Model Successfully Compiled")

    def load_voacb(self,model_folder):
        #loading vocabulry dict
        import pickle
        with open(os.path.join(model_folder,"word.idx"), "rb") as file:
           #word_idx = pickle.load(file)
           word_idx = eval(file.read())

        vocab_size = len(word_idx)
        return vocab_size, word_idx


    def visualize(self,model_folder):

            #Preparing Data
            print('Load Vocabulary')
            self.vocab_size, self.word_idx = self.load_voacb(model_folder)

            #Building Model
            print('Building model')
            model = self.build_model()

            #loading weight
            print('loading weight')
            self.load_model(model,model_folder)
            
            e = model.layers[0]
            weights = e.get_weights()[0]
            print(weights.shape) # shape: (vocab_size, embedding_dim)
            self.plot_tsne(weights)

            '''
            # for tensorflow visulationation
            import io
            out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
            out_m = io.open('meta.tsv', 'w', encoding='utf-8')

            code_emded = {}
            for num, word in enumerate(self.word_idx):
                vec = weights[num] 
                code_emded[word] = vec
                out_m.write(word + "\n")
                out_v.write('\t'.join([str(x) for x in vec]) + "\n")
                
            out_v.close()
            out_m.close()
            '''

    def plot_tsne(self,weights):
        "Creates TSNE model and plots it"
        labels = []
        tokens = []

        print("Extracting Code-Embeddings")
        count = 0
        MAX_SHOW = 300
        for num, word in enumerate(self.word_idx):
            count +=1
            if count == MAX_SHOW:
                break
            vec = weights[num] 
            tokens.append(vec)
            labels.append(word)
        
        print(f"({len(tokens)},{MAX_SHOW})")
        print("Building TSNE model")
        tsne_model = TSNE(n_components=2, random_state=7)
        new_values = tsne_model.fit_transform(tokens)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
            
        
        print("ploting TSNE")
        plt.figure(figsize=(16, 16))  
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                        xy=(x[i], y[i]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')
                        
        #plt.xlim([-1, 1])
        #plt.ylim([-1, 1])   
        #plt.show()
        
        print("Saving TSNE plot")
        #plt.savefig('poi_word2vec_vis.eps', format='eps')
        plt.savefig(f'{self.learner}_{MAX_SHOW}.png', format='png')
        print("Done Saving")


import argparse
# Create the argument parser.
parser = argparse.ArgumentParser(description='CodeGRU Model Trainer')

parser.add_argument('-m','--model_folder', type=str, required=True, default=(r"saveModels"),
                    help='Saved model location')


parser.add_argument('-l','--learner', type=str, required=True, 
                    help='which type of learner to use (RNN,LSTM,GRU,BiRNN,BiLSTM,BiGRU,BNBiRNN,BNBiLSTM,BNBiGRU,\
                                                        CNN,CNN-RNN,CNN-LSTM,CNN-GRU,CNN-BiRNN,CNN-BiLSTM,CNN-BiGRU, \
                                                        BNCNN,BNCNN-RNN,BNCNN-LSTM,BNCNN-GRU,BNCNN-BiRNN,BNCNN-BiLSTM,BNCNN-BiGRU)')

def main():

    #Setting Prams
    ROOT = os.getcwd()
    LEARNER = "LSTM" #"RNN,LSTM,GRU"
    MODEL_DIR = os.path.join(ROOT,os.path.join("SavedModels",LEARNER))
    EPOCHS = 2 ** 32 - 1

    # Clear the session
    print('Clear Session...')
    tf.keras.backend.clear_session()

    print("Learner: {} > Model Location:{}".format(LEARNER,MODEL_DIR))

    Word2Vec = Word2Vec_Visualize(
        context_size = 7,
        n_dim = 21,
        n_epochs = EPOCHS,
        n_batch = 256,
        n_drop = 0.5,
        learner = LEARNER,
        optimizer = "adam",
        learn_rate = 0.001,
        activation = "softmax",
        loss = "sparse_categorical_crossentropy",
        patience = 3)

    Word2Vec.visualize(model_folder=MODEL_DIR)


if __name__ == "__main__":
    main()
