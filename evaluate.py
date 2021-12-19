



import re

import os, re, time, sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(sess)

class Trainer(object):

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
        patience = 10,
        out_dir=None):

        self.context_size = context_size
        self.n_dim = n_dim
        self.n_epochs = n_epochs  # Yes, 2**32 is technically infinity
        self.n_batch = n_batch
        self.n_drop = n_drop
        self.vocab_size = 67296
        self.learn_rate = learn_rate
        self.learner = learner
        self.optimizer = optimizer
        self.activation = activation
        self.loss = loss
        self.patience = patience
        self.word_idx = None

        if out_dir:
            self.out_dir = out_dir
        else:
            root_path = os.getcwd()
            self.out_dir = os.path.join(root_path,os.path.join('saveModels',self.learner))

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


    def train_model(self,features,labels):

        with tf.device('/gpu:0'):

            #Building Model

            print('Building model')
            model = self.build_model()

            self.compile_model(model)
            self.save_model_summary(model)

            X_train = np.array(features)
            y_train = np.array(labels)

            print(len(X_train))
            print(len(y_train))
            
            #X_train, X_test, y_train, y_test = train_test_split(np.array(features), np.array(labels), test_size=0.3, shuffle=true, random_state=7)

            #Calbacks
            tbCallBack = keras.callbacks.TensorBoard(log_dir=self.out_dir, histogram_freq=0, write_graph=True, write_images=True)
            earlyStopCallBack = keras.callbacks.EarlyStopping(patience=self.patience, monitor='val_loss', mode='auto')
            ModelCheckpointCallBack = keras.callbacks.ModelCheckpoint(os.path.join(self.out_dir, "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"), monitor='val_loss', save_best_only=False, mode='auto')
            CSVLoggerCallBack = keras.callbacks.CSVLogger(os.path.join(self.out_dir,"train_scores.csv"), append=True)
            callbacks_list = [ModelCheckpointCallBack,CSVLoggerCallBack,tbCallBack,earlyStopCallBack]

            #Fit model
            start_time = time.time()
            train_scores = model.fit(X_train, y_train, epochs=self.n_epochs, batch_size=self.n_batch, verbose=1,
                                        validation_split=0.3, callbacks=callbacks_list)

            self.train_time = time.time()-start_time

            #saving model
            self.model_save(model,train_scores)

            print('Model training and testing is done. Exiting by saving model.')

    def save_model_summary(self,model):
        from contextlib import redirect_stdout
        with open(os.path.join(self.out_dir,'model-summary.txt'), 'w') as f:
            with redirect_stdout(f):
                model.summary()

    def model_save(self,model,train_scores):
        with open(os.path.join(self.out_dir,"model.info"),"w") as file:
            file.write(f"Context Size: {str(self.context_size)}")
            file.write("\n")
            file.write(f"Hidden Dim: {str(self.n_dim)}")
            file.write("\n")
            file.write(f"Batch Size: {str(self.n_batch)}")
            file.write("\n")
            file.write(f"Dropout Size: {str(self.n_drop)}")
            file.write("\n")
            file.write(f"Vocab Size: {str(self.vocab_size)}")
            file.write("\n")
            file.write(f"Learner: {str(self.learner)}")
            file.write("\n")
            file.write(f"Optimizer: {str(self.optimizer)}")
            file.write("\n")
            file.write(f"Learn Rate: {str(self.learn_rate)}")
            file.write("\n")
            file.write(f"Activation Function: {str(self.activation)}")
            file.write("\n")
            file.write(f"Loss: {str(self.loss)}")
            file.write("\n")
            file.write(f"Patience: {str(self.patience)}")
            file.write("\n")
            file.write(f"Time (Train & Test): {str(self.train_time)}")
            file.write("\n")

        #Saving train score
        with open(os.path.join(self.out_dir,"train.history"),"w") as file:
            file.write(str(train_scores.history))

        #Saving model
        model.save(os.path.join(self.out_dir,"model.hdf5"))

        #Saving vocabulry dict
        import pickle
        with open(os.path.join(self.out_dir,"word.idx"), "wb") as file:
           pickle.dump(self.word_idx, file)

        print("Done Traning & Saving")



# which type of learner to use \
# RNN,LSTM,GRU,BiRNN,BiLSTM,BiGRU,BNBiRNN,BNBiLSTM,BNBiGRU,\
# CNN,CNN-RNN,CNN-LSTM,CNN-GRU,CNN-BiRNN,CNN-BiLSTM,CNN-BiGRU, \
# BNCNN,BNCNN-RNN,BNCNN-LSTM,BNCNN-GRU,BNCNN-BiRNN,BNCNN-BiLSTM,BNCNN-BiGRU

def main():

    #Setting Prams
    ROOT = os.getcwd()
    LEARNER = "BiGRU" #"RNN,LSTM,GRU"
    OUT_DIR = os.path.join(ROOT,os.path.join("SavedModels",LEARNER))
    EPOCHS = 2 ** 32 - 1


    print("Reading features...")
    with open("encoded_features_revised.txt","r", encoding="utf-8") as in_file:
        file_data = in_file.read()
        features = eval(file_data)

    print("Reading labels...")
    with open("encoded_labels_revised.txt","r", encoding="utf-8") as in_file:
        for line in in_file.readlines():
            try:
                labels = eval(line)
                # print(type(data))
                # print(len(data))
            except Exception as e:
                print("error encounter .... continue....")
                continue




    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    print(OUT_DIR)

    # Clear the session
    print('Clear Session...')
    tf.keras.backend.clear_session()

    print("Traning Model: {} ".format(LEARNER))
    print(f"OUT_DIR: {OUT_DIR}")

    trainer = Trainer(
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
        patience = 3,
        out_dir=OUT_DIR)

    trainer.train_model(features=features,labels=labels)
        
if __name__ == "__main__":
    main()


