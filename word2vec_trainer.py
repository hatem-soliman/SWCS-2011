
import os
import sys
import time
from nltk.util import ngrams
from gensim.models import Word2Vec

from util import sec_to_hms

class trainer(object):

    def train_model(self,projects,window=5,size=300,min_count=1,epochs=100,workers=4,algorithm="CBOW",negative_samplig=True,ngram_count=1):
        # build model
        # vec size usually more is better, but not always
        # nagative sampling for skip-gram usually around 10, for CBOW around 5
        sg = 0  # CBOW Method (skip-gram (slower, better for infrequent words) vs CBOW (fast))
        hs = 0  # Negative sampling ( <1> hierarchical softmax (better for infrequent words) vs <0> negative sampling (better for frequent words, better with low dimensional vectors))
        negative = 0 # negative sampling size (for skip-gram usually around 10, for CBOW around 5)

        if algorithm == "CBOW":
            sg = 0
        elif algorithm == "skip-gram":
            sg = 1
        else:
            sys.exit("invalid traning method")


        if negative_samplig == True:
            hs = 0
            if algorithm == "CBOW":
                negative = 5
            elif algorithm == "skip-gram":
                negative = 10
            else:
                sys.exit("invalid traning method")
        else:
            hs = 1
            negative = 0


        try:
            start_time = time.time()
            print("Traning model")
            self.model = Word2Vec(projects, size=size, window=window, min_count=min_count, iter=epochs, workers=workers, sg=sg, hs=hs, negative=negative)


            self.size = size
            self.window = window
            self.min_count = min_count
            self.epochs = epochs
            self.workers = workers
            self.sg = sg
            self.hs = hs
            self.negative = negative

            self.vocab_count = len(self.model.wv.vocab)
            self.train_time = sec_to_hms(time.time()-start_time)

        except Exception as e:
            sys.exit(f"Error Traning Model {e}")


    def save_model(self,out_dir):
        print("Saving Model")
        
        file_name = os.path.join(out_dir,f"Word2Vec-{str(round(time.time()))}")
        self.model.wv.save_word2vec_format(file_name+'.bin', binary=True)
        self.model.wv.save_word2vec_format(file_name+'.txt', binary=False)

        with open(file_name+".info","w+") as file:
            file.write(f"vocab_count: {self.vocab_count}")
            file.write("\n")
            file.write(f"size: {self.size}")
            file.write("\n")
            file.write(f"min_count: {self.min_count}")
            file.write("\n")
            file.write(f"epochs: {self.epochs}")
            file.write("\n")
            file.write(f"workers: {self.workers}")
            file.write("\n")
            file.write(f"Skip-gram: {self.sg}")
            file.write("\n")
            file.write(f"window-size: {self.window}")
            file.write("\n")
            file.write(f"negative-sampling: {self.hs}")
            file.write("\n")
            file.write(f"negative-sampling-size: {self.negative}")
            file.write("\n")
            file.write(f"train_time: {self.train_time}")
            file.write("\n")




import argparse
# Create the argument parser.
parser = argparse.ArgumentParser(description='Word2Vec Model Trainer')

# required options
parser.add_argument('-i', '--in_folder', type=str, required=True,
                    help='Model traning folder')

parser.add_argument('-o','--out_dir', type=str, required=True,
                    help='Model Saving directory')

parser.add_argument('-w','--window', type=int, required=False, default=5,
                    help='Window size')

parser.add_argument('-s','--size', type=int, required=False, default=300,
                    help='Vector size')

parser.add_argument('-m','--min_count', type=int, required=False, default=1,
                    help='Minimum word count')

parser.add_argument('-e','--epochs', type=int, required=False, default=5,
                    help='Num epochs')

parser.add_argument('--workers', type=int, required=False, default=4,
                    help='Num workers')

parser.add_argument('-a','--algorithm', type=str, required=False, default="CBOW", choices=("CBOW","skip-gram"),
                    help='Traning method ')

parser.add_argument('-n','--negative_samplig', type=bool, required=False, default=True,
                    help='Negative Samplig')



def main():
    print("Loading & Preprocessing Files...")
    
    root_path = os.getcwd()
    args = parser.parse_args()
    in_folder = os.path.join(root_path,args.in_folder)
    out_dir = os.path.join(root_path,os.path.join(args.out_dir,'Word2Vec'))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    list_projects =  [name for name in os.listdir(in_folder)]

    word_count= {}

    projects = []
    for project in list_projects:
        in_file = os.path.join(in_folder,project)
        data = open(in_file,"r",encoding="utf-8").read() #read Project
        files_list = data.split('\n') #get files
        for file in files_list:
            tokens_stream = file.split()
            projects.append(tokens_stream) #append to tokanized projects list
            for word in tokens_stream:
                try:
                    word_count[word] += 1
                except:
                    word_count[word] = 1

    word_count = dict(sorted(word_count.items(), key=lambda kv: kv[1], reverse=True))

    count = 0
    word_idx = {}
    word_idx["unk"] = 0
    for key, value in word_count.items():
        if value <= 1:
            continue
        else:
            count += 1
            word_idx[key] = count

    vocab_size = len(word_idx)
    print(f"vocab_size: {vocab_size}")

    for tokens_stream in projects:
        for c, token in enumerate(tokens_stream):
            if token in word_idx:
                continue
            else:
                tokens_stream[c] = 'unk'
                print(tokens_stream)
                sys.exit()


    workers  = 32
    window = 5
    size = 300
    min_count = 1
    epochs = 100
    algorithm = "CBOW"
    negative_samplig =True
    trainer_obj = trainer()
    trainer_obj.train_model(projects=projects,
                            window=window,
                            size=size,
                            min_count=min_count,
                            epochs=epochs,
                            workers=workers,
                            algorithm=algorithm,
                            negative_samplig=negative_samplig,
                            ngram_count=1)

    trainer_obj.save_model(out_dir)
    print("Done Model Training & Saving...")


if __name__ == '__main__':
    main()
