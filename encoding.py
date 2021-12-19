

import tensorflow as tf


with open("word_idx_revised.txt","r", encoding="utf-8") as in_file:
    print("reading word idx...")
    for line in in_file.readlines():
        try:
            word_idx = eval(line)
            print(type(word_idx))
            print(len(word_idx))
        except Exception as e:
            print("error encounter .... continue....")
            continue

MAX_LENGTH = 7
count = 0
features = []
labels = []
with open("dbpedia_features_preprocessed_revised.txt","r", encoding="utf-8") as in_file:
    print("reading triplets...")
    for line in in_file.readlines():
        try:
            count += 1
            print(count)

            # if count >= 10: 
            #     break
            
            line = eval(line)
            feature = line[0]
            label = line[1]
            # print(feature)
            # print(label)

            #[print(f"{word} : {word_idx.get(word,0)}") for word in feature ]
            #print(f"{label} : {word_idx.get(label,0)}")

            features.append([word_idx.get(word,0) for word in feature]) # encoding features to integers
            labels.append(word_idx.get(label,0))
            
            # print(features)
            # print(labels)

        except Exception as e:
            # print(e)
            # break
            print("error encounter .... continue....")
            continue

features = tf.keras.preprocessing.sequence.pad_sequences(features,MAX_LENGTH).tolist()
#print(features)
#print(labels)

with open("./encoded_features_revised.txt", "a") as f:
    f.write(str(features))

with open("./encoded_labels_revised.txt", "a") as f:
    f.write(str(labels))