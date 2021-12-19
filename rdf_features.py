import re
import nltk 

from textblob import Word
from nltk.corpus import stopwords
from nltk.tag import pos_tag

import splitter
import wordninja

from nltk.stem import PorterStemmer
porter = PorterStemmer()

from nltk.stem import LancasterStemmer
lancaster=LancasterStemmer()

from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()  
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')


def custom_pos_lema(word,tag):
    if tag.startswith("NN"):
        return lemmatizer.lemmatize(word, pos='n')
    elif tag.startswith('VB'):
        return lemmatizer.lemmatize(word, pos='v')
    elif tag.startswith('JJ'):
        return lemmatizer.lemmatize(word, pos='a')
    else:
        return word


word_count = {}
features = []
labels = []
count = 0
preprocessed = []
print("loading...")

f = open("./dbpedia_features_preprocessed.txt", "a")


with open("./dbpedia_features_full.txt","r", encoding="utf-8") as in_file:
    print("reading triplets...")
    for line in in_file.readlines():
        count +=1
        #if count >= 20:
        #   break
        print(count)
        try:
            triplet = eval(line)
            if not triplet[0]:
                print(f"no subject found: {triplet[0]}")
                continue
            elif not triplet[-1]:
                print(f"no object found: {triplet[-1]}")
                continue
           
            #print(f"Raw: {triplet}")

            raw = re.sub(r'\_', " "," ".join(triplet[0:2])).lower()
            #print(raw)
            #use one of below
            if splitter.split(raw):
                raw = splitter.split(raw)
            else:
                raw = raw.split()
            #print(raw)

            raw = list(filter(None,raw))                    #filter result
            raw = [Word(token).correct() for token in raw]  # Spell Correction
            #print(f"Spell correction: {raw}")

            #Stemmer
            raw = [porter.stem(token) for token in raw]     
            #print(f"Stemmer: {raw}")  

            raw = [re.sub(r'\W', '', word) for word in raw] # non-dict word removal
            raw = list(filter(None,raw))                    #filter result
            #print(f"Non Dict: {raw}")

            STOPWORDS = set(stopwords.words('english')) 
            raw = [w for w in raw if not w in STOPWORDS]    #stop words removel
            #print(f"Stop Words: {raw}")

            raw = pos_tag(raw)                      # POS Taging
            raw = [custom_pos_lema(token[0],token[1]) for token in raw]     
            #raw = [lemmatizer.lemmatize(token, pos ="a") for token in raw]     
            #print(f"Lemmatizer: {raw}")   

            #raw = [Word(token).singularize() for token in raw]  # inflections (singularize(),pluralize())
            ##print(f"Inflection: {raw}")

            #features.append(raw)
            for word in raw:
                try:
                    word_count[word] += 1
                except Exception as e:
                    word_count[word] = 1


            pred = triplet[-1]
            #labels.append([pred])
            try:
                word_count[pred] += 1
            except Exception as e:
                word_count[pred] = 1

            preprocessed = [raw,pred]

            print(f"Final: {preprocessed}")
            f.write(f"{str(preprocessed)}\n")
            
        except Exception as e:
            print("error encounter .... continue....")
            #print(e)
            continue

f.close()
#print(features)
#print(labels)
print(len(word_count))
f = open("./word_count.txt", "w")
f.write(str(word_count))
f.close()


word_count = dict(sorted(word_count.items(), key=lambda kv: kv[1], reverse=True))
print(len(word_count))
f = open("./word_count_sorted.txt", "w")
f.write(str(word_count))
f.close()


word_idx = {}
word_idx["unk"] = 0
count = 0
for key,value in word_count.items():
    if value <= 1:
        continue
    else:
        count += 1
        word_idx[key] = count

print(len(word_idx))
f = open("./word_idx.txt", "w")
f.write(str(word_idx))
f.close()





