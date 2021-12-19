


triplets = []
with open("./dbpedia_features_preprocessed_updated.txt","r", encoding="utf-8") as in_file:
    print("reading triplets...")
    for line in in_file.readlines():
        try:
            line = eval(line)
            features = line[0]
            label = line[1]
            triplets.append(features)
            #print(features)
            #print(label)
        except Exception as e:
            print("error encounter .... continue....")
            continue

minLength = min(len(x) for x in triplets ) 
maxLength = max(len(x) for x in triplets ) 
meanLength = sum(len(x) for x in triplets ) / len(triplets)

print(f"Min Length: {minLength}")
print(f"Max Length: {maxLength}")
print(f"Mean Length: {meanLength}")