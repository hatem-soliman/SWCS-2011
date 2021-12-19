

with open("word_count_sorted_updated.txt","r", encoding="utf-8") as in_file:
    print("reading word count...")
    for line in in_file.readlines():
        try:
            word_count = eval(line)
            print(type(word_count))
            print(len(word_count))
        except Exception as e:
            print("error encounter .... continue....")
            continue

word_idx = {}
word_idx["unk"] = 0
count = 0
for key,value in word_count.items():
    if value <= 10:
        continue
    else:
        count += 1
        word_idx[key] = count

print(len(word_idx))
f = open("./word_idx_updated.txt", "w", encoding="utf-8")
f.write(str(word_idx))
f.close()