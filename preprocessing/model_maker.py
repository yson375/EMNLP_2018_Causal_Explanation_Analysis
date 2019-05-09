import csv
import pickle


glove_25_file=open("glove.twitter.27B.25d.txt")
glove_50_file=open("glove.twitter.27B.50d.txt")
glove_100_file=open("glove.twitter.27B.100d.txt")
glove_200_file=open("glove.twitter.27B.200d.txt")

glove_25_dict=dict()
for line in glove_25_file:
    line=line.split(' ')
    glove_25_dict[line[0]]=[float(dim) for dim in line[1:]]
print("25_dict_made")

glove_50_dict=dict()
for line in glove_50_file:
    line=line.split(' ')
    glove_50_dict[line[0]]=[float(dim) for dim in line[1:]]
print("50_dict_made")

glove_100_dict=dict()
for line in glove_100_file:
    line=line.split(' ')
    glove_100_dict[line[0]]=[float(dim) for dim in line[1:]]
print("100_dict_made")

glove_200_dict=dict()
for line in glove_200_file:
    line=line.split(' ')
    glove_200_dict[line[0]]=[float(dim) for dim in line[1:]]
print("200_dict_made")

pickle.dump(glove_25_dict, open("glove_25.dict","wb"))
pickle.dump(glove_50_dict, open("glove_50.dict","wb"))
pickle.dump(glove_100_dict, open("glove_100.dict","wb"))
pickle.dump(glove_200_dict, open("glove_200.dict","wb"))
