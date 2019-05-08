import torch.autograd as autograd
import torch
import pickle
import csv
import sys

########## get the Discourse Argument extraction results ###########

# Get the arguments
# open the argument parsing results (discourse arguments)
if len(sys.argv)!=3:
    print("USAGE> Prepare_DA_WordEmbeddings_Seqs.py [arg_csv_file] [dataset_name]")
da_file = open(sys.argv[1], "r")
da_csv = csv.reader(da_file)

tweet_id = 0
tweet_da_seqs = []
das = []



next(da_csv) # skip the header
for line in da_csv:
    if tweet_id != int(line[0]):
        tweet_da_seqs.append(das)
        das = []

    das.append([token for token in line[2].split(' ')])
    tweet_id = int(line[0])

tweet_da_seqs.append(das)  # input the final da sequence for the final tweet
del (tweet_da_seqs[0])  # delete the first empty element
########## tweet da word seq list load completed ##########


########## saving pickle files ############
glove_25_dict=pickle.load(open("/data/glove/glove_25.dict","rb"))
glove_50_dict=pickle.load(open("/data/glove/glove_50.dict","rb"))
glove_100_dict=pickle.load(open("/data/glove/glove_100.dict","rb"))
glove_200_dict=pickle.load(open("/data/glove/glove_200.dict","rb"))

tweet_25_da_embedding_seqs=[]
tweet_50_da_embedding_seqs=[]
tweet_100_da_embedding_seqs=[]
tweet_200_da_embedding_seqs=[]
for tweet in tweet_da_seqs:
    tweet_25_word_embedding_seqs=[]
    tweet_50_word_embedding_seqs=[]
    tweet_100_word_embedding_seqs=[]
    tweet_200_word_embedding_seqs=[]
    for da in tweet:
        tweet_25_word_embedding_seqs.append([autograd.Variable(torch.FloatTensor(glove_25_dict[u'' + word.lower()])) for word in da if u''+word.lower() in glove_25_dict])
        tweet_50_word_embedding_seqs.append([autograd.Variable(torch.FloatTensor(glove_50_dict[u'' + word.lower()])) for word in da if u''+word.lower() in glove_50_dict])
        tweet_100_word_embedding_seqs.append([autograd.Variable(torch.FloatTensor(glove_100_dict[u'' + word.lower()])) for word in da if u'' + word.lower() in glove_100_dict])
        tweet_200_word_embedding_seqs.append([autograd.Variable(torch.FloatTensor(glove_200_dict[u'' + word.lower()])) for word in da if u'' + word.lower() in glove_200_dict])
    tweet_25_da_embedding_seqs.append(tweet_25_word_embedding_seqs)
    tweet_50_da_embedding_seqs.append(tweet_50_word_embedding_seqs)
    tweet_100_da_embedding_seqs.append(tweet_100_word_embedding_seqs)
    tweet_200_da_embedding_seqs.append(tweet_200_word_embedding_seqs)


pickle.dump(tweet_25_da_embedding_seqs, open("causal_explanation_25_da_embedding_seqs_"+sys.argv[2]+".list", "wb"))
pickle.dump(tweet_50_da_embedding_seqs, open("causal_explanation_50_da_embedding_seqs_"+sys.argv[2]+".list", "wb"))
pickle.dump(tweet_100_da_embedding_seqs, open("causal_explanation_100_da_embedding_seqs_"+sys.argv[2]+".list", "wb"))
pickle.dump(tweet_200_da_embedding_seqs, open("causal_explanation_200_da_embedding_seqs_"+sys.argv[2]+".list", "wb"))