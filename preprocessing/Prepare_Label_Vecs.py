import torch.autograd as autograd
import torch
import pickle
import csv
import sys


# Get the arguments
# open the argument parsing results (discourse arguments)
if len(sys.argv)!=3:
    print("USAGE> Prepare_Label_Vecs.py [label_file] [dataset_name]")
ce_file = open(sys.argv[1], "r")
ce_csv = csv.reader(ce_file)

tweet_id = 0
ce_vec_seqs = []
ce_vec = []


next(ce_csv) # skip the header
for line in ce_csv:
    causality_vec = []
    if tweet_id != int(line[0]):
        
        ce_vec_seqs.append(autograd.Variable(torch.LongTensor(ce_vec)))
        ce_vec = []

    ce_vec.append(int(line[2]))
    tweet_id = int(line[0])


ce_vec_seqs.append(autograd.Variable(torch.LongTensor(ce_vec))) # input the final ce vec for the final tweet
del (ce_vec_seqs[0])  # delete the first empty element

pickle.dump(ce_vec_seqs, open("causal_explanation_da_labels_"+sys.argv[2]+".list", "wb"))