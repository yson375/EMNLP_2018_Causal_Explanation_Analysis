import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
import pickle
import sys
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix


class BiLSTM_Causal_Explanation_Identification(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, is_cuda, causality_num_layer=1,causality_num_direction=1):
        super(BiLSTM_Causal_Explanation_Identification, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.causality_num_layer = causality_num_layer
        self.causality_num_direction = causality_num_direction
        self.is_cuda=is_cuda
        self.du_lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.causality_lstm = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=causality_num_direction == 2)



        self.hidden_to_causality = nn.Linear(hidden_dim * causality_num_direction, 2)

        if is_cuda:
            self.du_lstm = self.du_lstm.cuda()
            self.causality_lstm = self.causality_lstm.cuda()
            self.hidden_to_causality = self.hidden_to_causality.cuda()

    def init_du_hidden(self):
        if self.is_cuda:
            return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim)).cuda(),autograd.Variable(torch.zeros(2, 1, self.hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim)),autograd.Variable(torch.zeros(2, 1, self.hidden_dim)))

    def init_classifier_hidden(self):
        if self.is_cuda:
            return (autograd.Variable(torch.zeros(self.causality_num_layer * self.causality_num_direction, 1, self.hidden_dim)).cuda(),autograd.Variable(torch.zeros(self.causality_num_layer * self.causality_num_direction, 1, self.hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(self.causality_num_layer * self.causality_num_direction, 1, self.hidden_dim)),autograd.Variable(torch.zeros(self.causality_num_layer * self.causality_num_direction, 1, self.hidden_dim)))

    def forward(self, du_embedding_seq):
        # use batch input and get all the hidden vectors and concatenate them
        tweet_input = []
        # keep track of missing dus and average all embeddings of dus 
        empty_seq_du_idxs=[]
        for i in range(len(du_embedding_seq)):
            # get du embeddings from the given tweet
            # print(word_embedding_seq)
            word_embedding_seq = du_embedding_seq[i]
            if len(word_embedding_seq)==0:
                empty_seq_du_idxs.append(i)
                continue
            word_embedding_seq = torch.cat(word_embedding_seq).view(len(word_embedding_seq), 1, -1)
            if self.is_cuda:
                word_embedding_seq = word_embedding_seq.cuda()

            du_output, (du_hidden, du_cell_state) = self.du_lstm(word_embedding_seq, self.init_du_hidden())
            tweet_input.append(du_hidden.view(self.hidden_dim*2, 1, -1))

        tweet_input=list(tweet_input)


        tweet_input_mean = torch.stack(tweet_input).mean(dim=0)




        for i in range(len(empty_seq_du_idxs)):
            tweet_input.insert(i+empty_seq_du_idxs[i], tweet_input_mean)


        tweet_input = torch.cat(tweet_input).view(len(tweet_input), 1,-1)  # concat hidden vectors from the last cells of forward and backward LSTM
        tweet_input = F.dropout(tweet_input, p=0.3, training=self.training)
        tweet_output, (tweet_hidden, tweet_cell_state) = self.causality_lstm(tweet_input, self.init_classifier_hidden())

        causality_vec = self.hidden_to_causality(tweet_output.view(len(du_embedding_seq), -1))
        causality_score = F.log_softmax(causality_vec,dim=1)
        return causality_score

