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
import argparse
import time
import math
from BiLSTM_Causal_Explanation_Identification import BiLSTM_Causal_Explanation_Identification


def time_since(start):
    now = time.time()
    s = now - start
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


parser = argparse.ArgumentParser(description='BiLSTM_Causal_Explanation_Identificatio_Training')
def main():

    parser.add_argument('--word_dim', type=int, default=25, help='the dimension of the word embedding to be used. Default=25')
    parser.add_argument('--hidden_dim', type=int, default=25, help='the dimension of the hidden layer to be used. Default=25')
    parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=1')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--grad', type=str, default='SGD', help='Optimzer type: SGD? Adam? Default=SGD')
    parser.add_argument('--train_shuffle', type=str, default='no',
                        help='Shuffle the dataset for each epoch? no/shuffle/replace')
    parser.add_argument('--model_path', type=str, default='./Trained_Models/',
                        help='the path for saving intermediate models')
    parser.add_argument('--train_data', type=str, default='./preprocessed_embeddings/',
                        help='the path of word embedding seqs for training')
    parser.add_argument('--train_labels', type=str, default='./preprocessed_embeddings/',
                        help='the path of label vectors for training')
    parser.add_argument('--valid_data', type=str, default='./preprocessed_embeddings/',
                        help='the path of word embedding seqs for validation')
    parser.add_argument('--valid_labels', type=str, default='./preprocessed_embeddings/',
                        help='the path of label vectors for validation')
    parser.add_argument('--causality_num_direction', type=int, default=2,
                        help='# of direction of RNN for causality detection, Default=2 (bidirectional)')

    opt = parser.parse_args()
    print(opt)
    word_embedding_dim=opt.word_dim
    hidden_dim=opt.hidden_dim
    learning_rate=opt.lr
    optimizer_type=opt.grad
    c_num=opt.causality_num_direction
    


    is_cuda= opt.cuda
    model_name="CE_"+str(word_embedding_dim)+'_'+str(hidden_dim)+'_lr'+str(learning_rate).replace('.','_')+'_'+optimizer_type+'_'
    torch.manual_seed(opt.seed)
    if is_cuda:
        torch.cuda.manual_seed(opt.seed)


    # word embedding pickle
    train_word_embed_seqs = pickle.load(open(opt.train_data, "rb"))
    valid_word_embed_seqs = pickle.load(open(opt.valid_data, "rb"))
    training_set_size=len(train_word_embed_seqs)
    valid_set_size=len(valid_word_embed_seqs)
    print("Training Set Size:",training_set_size)
    print("Validation Set Size:",valid_set_size)
    # tweet da pickle
    # causality gs pickle
    ce_train_label_vecs = pickle.load(open(opt.train_labels,"rb"))
    ce_train_labels = [int(ce) for ce_train_label_vec in ce_train_label_vecs for ce in ce_train_label_vec]
    ce_valid_label_vecs = pickle.load(open(opt.valid_labels,"rb"))
    ce_valid_labels = [int(ce) for ce_valid_label_vec in ce_valid_label_vecs for ce in ce_valid_label_vec]

    model = BiLSTM_Causal_Explanation_Identification(word_embedding_dim, hidden_dim, is_cuda, causality_num_direction=c_num)
    loss_function = nn.NLLLoss()
    
    if optimizer_type=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if is_cuda:
        model = model.cuda()
        loss_function = loss_function.cuda()
        ce_train_label_vecs=[ce_train_label_vec.cuda() for ce_train_label_vec in ce_train_label_vecs]
        ce_valid_label_vecs=[ce_valid_label_vec.cuda() for ce_valid_label_vec in ce_valid_label_vecs]



    best_weighted_f1=0.80 # preliminary best f1
    total_start=time.time()

    for epoch in range(1,opt.nEpochs+1):
        epoch_start=time.time()
        model.train()
        losses=0
        if opt.train_shuffle=='replace':
            training_set=np.random.choice(training_set_size,training_set_size,replace=True)
        elif opt.train_shuffle=='shuffle':
            training_set=np.random.choice(training_set_size,training_set_size,replace=False)
        else:
            training_set=range(training_set_size)
        
        for i in training_set:
            model.zero_grad()
            causality_score = model(train_word_embed_seqs[i])
            loss = loss_function(causality_score, ce_train_label_vecs[i])
            loss.backward()
            optimizer.step()
            losses+=float(loss)

        end_time=time_since(epoch_start)
        total_time=time_since(total_start)
        print("Epoch #: " + str(epoch))
        print("Training Loss: " + str(losses))
        print("Epoch Time: %s"%(end_time))


        if epoch%10==0:
            model.eval()
            predictions=[]
            valid_losses = 0
            for i in range(valid_set_size):
                causality_score=model(valid_word_embed_seqs[i])
                _,prediction=causality_score.max(1)
                [predictions.append(int(int_label)) for int_label in prediction]
                valid_loss = loss_function(causality_score, ce_valid_label_vecs[i])
                valid_losses += float(valid_loss)

            precision, recall, f1, support = precision_recall_fscore_support(ce_valid_labels, predictions)
            weighted_precision, weighted_recall, weighted_f1, weighted_support = precision_recall_fscore_support(ce_valid_labels, predictions, average='weighted')


            if weighted_f1>best_weighted_f1:

                best_weighted_f1=weighted_f1
                print("The bset F1 upto this point: " + str(weighted_f1))
                torch.save(model.state_dict(),opt.model_path + model_name + "dir_" + str(c_num) + "_f1_" + str(weighted_f1)[:7] + '_epoch_' + str(epoch) + '_Dropout_0_3' + "_early_stop_saved.ptstdict")



            print("[Results at epoch #: "+str(epoch)+"]")
            print("F1: "+str(f1))
            print("Weighted F1: " + str(weighted_f1))
            print("Precision: " + str(precision))
            print("Recall: " + str(recall))
            print("Confusion Matrix (Label \\ Prediction)")
            print(confusion_matrix(ce_valid_labels, predictions))
            print("Loss: " + str(losses))
            print("Validation Loss: " + str(valid_losses))
            print("Epoch Time: %s"%(end_time))
            print("Total Time: %s"%(total_time))







if __name__=="__main__":
    main()
