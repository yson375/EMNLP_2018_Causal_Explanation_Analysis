import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import csv
import pickle
import sys
import torch.nn.functional as F
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


parser = argparse.ArgumentParser(description='BiLSTM_Causal_Explanation_Identification_Prediciton')
def main():

    parser.add_argument('--word_dim', type=int, default=25, help='the dimension of the word embedding to be used. Default=25')
    parser.add_argument('--hidden_dim', type=int, default=25,
                        help='the dimension of the hidden layer to be used. Default=25')
    parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=1')
    parser.add_argument('--target_data', type=str, default='./preprocessed_embeddings/',
                        help='the path of target data')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')

    parser.add_argument('--causality_num_direction', type=int, default=2,
                        help='# of direction of RNN for causality detection, Default=2 (bidirectional)')
    parser.add_argument('--prediction_csv', type=str, default='prediction.csv')
    parser.add_argument('--start_idx', type=int, default='0')
    parser.add_argument('--CE_classifier', type=str, default='./Performance_Logs/')
    parser.add_argument('--dataset', type=str, default='dataset')

    opt = parser.parse_args()
    print(opt)
    word_embedding_dim=opt.word_dim
    hidden_dim=opt.hidden_dim
    c_num=opt.causality_num_direction

    if opt.prediction_csv != 'prediction.csv':
        predict_file = open(opt.prediction_csv,'w')
    else:
        predict_file = open(opt.dataset+'_prediction.csv','w')


    is_cuda= opt.cuda
    torch.manual_seed(opt.seed)
    if is_cuda:
        torch.cuda.manual_seed(opt.seed)



    # word embedding pickle
    target_word_embed_seqs = pickle.load(open(opt.target_data, "rb"))



    model = BiLSTM_Causal_Explanation_Identification(word_embedding_dim, hidden_dim, is_cuda, causality_num_direction=c_num)
    model.load_state_dict(torch.load(opt.CE_classifier))

    if is_cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    start=time.time()
    predict_csv = csv.writer(predict_file)
    predict_csv.writerow(['message_id', 'prediction'])
    model.eval()
    twt_cnt=opt.start_idx+1 # consider that index is N-1
    for i in range(opt.start_idx,len(target_word_embed_seqs)):
        arg_cnt=0
        causality_score = model(target_word_embed_seqs[i])
        if causality_score is None:
            print("Problem occurred at message: "+ str(twt_cnt) +"!")
            twt_cnt += 1
            continue
        _, predictions = causality_score.max(1)
        
        for prediction in predictions:
            predict_csv.writerow([str(twt_cnt)+'-'+str(arg_cnt), int(prediction)])
            print(str(twt_cnt)+'-'+str(arg_cnt)+","+str(int(prediction)))
            arg_cnt+=1

        twt_cnt+=1



    predict_file.close()
    end_time=time_since(start)
    print("Done. Prediction Time: %s"%(end_time))












if __name__=="__main__":
    main()
