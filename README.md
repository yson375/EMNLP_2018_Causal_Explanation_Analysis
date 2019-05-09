# EMNLP 2018 Causal Explanation Analysis on Social Media

Codes and examples for the paper [Causal Explanation Analysis on Social Media](https://arxiv.org/pdf/1809.01202.pdf) published in EMNLP 2018


## Dependencies
- [DLATK](https://github.com/dlatk/dlatk)
- [Pytorch](https://pytorch.org/)
- [TweeboParser](https://github.com/ikekonglp/TweeboParser)
- [mysqlclient](https://github.com/PyMySQL/mysqlclient-python)
- [NumPy](http://www.numpy.org)
- [scikit-learn](http://www.scikit-learn.org/)
- [SciPy](http://www.scipy.org/)
- [GloVe Pretrained Word Vectors](http://nlp.stanford.edu/data/glove.twitter.27B.zip)

## General Information
### 1. Status
Currently, this repository only contains the codes and examples for argument extraction and causal explanation identification.
The Linear SVM model for causality prediction is implemented in 'DLATK' and needs more work for more automated pipeline connection (e.g., importing the feature extraction results into DLATK).
I will keep updating this repository until I complete the fully automated end-to-end pipeline.

### 2. Pretrained Models
Pretrained models and preprocessed embeddings are uploaded separtely on [my website](https://www3.cs.stonybrook.edu/~yson/)

### 3. Gold Standard for Causal Explanation Discourse Arguments as in the Paper
If you want to use the performance metrics reported on our paper, use the label files in "ArgumentExtractionFiles/CE_Labels_for_Args"

## Pipeline
### 1. Argument Extraction:
- Tweet Depedency Parsing with TweeboParser:

  Go to the directory of TweeboParser and run:
```bash run.sh EMNLP_2018_CE_Training_message.txt```

- Run ```ArgumentExtractor.py``` on the dependency results:

  ```python3 ArgumentExtractor.py EMNLP_2018_CE_Training_message.txt.predict```
  
### 2. Causality Prediction:
- Extract features from argument files
- Train / test DLATK Linear SVM model

### 3. Causal Explanation Identification:
- Take the predicted causality messages from the Linear SVM model

- Preprocess word embeddings:

  e.g., ```python3 ./preprocessing/Prepare_DA_WordEmbeddings_Seqs.py ./ArgumentExtractionFiles/EMNLP_2018_CE_Training_message.txt.predict.args.csv Training```

#### Training
- Load causal explanation label vectors for Pytorch

  e.g., ```python3 ./preprocessing/Prepare_Label_Vecs.py ./ArgumentExtractionFiles/CE_Labels_for_Args/EMNLP_2018_CE_Training_message.txt.predict.args_ce_labels.csv Training```

- Set up your configuration and start training:

  e.g., ```python3 cei_train.py --word_dim 200 --hidden_dim 200 --lr 0.001 --grad SGD --cuda --model_path ./Saved_Models/ --train_data ./preprocessing/causal_explanation_200_da_embedding_seqs_Training.list --train_labels ./preprocessing/causal_explanation_da_labels_Training.list --valid_data ./preprocessing/causal_explanation_200_da_embedding_seqs_Validation.list --valid_labels ./preprocessing/causal_explanation_da_labels_Validation.list```

#### Prediction
- Predict causal explanation discourse arguments:

  e.g., (our best predicting model): ```python3 cei_predict.py --word_dim 200 --hidden_dim 200 --prediction_csv CEI_subset_prediction.csv --target_data ./preprocessing/causal_explanation_200_da_embedding_seqs_Test.list --CE_classifier ./Pretrained_Models/CE_200_200_lr0_001_SGD_dir_2_f1_0.84618_epoch_180_Dropout_0_3_early_stop_saved.ptstdict```


  



