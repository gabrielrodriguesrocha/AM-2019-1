import matplotlib.pyplot as plt

import numpy as np #importa a biblioteca usada para trabalhar com vetores e matrizes
#np.set_printoptions(threshold=np.inf)
import pandas as pd #importa a biblioteca usada para trabalhar com dataframes
import util
import sys
from scipy.sparse import bsr_matrix
#importa o arquivo e extrai as features
Xfeatures, Y, vocab = util.extract_features('datasets/everything.csv', rep='ngrams',n=3)

import svmutil
from svmutil import svm_read_problem
from svmutil import svm_problem
from svmutil import svm_parameter
from svmutil import svm_train
from svmutil import svm_predict
from svmutil import svm_save_model

t_features = util.tweet_preProcess(sys.argv[1], vocab, rep='ngrams', n = 3)
# semente usada na randomizacao dos dados.
randomSeed = 10 

# gera os indices aleatorios que irao definir a ordem dos dados
idx_perm = np.random.RandomState(randomSeed).permutation(range(len(Y)))

# ordena os dados de acordo com os indices gerados aleatoriamente
Xk, Yk = Xfeatures[idx_perm, :], Y[idx_perm]

# define a porcentagem de dados que irao compor o conjunto de treinamento
pTrain = 0.8 

# obtem os indices dos dados da particao de treinamento e da particao de teste
train_index, test_index = util.stratified_holdOut(Yk, pTrain)

X_train, X_test = Xk[train_index, :], Xk[test_index, :];
Y_train, Y_test = Yk[train_index], Yk[test_index];

train_index, val_index = util.stratified_holdOut(Y_train, pTrain)

Xtrain, Xvalid = X_train[train_index, :], X_train[val_index, :]
Ytrain, Yvalid = Y_train[train_index], Y_train[val_index]


### Classificador - Kernel radial ### 
# treina o classificador com o melhor custo e o melhor gamma encontrados 
# Treinamento e clasificacao com valores de kernel:

# 0 -- linear: u\'\*v
# 1 -- polynomial: (gamma\*u\'\*v + coef0)^degree
# 2 -- radial basis function: exp(-gamma\*|u-v|^2)

# Kernel raidal:
model = svm_train(Ytrain, bsr_matrix(Xtrain), '-q -c %f -t %d -g %f' %(1, 2, 0.031))

p_labs, p_acc, p_vals = svm_predict([], bsr_matrix(t_features), model, options='-q')
print(p_labs)