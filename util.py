#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from stemmer import Stemmer
import re,string

###########################
#                         #
#    FEATURE EXTRACTION   #
#                         #
###########################

def tweet_cleaner(tweet, stopwords = None, s = None):
    if stopwords is None:
        stopwords = pd.read_csv( 'datasets/stopwords_nltk.csv', sep=',', index_col=None, header=None)
        stopwords =  np.concatenate((stopwords.iloc[:,0].values, ['AT_USER', 'URL']))
    if s is None:
        s = Stemmer()
    tweet = tweet.lower() # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet) # remove URLs
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet) # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet) # remove the # in #hashtag
    tweet = tweet.translate(str.maketrans(string.punctuation,len(string.punctuation)*' ')) # remove punctuation
    tweet = re.findall(r'[a-z]+', tweet) # split words
    tweet = [s.stem(word) for word in tweet] # stemming words
    tweet = [word for word in tweet if word not in stopwords] # removing stopwords
    return tweet

def tweets_cleaner(tweets, stopwords = None, s = None):
    cleaned_tweets = []
    if stopwords is None:
        stopwords = pd.read_csv( 'datasets/stopwords_nltk.csv', sep=',', index_col=None, header=None)
        stopwords =  np.concatenate((stopwords.iloc[:,0].values, ['AT_USER', 'URL']))
    if s is None:
        s = Stemmer()
    for i in tweets:
        cleaned_tweets.append(tweet_cleaner(i, stopwords, s))
    return cleaned_tweets

def build_vocab(cleaned_tweets):
    vocab = []
    
    for i in cleaned_tweets:
        vocab.extend(i)
        
    vocab = np.sort(list(set(vocab)))
    
    return vocab

def build_features(tweet, vocab):
    features = np.zeros(len(vocab))
    unique, counts = np.unique(tweet, return_counts=True)
    features[np.searchsorted(vocab, unique).astype(int)] = counts
    #np.insert(features,np.searchsorted(vocab,unique),counts)
    #features = np.isin(vocab, tweet).astype(int)
    return features.astype(int)

def build_representation(tweets, vocab):
    data = []
    for idx, i in enumerate(tweets):
        data.append(build_features(i,vocab))
    return data

###########################
#                         #
#    MODEL VALIDATION     #
#                         #
###########################

def stratified_holdOut(target, pTrain):
    """
    Retorna os indices dos dados de treinamento e teste 
    
    Parâmetros
    ----------   
    target: vetor com as classes dos dados
    
    pTrain: porcentagem de dados de treinamento
    
    Retorno
    -------
    train_index: índices dos dados de treinamento 
    test_index: índices dos dados de teste 
    
    """
    
    # inicializa as variaveis que precisam ser retornadas 
    train_index = []
    test_index = []

    ########################## COMPLETE O CÓDIGO AQUI  ###############################
    #  Instruções: Complete o codigo para retornar os índices dos dados de  
    #              treinamento e dos dados de teste.
    #              
    #              Obs: - os conjuntos de treinamento e teste devem ser criados
    #                     de maneira estratificada, ou seja, deve ser mantida a 
    #                     a distribuição original dos dados de cada classe em cada 
    #                     conjunto. Em outras palavras, o conjunto de treinamento deve ser
    #                     formado por pTrain% dos dados da primeira classe, pTrain% dos dados da 
    #                     segunda classe e assim por diante. 
    #                   - a porcentagem de dados de teste para cada classe é igual a 
    #                     1-pTrain (parametro da funcao que contem a porcentagem de dados 
    #                     de treinamento)

    classes = np.unique(target)
    for n in classes:
        a = (np.where(target == n))[0]
        q = int((len(a)) * pTrain)
        train_index = np.concatenate((train_index, a[0:q]))
    train_index = np.sort(train_index.astype(int))
    test_index = np.sort(np.setdiff1d(np.arange(len(target)),train_index))
    ##################################################################################
    
    return train_index, test_index


def get_confusionMatrix(Y_test, Y_pred, classes):
    """
    Retorna a matriz de confusao, onde o numero de linhas e 
        e numero de colunas e igual ao numero de classes
        
    Parametros
    ----------   
    Y_test: vetor com as classes verdadeiras dos dados
    
    Y_pred: vetor com as classes preditas pelo metodo de classificacao
    
    classes: classes do problema
    
    
    Retorno
    -------
    cm: matriz de confusao (array numpy, em que o numero de linhas e de colunas
        e igual ao numero de classes)
    
    """
    
    # inicia a matriz de confusão
    cm = np.zeros( [len(classes),len(classes)], dtype=int )

    ########################## COMPLETE O CÓDIGO AQUI  ###############################
    #  Instruções: Complete o codigo para retornar a matriz de confusao baseada nas
    #           classes verdadeiras dos dados versus as classes preditas pelo metodo
    #           de classificacao. 
    #
    #           Obs: cuidado com a ordem das classes na geracao da matriz de confusao.  
    #                Os valores da i_esima linha da matriz de confusao devem ser calculados
    #                com base na i-esima classe do vetor "classes" que é passado como 
    #                parametro da funcao. Os valores da j-esima coluna da matriz de confusao 
    #                devem ser calculados com base na j-esima classe do vetor "classes". 
    #                
    #
    
    
    for i in classes:
        for j in classes:
            cm[i][j] = np.count_nonzero(np.logical_and(Y_test == i, Y_pred == j))

    ##########################################################################
    
    return cm

def relatorioDesempenho(matriz_confusao, classes, imprimeRelatorio=False):
  """
  Funcao usada calcular as medidas de desempenho da classificação.
  
  Parametros
  ----------   
  matriz_confusao: array numpy que representa a matriz de confusao 
                   obtida na classificacao. O numero de linhas e de colunas
                   dessa matriz e igual ao numero de classes.
    
  classes: classes do problema
  
  imprimeRelatorio: variavel booleana que indica se o relatorio de desempenho
                    deve ser impresso ou nao. 
     
  Retorno
  -------
  resultados: variavel do tipo dicionario (dictionary). As chaves
              desse dicionario serao os nomes das medidas de desempenho; os valores
              para cada chave serao as medidas de desempenho calculadas na funcao.
              
              Mais especificamente, o dicionario devera conter as seguintes chaves:
              
               - acuracia: valor entre 0 e 1 
               - revocacao: um vetor contendo a revocacao obtida em relacao a cada classe
                            do problema
               - precisao: um vetor contendo a precisao obtida em relacao a cada classe
                            do problema
               - fmedida: um vetor contendo a F-medida obtida em relacao a cada classe
                            do problema
               - revocacao_macroAverage: valor entre 0 e 1
               - precisao_macroAverage: valor entre 0 e 1
               - fmedida_macroAverage: valor entre 0 e 1
               - revocacao_microAverage: valor entre 0 e 1
               - precisao_microAverage: valor entre 0 e 1
               - fmedida_microAverage: valor entre 0 e 1
  """

  n_teste = sum(sum(matriz_confusao))
  
  nClasses = len( matriz_confusao ) #numero de classes
    
  # inicializa as medidas que deverao ser calculadas
  vp=np.zeros( nClasses ) # quantidade de verdadeiros positivos
  vn=np.zeros( nClasses ) # quantidade de verdadeiros negativos
  fp=np.zeros( nClasses ) # quantidade de falsos positivos
  fn=np.zeros( nClasses ) # quantidade de falsos negativos

  acuracia = 0.0 
  
  revocacao = np.zeros( nClasses ) # nesse vetor, devera ser guardada a revocacao para cada uma das classes
  revocacao_macroAverage = 0.0
  revocacao_microAverage = 0.0
    
  precisao = np.zeros( nClasses ) # nesse vetor, devera ser guardada a revocacao para cada uma das classes
  precisao_macroAverage = 0.0
  precisao_microAverage = 0.0

  fmedida = np.zeros( nClasses ) # nesse vetor, devera ser guardada a revocacao para cada uma das classes
  fmedida_macroAverage = 0.0
  fmedida_microAverage = 0.0

  ########################## COMPLETE O CÓDIGO AQUI  ###############################
  #  Instrucoes: Complete o codigo para calcular as seguintes medidas 
  #              de desempenho: acuracia, revocacao, precisao e F-medida.
  #              Para as medidas revocacao, precisao e F-medida, voce
  #              devera obter o valor correspondente a cada uma das classes.
  #              Voce também precisara calcular as medias macro e micro das 
  #              medidas revocacao, precisao e F-medida.
  #              
  #              Obs: voce deve calcular a quantidade de verdadeiros/falsos positivos e  
  #              verdadeiros/falsos negativos em relacao a cada classe e usar esses 
  #              valores para calcular todas as medidas de desempenho. 
    
  vp = np.diag(matriz_confusao)
  fp = np.sum(matriz_confusao, axis=0) - vp
  fn = np.sum(matriz_confusao, axis=1) - vp
  vn = np.repeat(n_teste, nClasses) - fp - fn - vp
    
  acuracia = np.sum(vp + vn) / (n_teste * nClasses)
  precisao = vp / (vp + fp)
  revocacao = vp / (vp + fn)
  fmedida = 2 * (precisao * revocacao) / (precisao + revocacao)
    
  precisao_macroAverage = 1 / nClasses * np.sum(precisao)
  revocacao_macroAverage = 1 / nClasses * np.sum(revocacao)
  fmedida_macroAverage = 2 * (precisao_macroAverage * revocacao_macroAverage) / (precisao_macroAverage + revocacao_macroAverage)
    
  precisao_microAverage = np.sum(vp) / np.sum(vp + fp)
  revocacao_microAverage = np.sum(vp) / np.sum(vp + fn)
  fmedida_microAverage = 2 * (precisao_microAverage * revocacao_microAverage) / (precisao_microAverage + revocacao_microAverage)  
  ##################################################################################
    
    
    
    
  # imprimindo os resultados para cada classe
  if imprimeRelatorio:
        
      print('\n\tRevocacao   Precisao   F-medida   Classe')
      for i in range(0,nClasses):
        print('\t%1.3f       %1.3f      %1.3f      %s' % (revocacao[i], precisao[i], fmedida[i],classes[i] ) )
    
      print('\t------------------------------------------------');
      
      #imprime as médias
      print('\t%1.3f       %1.3f      %1.3f      Média macro' % (revocacao_macroAverage, precisao_macroAverage, fmedida_macroAverage) )
      print('\t%1.3f       %1.3f      %1.3f      Média micro\n' % (revocacao_microAverage, precisao_microAverage, fmedida_microAverage) )
    
      print('\tAcuracia: %1.3f' %acuracia)
      
    
  # guarda os resultados em uma estrutura tipo dicionario
  resultados = {'revocacao': revocacao, 'acuracia': acuracia, 'precisao':precisao, 'fmedida':fmedida}
  resultados.update({'revocacao_macroAverage':revocacao_macroAverage, 'precisao_macroAverage':precisao_macroAverage, 'fmedida_macroAverage':fmedida_macroAverage})
  resultados.update({'revocacao_microAverage':revocacao_microAverage, 'precisao_microAverage':precisao_microAverage, 'fmedida_microAverage':fmedida_microAverage})
  resultados.update({'confusionMatrix': matriz_confusao})

  return resultados 


def gridSearch(X, Y, Xval, Yval):
    """
    Retorna o melhor valor para os parametros lamba da regularizacao da Regressao Logistica.
    
    Parametros
    ----------
    X : matriz com os dados de treinamento
    
    Y : vetor com as classes dos dados de treinamento
    
    Xval : matriz com os dados de validacao
    
    Yval : vetor com as classes dos dados de validacao
    
    Retorno
    -------
    bestReg: o melhor valor para o parametro de regularizacao
    
    """
    
    # inicializa a variável que deverá ser retornada pela função
    bestReg = -100
    
    # valores que deverao ser testados para o parametro de regularizacao 
    reg = [0,0.5,1,10,50,100];
        
    ########################## COMPLETE O CÓDIGO AQUI  ###############################
    # Instrucoes: Complete esta função para retornar os melhores valores do parametro
    #             de regularizacao da regressao Logistica. 
    #
    #             Você pode calcular o desempenho do classificador atraves da funcao
    #             relatorioDesempenho() criada anteriormente. Use a acuracia para decidir
    #             o melhor parametro.            
    #
    
    perf = []
    
    def regmax(i):
        theta = treinamento(X, Y, i, iteracoes)
        p = predicao(Xval, theta)
        cm = get_confusionMatrix(Yval, p, [0,1])
        results = relatorioDesempenho(cm, [0,1])
        return results['acuracia']
    
    vregmax = np.vectorize(regmax, otypes=['float'])
    perf = vregmax(reg)

    '''
    for i in reg:
        theta = treinamento(X, Y, i, iteracoes)
        p = predicao(Xval, theta)
        cm = get_confusionMatrix(Yval, p, [0,1])
        results = relatorioDesempenho(cm, [0,1])
        perf.append(results['acuracia'])
    '''
        
    bestReg = reg[np.argmax(perf)]
    ################################################################################## 

    return bestReg

def stratified_kfolds(target, k, classes):
    """
    Retorna os indices dos dados de treinamento e teste para cada uma das k rodadas 
    
    Parametros
    ----------   
    target: vetor com as classes dos dados
    
    k: quantidade de folds 
    
    Retorno
    -------
    folds_final: os indices dos dados de treinamento e teste para cada uma das k rodadas 
    
    """

    # Inicializa a variavel que precisa ser retornada. 
    # Cada elemento do vetor folds_final deve ser outro vetor de duas posicoes: a primeira
    #    posicao deve conter um vetor com os indices de treinamento relativos ao i-esimo fold;
    #    a segunda posicao deve conter um vetor com os indices de teste relativos ao i-esimo fold.
    folds_final = np.zeros( k,dtype='object')

    # inicializa o vetor onde o k-esimo elemento guarda os indices dos dados de treinamento 
    # relativos ao k-esimo fold 
    train_index = np.zeros( k,dtype='object')
    
    # inicializa o vetor onde o k-esimo elemento guarda os indices dos dados de teste 
    # relativos ao k-esimo fold 
    test_index = np.zeros( k,dtype='object')
    
    # inicializa cada posicao do vetor folds_final que devera ser retornado pela funcao
    for i in folds_final:
        
        train_index[i] = [] # indices dos dados de treinamento relativos ao fold i
        test_index[i] = [] # indices dos dados de teste relativos ao fold i
        
        # inicializa o i-esimo elemento do vetor que devera ser retornado
        folds_final[i] = np.array( [train_index[i],test_index[i]] ) 
      
    

    ########################## COMPLETE O CÓDIGO AQUI  ###############################
    #  Instrucoes: Complete o codigo para retornar os indices dos dados de  
    #              treinamento e dos dados de teste para cada rodada do k-folds.
    #              
    #              Obs: - os conjuntos de treinamento e teste devem ser criados
    #                     de maneira estratificada, ou seja, deve ser mantida a 
    #                     a proporcao original dos dados de cada classe em cada 
    #                     conjunto.
    #                   - Para cada rodada k, os dados da k-esima particao devem compor 
    #                     os dados de teste, enquanto os dados das outras particoes devem 
    #                     compor os dados de treinamento.
    #                   - voce devera retornar a variavel folds_final: essa variavel e uma 
    #                     vetor de k posicoes. Cada posicao k do vetor folds_final deve conter 
    #                     outro vetor de duas posicoes: a primeira posicao deve conter um vetor 
    #                     com os indices de treinamento relativos ao k-esimo fold; a segunda posicao 
    #                     deve conter um vetor com os indices de teste relativos ao k-esimo fold. 

    class_indices = [(np.where(target == i))[0] for i in classes]
        
    for i in np.arange(k):
        test_index[i] = []
        for n in classes:
            shift = int(len(class_indices[n]) / k)
            test_index[i] = np.concatenate((test_index[i], class_indices[n][i*shift:(i+1)*shift]))
        
        test_index[i] = np.sort(test_index[i].astype(int))
        train_index[i] = np.sort(np.setdiff1d(np.arange(len(target)),test_index[i]))
        folds_final[i] = [train_index[i], test_index[i]]
    ##################################################################################
    
    return folds_final

def mediaFolds( resultados, classes ):
    
    nClasses = len(classes)
    
    acuracia = np.zeros( len(resultados) )

    revocacao = np.zeros( [len(resultados),len(classes)] )
    precisao = np.zeros( [len(resultados),len(classes)] )
    fmedida = np.zeros( [len(resultados),len(classes)] )

    revocacao_macroAverage = np.zeros( len(resultados) )
    precisao_macroAverage = np.zeros( len(resultados) )
    fmedida_macroAverage = np.zeros( len(resultados) )

    revocacao_microAverage = np.zeros( len(resultados) )
    precisao_microAverage = np.zeros( len(resultados) )
    fmedida_microAverage = np.zeros( len(resultados) )


    for i in range(len(resultados)):
        acuracia[i] = resultados[i]['acuracia']
        
        revocacao[i,:] = resultados[i]['revocacao']
        precisao[i,:] = resultados[i]['precisao']
        fmedida[i,:] = resultados[i]['fmedida']

        revocacao_macroAverage[i] = resultados[i]['revocacao_macroAverage']
        precisao_macroAverage[i] = resultados[i]['precisao_macroAverage']
        fmedida_macroAverage[i] = resultados[i]['fmedida_macroAverage']

        revocacao_microAverage[i] = resultados[i]['revocacao_microAverage']
        precisao_microAverage[i] = resultados[i]['precisao_microAverage']
        fmedida_microAverage[i] = resultados[i]['fmedida_microAverage']
        
    # imprimindo os resultados para cada classe
    print('\n\tRevocacao   Precisao   F-medida   Classe')
    for i in range(0,nClasses):
        print('\t%1.3f       %1.3f      %1.3f      %s' % (np.mean(revocacao[:,i]), np.mean(precisao[:,i]), np.mean(fmedida[:,i]), classes[i] ) )

    print('\t---------------------------------------------------------------------')
  
    #imprime as medias
    print('\t%1.3f       %1.3f      %1.3f      Média macro' % (np.mean(revocacao_macroAverage), np.mean(precisao_macroAverage), np.mean(fmedida_macroAverage)) )
    print('\t%1.3f       %1.3f      %1.3f      Média micro\n' % (np.mean(revocacao_microAverage), np.mean(precisao_microAverage), np.mean(fmedida_microAverage)) )

    print('\tAcuracia: %1.3f' %np.mean(acuracia))
