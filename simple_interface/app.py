import numpy as np #importa a biblioteca usada para trabalhar com vetores e matrizes
import pandas as pd #importa a biblioteca usada para trabalhar com dataframes
import util
from scipy.sparse import bsr_matrix

import svmutil
from svmutil import svm_read_problem
from svmutil import svm_problem
from svmutil import svm_parameter
from svmutil import svm_train
from svmutil import svm_predict
from svmutil import svm_save_model, svm_load_model

from flask import Flask,  render_template, request
app = Flask(__name__)

model = svm_load_model('libsvm.model')
vocab = pd.read_csv('vocab.csv', sep=',', header=None, squeeze=True).values

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']
        features = util.extract_features_single(tweet, vocab, rep = 'ngrams', n=3)
        print(np.argwhere(features > 0))
        pred, p_acc, p_vals = svm_predict([], bsr_matrix(features), model)
        print(pred)
        return render_template('result.html', pred=pred[0])
