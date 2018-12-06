# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:48:29 2018

@author: dhruv
"""
import os
import re
from nltk.tokenize import word_tokenize
import collections
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from glove import Glove

train_data_path = '../arxiv/papers/'
train_summary_path = '../arxiv/summary/'

valid_data_path = '../BBC News Summary/valid/News Articles/'
valid_summary_path = '../BBC News Summary/valid/Summaries/'

glove_path = '../glove.model'

def read_files(data_path):
    data_folders = os.listdir(data_path)
    data = []
    
    for folder in data_folders:
        sub_folder = data_path+folder
        articles = os.listdir(sub_folder)
        for article in articles:
            article_path = sub_folder+'/'+article
            with open(article_path,'r', encoding="utf8", errors='ignore') as f:
                data.append(f.read())
    
    return data

def build_dict(step, toy=False):
    if step == "train":
        articles = read_files(train_data_path)
        summaries = read_files(train_summary_path)
#        train_article_list = get_text_list(train_article_path, toy)
#        train_title_list = get_text_list(train_title_path, toy)
        print("files  read")
        words = list()
        for sentence in articles + summaries:
            for word in word_tokenize(sentence):
                words.append(word)

#        word_counter = collections.Counter(words).most_common()
        print("loading glove")
        glove = Glove.load(glove_path)
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        print("preparing dict")
        for word, _ in glove.dictionary.items():
            if(word in word_dict):
                print (word)
            word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    elif step == "valid":
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    article_max_len = 600
    summary_max_len = 250

    return word_dict, reversed_dict, article_max_len, summary_max_len


def build_dataset(step, word_dict, article_max_len, summary_max_len, toy=False):
    if step == "train":
        articles = read_files(train_data_path)
        summaries = read_files(train_summary_path)
#        article_list = get_text_list(train_article_path, toy)
#        title_list = get_text_list(train_title_path, toy)
    elif step == "valid":
        articles = read_files(valid_data_path)
    else:
        raise NotImplementedError

    x = [word_tokenize(d) for d in articles]
    x = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in x]
    x = [d[:article_max_len] for d in x]
    x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in x]
    
    if step == "valid":
        return x
    else:        
        y = [word_tokenize(d) for d in summaries]
        y = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in y]
        y = [d[:(summary_max_len - 1)] for d in y]
        return x, y


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def get_init_embedding(reversed_dict, embedding_size):
    glove_embed = Glove.load(glove_path)    
    glove_vectors = glove_embed.word_vectors
    glove_dictionary = glove_embed.dictionary
    
#    glove_file = "glove/glove.42B.300d.txt"
#    word2vec_file = get_tmpfile("word2vec_format.vec")
#    glove2word2vec(glove_file, word2vec_file)
#    print("Loading Glove vectors...")
#    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    word_vec_list = list()
    for word, ind in sorted(glove_dictionary.items()):
        try:
            word_vec = glove_vectors[ind]
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    # Assign random vector to <s>, </s> token
    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)

    return np.array(word_vec_list)
    
#word_dict, rev_dict, art, summ = build_dict(step="train")