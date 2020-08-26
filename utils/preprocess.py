import numpy as np
import pandas as pd
import re
from jieba import posseg
import jieba
from tokenizer import segment
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)

def load_stop_words(stopword_path):
    '''
    加载停用词
    :param stopword_path:
    :return:
    '''
    file = open(stopword_path,'r',encoding='utf-8')
    stop_words = file.readlines()
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words

stop_words_list = load_stop_words(f'{BASE_DIR}/哈工大停用词表.txt')
# print(f"停用词已加载: {len(stop_words_list)}")
# stop_words_list = ["，", "|", "[", "]" , "。", ",", "."]

def filter_sentence(words_list):
    '''
    过滤停用词
    :param words: 切好词的列表 [word1 ,word2 .......]
    :param stop_words: 停用词列表
    :return: 过滤后的停用词
    '''
    return [word for word in words_list if word not in stop_words_list]

def preprocess_sentence(sentence):
    seg_list = segment(sentence.strip(), cut_type='word')
    seg_list = filter_sentence(seg_list)
    seg_line = ' '.join(seg_list)
    return seg_line

def parse_data(train_path, test_path):
    train_df = pd.read_csv(train_path, encoding='utf-8')
    # train_df.dropna(subset=['Report'], how='any', inplace=True)
    train_df.fillna('', inplace=True)
    # 将所有车型名字加入词典
    for model in train_df.Model:
        if model:
            jieba.add_word(model.lower())
    print("train model name added")

    train_x = train_df.Question.str.cat(train_df.Dialogue)
    # print('train_x is ', len(train_x))
    train_x = train_x.apply(preprocess_sentence)
    print('train_x is ', len(train_x))
    train_y = train_df.Report
    # print('train_y is ', len(train_y))
    train_y = train_y.apply(preprocess_sentence)
    print('train_y is ', len(train_y))
    # if 'Report' in train_df.columns:
        # train_y = train_df.Report
        # print('train_y is ', len(train_y))
    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df.fillna('', inplace=True)
    for model in test_df.Model:
        if model:
            jieba.add_word(model.lower())
    print("test model name added")
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_x = test_x.apply(preprocess_sentence)
    print('test_x is ', len(test_x))

    train_x.to_csv('{}/data/train_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)
    train_y.to_csv('{}/data/train_set.seg_y.txt'.format(BASE_DIR), index=None, header=False)
    test_x.to_csv('{}/data/test_set.seg_x.txt'.format(BASE_DIR), index=None, header=False)

if __name__ == '__main__':
    # jieba.load_userdict(f"{BASE_DIR}/user_dict.txt")

    # 需要更换成自己数据的存储地址
    parse_data('{}/data/AutoMaster_TrainSet.csv'.format(BASE_DIR),
               '{}/data/AutoMaster_TestSet.csv'.format(BASE_DIR))



