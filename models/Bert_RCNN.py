# coding: utf-8
__author__='zhoubin'
__mtime__='20210604'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertTokenizerFast
from collections import OrderedDict



class Config(object):

    """配置参数"""
    def __init__(self, dataset, char_embedding_sg, word_embedding_tc):
        self.model_name = 'Bert_RCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_char_path = dataset + '/data/vocab_chars.pkl'                     # 字表
        self.vocab_word_path = dataset + '/data/vocab_words.pkl'                     # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained_char = torch.tensor(
            np.load(dataset + '/data/' + char_embedding_sg)["embeddings"].astype('float32'))\
            if char_embedding_sg != 'random' else None                               # 预训练字向量
        self.embedding_pretrained_word = torch.tensor(
            np.load(dataset + '/data/' + word_embedding_tc)["embeddings"].astype('float32')) \
            if char_embedding_sg != 'random' else None                               # 预训练词向量
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 1.0                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        # self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.bert_pad_size = 36                                         #即seqlen
        self.char_pad_size = 32                                         # 每句话处理成字的长度(短填长切)
        self.word_pad_size = 16                                         # 每句话处理成词的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed_char = self.embedding_pretrained_char.size(1)\
            if self.embedding_pretrained_char is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.embed_word = self.embedding_pretrained_word.size(1) \
            if self.embedding_pretrained_word is not None else 200  # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 1                                             # lstm层数
        self.bert_path = '/Users/zhoubin/pretrained/macbert-chinese'
        self.tokenizer_bert = BertTokenizerFast.from_pretrained(self.bert_path)
        self.bert_embed_dim = 768
        self.label_one_hot = True
        self.loss_func = torch.nn.MultiLabelSoftMarginLoss()
        self.topk = 5


'''Recurrent Convolutional Neural Networks for Text Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 预训练bert加载
        if config.bert_path:
            self.bert = BertModel.from_pretrained(config.bert_path)
            #固定bert参数不变，只下游任务做梯度更新
#             for p in self.bert.parameters():
#                 p.requires_grad = False
        # 预训练字词向量加载
        if config.embedding_pretrained_char is not None:
            self.embedding_char = nn.Embedding.from_pretrained(config.embedding_pretrained_char, freeze=False)
            self.embedding_word = nn.Embedding.from_pretrained(config.embedding_pretrained_word, freeze=False)
        else:
            # self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
            raise ValueError('pretrained embedding should provide here!')
        # bert
        self.fc_b = nn.Linear(config.bert_embed_dim, config.num_classes)
        # char
        self.lstm_c = nn.LSTM(config.embed_char, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool_c = nn.MaxPool1d(config.char_pad_size)
        self.fc_c = nn.Linear(config.hidden_size * 2 + config.embed_char, config.num_classes)
        # word
        self.lstm_w = nn.LSTM(config.embed_word, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpool_w = nn.MaxPool1d(config.word_pad_size)
        self.fc_w = nn.Linear(config.hidden_size * 2 + config.embed_word, config.num_classes)


    def forward(self, x):
        # x为(x_bert, x_att_mask, seq_len_bert, x_word, seq_len_word, x_char, seq_len_char)
        xb,xatt,_,xw,_,xc,_ = x
        #bert分支
        bert_output = self.bert(input_ids=xb, attention_mask=xatt)
        embeds, pooled_output = bert_output[0], bert_output[1]
        #rcnn分支
        embed_c = self.embedding_char(xc)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        embed_w = self.embedding_word(xw)  # [batch_size, seq_len, embeding]=[128, 16, 200]
        out, _ = self.lstm_c(embed_c)
        out2, _ = self.lstm_w(embed_w)
        out = torch.cat((embed_c, out), 2)
        out2 = torch.cat((embed_w, out2), 2)
        out = F.relu(out)
        out2 = F.relu(out2)
        out = out.permute(0, 2, 1)
        out2 = out2.permute(0, 2, 1)
        out = self.maxpool_c(out).squeeze()
        out2 = self.maxpool_w(out2).squeeze()
        out = self.fc_c(out)
        out2 = self.fc_w(out2)
        final_out = (out+out2)/2
        return final_out
