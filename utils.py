# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import jieba
jieba.initialize()

# MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(files_path, tokenizer, max_size, min_freq):
    #max_size取None则表示不指定max size大小
    vocab_dic = {}
    if isinstance(files_path, str):
        raise ValueError('please input paths in list format!')
    for file_path in files_path:
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content = lin.split('\t')[0]
                for word in tokenizer(content.replace(' ','')):
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_list = [(PAD, -1), (UNK, -1)] + vocab_list
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    return vocab_dic


def build_dataset(config, use_word, min_freq=1, max_size=None):
    if use_word:
        # tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
        tokenizer = jieba.lcut  # jieba分词 用腾讯词向量
    else:
        tokenizer = lambda x: [y for y in x]  # char-level 用搜狗字向量
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=max_size, min_freq=min_freq)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []
                token = tokenizer(content.replace(' ',''))
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test

def build_ds_full(config, min_freq=1, max_size=None):
    '''
    该方法会将句子用不同的方法进行tokenize，
    1.bert tokenizer(roberta, macbert, longformer4096等)
    2.jieba cut 用腾讯词向量
    3.char-level 用搜狗字向量
    :return:
    '''
    tokenizer_bert = config.tokenizer_bert #bert tokenizer
    tokenizer_word = jieba.lcut  # jieba分词 用腾讯词向量
    tokenizer_char = lambda x: [y for y in x]  # char-level 用搜狗字向量
    if not os.path.exists(config.vocab_word_path) or not os.path.exists(config.vocab_char_path):
        raise ValueError('vocab path should provide!')

    vocab_word = pkl.load(open(config.vocab_word_path, 'rb'))
    vocab_char = pkl.load(open(config.vocab_char_path, 'rb'))

    def _prepare_bert_ids(text, tokenizer_bert, seqlen, mask_padding_with_zero=True):
        def tok(s):
            return tokenizer_bert.tokenize(s)
        tokens = [tokenizer_bert.cls_token] + tok(text)
        token_ids = tokenizer_bert.convert_tokens_to_ids(tokens)
        token_ids = token_ids[:seqlen-1] +[tokenizer_bert.sep_token_id]
        input_len = len(token_ids)
        attention_mask = [1 if mask_padding_with_zero else 0] * input_len
        padding_length = seqlen - input_len
        token_ids = token_ids + ([tokenizer_bert.pad_token_id] * padding_length)

        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

        assert len(token_ids) == seqlen, "Error with input length {} vs {}".format(
            len(token_ids), seqlen
        )
        assert len(attention_mask) == seqlen, "Error with input length {} vs {}".format(
            len(attention_mask), seqlen
        )

        return (token_ids, attention_mask)


    def load_dataset(path, word_pad_size=16, char_pad_size=32, bert_pad_size=None):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                content = content.replace(' ','')
                ## bert方式处理
                seq_len_bert = len(tokenizer_bert.encode(content))
                bert_line, bert_att_mask = _prepare_bert_ids(content, tokenizer_bert, config.bert_pad_size)
                ## words方式处理
                words_line = []
                token = tokenizer_word(content)
                seq_len_words = len(token)
                if word_pad_size:
                    if len(token) < word_pad_size:
                        token.extend([PAD] * (word_pad_size - len(token)))
                    else:
                        token = token[:word_pad_size]
                        seq_len_words = word_pad_size
                # word to id
                for word in token:
                    words_line.append(vocab_word.get(word, vocab_word.get(UNK)))
                ## char方式处理
                chars_line = []
                token = tokenizer_char(content)
                seq_len_chars = len(token)
                if char_pad_size:
                    if len(token) < char_pad_size:
                        token.extend([PAD] * (char_pad_size - len(token)))
                    else:
                        token = token[:char_pad_size]
                        seq_len_chars = char_pad_size
                # char to id
                for word in token:
                    chars_line.append(vocab_char.get(word, vocab_char.get(UNK)))

                #处理下label
                if config.label_one_hot:
                    if isinstance(label, list): #['3','7']一个text对应多个标签（形式一）
                        label = [int(x) for x in label]
                    elif isinstance(label, str) and ',' in label: #'3, 7'一个text对应多个标签（形式二）
                        label = [int(x) for x in label.split(',')]
                    else:
                        label = [int(label)] #'3'
                    label = torch.zeros(10).scatter_(0, torch.LongTensor(label), 1).long().tolist() #label有十类所以是10
                else:
                    label = int(label)
                contents.append((label, bert_line, bert_att_mask, seq_len_bert, words_line, seq_len_words, chars_line, seq_len_chars))
        return contents  # [([...], 0), ([...], 1), ...]
    train = load_dataset(config.train_path, config.word_pad_size, config.char_pad_size)
    dev = load_dataset(config.dev_path, config.word_pad_size, config.char_pad_size)
    test = load_dataset(config.test_path, config.word_pad_size, config.char_pad_size)
    return vocab_word, vocab_char, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, config):
        # 用到了config.batch_size, config.device
        self.batch_size = config.batch_size
        self.batches = batches
        self.n_batches = len(batches) // config.batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = config.device

    def _to_tensor(self, datas):
        y = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        x_bert = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        x_att_mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len_bert = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        x_word = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        seq_len_word = torch.LongTensor([_[5] for _ in datas]).to(self.device)
        x_char = torch.LongTensor([_[6] for _ in datas]).to(self.device)
        seq_len_char = torch.LongTensor([_[7] for _ in datas]).to(self.device)
        return (x_bert, x_att_mask, seq_len_bert, x_word, seq_len_word, x_char, seq_len_char), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        tokenizer = jieba.lcut  # jieba分词
        # tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=None, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):

        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
