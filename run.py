# coding: utf-8
__author__='zhoubin'
__mtime__='20210604'

import torch
import torch.nn as nn
import numpy as np
from models.Bert_RCNN import Config, Model
import math
import time
from sklearn import metrics
from utils import get_time_dif

# topk f1评测函数
def get_score(predict_label_and_marked_label_list, topk=5):
    #该评测函数参考自https://www.biendata.xyz/competition/zhihu/evaluation/
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]

    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)

    """
    right_label_num = 0  # 总命中标签数量
    right_label_at_pos_num = [0] * topk  # 在各个位置上总命中数量
    sample_num = 0  # 总问题数量
    all_marked_label_num = 0  # 总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), topk)), predict_labels):
            if label in marked_label_set:  # 命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

    precision = 0.0
    for pos, right_num in zip(range(0, topk), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num
    f1 = (precision * recall) / (precision + recall + 0.0000000000001)

    return f1, precision, recall, right_label_at_pos_num

################ train_eval.py ################
# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', bert='bert', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name and bert not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = config.loss_func(outputs, labels.float())
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                predict = outputs.data.topk(config.topk, dim=1)[1].cpu().tolist()
                true_target = labels.data.float().cpu().topk(config.topk, dim=1)
                true_index = true_target[1][:, :config.topk]
                true_label = true_target[0][:, :config.topk]
                predict_label_and_marked_label_list = []
                for k in range(labels.size(0)):
                    true_index_ = true_index[k]
                    true_label_ = true_label[k]
                    true = true_index_[true_label_ > 0]
                    predict_label_and_marked_label_list.append((predict[k], true.tolist()))
                train_f1_, prec_, recall_, _ss = get_score(predict_label_and_marked_label_list, config.topk)

                dev_f1, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train F1: {2:>6.2%},  Val Loss: {3:>5.2},  Val F1: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_f1_, dev_loss, dev_f1, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_f1, test_loss = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test F1: {1:>6.2%}'
    print(msg.format(test_loss, test_f1))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    f1_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)

            loss = config.loss_func(outputs, labels.float())
            loss_total += loss
            predict = outputs.data.topk(config.topk, dim=1)[1].cpu().tolist()
            true_target = labels.data.float().cpu().topk(config.topk, dim=1)
            true_index = true_target[1][:, :config.topk]
            true_label = true_target[0][:, :config.topk]
            predict_label_and_marked_label_list = []
            for k in range(labels.size(0)):
                true_index_ = true_index[k]
                true_label_ = true_label[k]
                true = true_index_[true_label_ > 0]
                predict_label_and_marked_label_list.append((predict[k], true.tolist()))
            f1_, prec_, recall_, _ss = get_score(predict_label_and_marked_label_list, config.topk)
            f1_all = np.append(f1_all, f1_)

    return f1_all.mean(), loss_total / len(data_iter)

######## END ########

if __name__ == '__main__':
    class args:
        model = 'Bert_RCNN'
        embedding = 'pre_trained'
        dataset = 'THUCNews'  # 数据集

    # 搜狗新闻字向量:embedding_sogou_chars.npz, 腾讯词向量:embedding_tencent_words.npz, 随机初始化:random
    embedding_char = 'embedding_sogou_chars.npz'
    embedding_word = 'embedding_tencent_words.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'Bert_RCNN'  # TextCNN, FastText, TextRNN_Att, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_ds_full, build_iterator, get_time_dif

    config = Config(args.dataset, embedding_char, embedding_word)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    vocab_word, vocab_char, train_data, dev_data, test_data = build_ds_full(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
