# 中文长文本多标签分类(pytorch)

TextCNN, TextRNN_Att, FastText参考：Chinese-Text-Classification-Pytorch实现

本仓主要实现并解决基于预训练bert，预训练字向量和词向量实现Bert_RCNN用于一个长文本对应对个标签的问题

## 介绍  

数据以bert_tokenize(sentence)、字、词为单位输入模型<br>
预训练bert模型，短文本用macbert，长文本用longformer(4096)
预训练字向量使用 [搜狗新闻 Word+Character 300d](https://github.com/Embedding/Chinese-Word-Vectors)，[点这里下载](https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ)
预训练词向量[腾讯词向量](https://ai.tencent.com/ailab/nlp/en/embedding.html)

## 环境
python 3.7
pytorch 1.7.1+cu101  
tqdm  
sklearn  

## 中文数据集
我从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，已上传至github，文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

数据集|数据量
--|--
训练集|18万
验证集|1万
测试集|1万

注：数据label为单标签，我统一用多标签的方式处理所以输入数据可以如下，label可以是列表，带逗号的string或者单数字：
```text
快讯：沪指震荡回升重返2500上方,新疆股走强	2
```
或者<br>
```text
快讯：沪指震荡回升重返2500上方,新疆股走强	2,5
```
或者<br>
```text
快讯：沪指震荡回升重返2500上方,新疆股走强	[2,5]
```

## 数据处理-生成训练集所需要的字、词向量及对应id文件

[处理代码](数据处理_根据当前语料生成字词id及向量.ipynb)

## 运行

```shell
python run.py
```
模型都在models目录下，超参定义和模型定义在同一文件中。

## bert长文本支持

```text
transformers==3.4.0 #高版本会报错
pip install git+https://github.com/allenai/longformer.git #安装longformer
```
```python
#具体参考longformer_demo
from longformer.longformer import Longformer, LongformerConfig
from longformer.sliding_chunks import pad_to_window_size
from transformers import BertTokenizerFast, AdamW

class args:
    adafactor=False
    adam_epsilon=1e-08
    attention_mode='sliding_chunks'
    longformer_pretrained_dir='~/pretrained/longformer-chinese-base-4096'

config = LongformerConfig.from_pretrained(args.longformer_pretrained_dir)
config.attention_mode = args.attention_mode
model_config = config
model = Longformer.from_pretrained(args.longformer_pretrained_dir, config=config)
print('max position embeddings: ', model.config.max_position_embeddings)
tokenizer = BertTokenizerFast.from_pretrained(args.longformer_pretrained_dir)
tokenizer.model_max_length = model.config.max_position_embeddings
#forward部分
def forward(self, input_ids, attention_mask, labels=None):
    input_ids, attention_mask = pad_to_window_size(
        input_ids, attention_mask, self.model_config.attention_window[0], self.tokenizer.pad_token_id)
    attention_mask[:, 0] = 2  # global attention for the first token
    #use Bert inner Pooler
    output = self.model(input_ids, attention_mask=attention_mask)[1]
```



## 对应论文
[1] Longformer: The Long-Document Transformer
[2] Convolutional Neural Networks for Sentence Classification  
[3] Recurrent Neural Network for Text Classification with Multi-Task Learning  
[4] Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification  
[5] Recurrent Convolutional Neural Networks for Text Classification  
[6] Bag of Tricks for Efficient Text Classification  
[7] Deep Pyramid Convolutional Neural Networks for Text Categorization  
[8] Attention Is All You Need
