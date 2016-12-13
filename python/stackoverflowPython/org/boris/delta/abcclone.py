#-*-coding:utf-8-*-
from tgrocery import Grocery
from collections import defaultdict
import tgrocery.learner.learner
import sys
import os
from os import path
# from tgrocery.learner.lib
import pandas as pd
from tgrocery.learner.learner import LearnerProblem
from tgrocery.learner.learner import LearnerParameter
from tgrocery.learner.learner import LearnerModel
LIBLINEAR_HOME = os.environ.get('LIBLINEAR_HOME') or os.path.dirname(os.path.abspath(__file__)) + '/liblinear'
sys.path = [LIBLINEAR_HOME, LIBLINEAR_HOME + '/python'] + sys.path
from liblinearutil import train as liblinear_train, predict as liblinear_predict, save_model as liblinear_save_model, load_model as liblinear_load_model
import liblinear
import tgrocery.learner.learner
from sklearn import metrics
import jieba
import sys
from ctypes import *
reload(sys)
sys.setdefaultencoding('utf-8')


# grocery = Grocery('group3')
#训练集数据
train_src=[]
#测试集数据
test_src = []
#token转ID
tok2idx = {'>>dummy<<': 0}
#ID转token
idx2tok = None
#类型与id映射
class2idx = {}
#id与类型映射
idx2class = None
# for line in open('/Users/boris/Downloads/dianxin03_news_data/t1.txt'):
#     line=line.encode('utf-8')
#     line=line.strip().rsplit('\t',1)
#     #info=(keydic[line[1]],line[0])
#     info = (line[1], line[0])
#     train_src.append(info)
# print "end read"

count = []

#读取训练数据
#按9：1生成训练集与测试集
#train_src  ， test_src
def read_data(data_file):
    data = pd.read_csv(data_file,sep="\t",header=None,names=["text","label"],)
    train = data[:int(len(data) * 0.9)]
    test = data[int(len(data) * 0.9):]
    print "获取训练样本与测试集：%d %d",len(train),len(test)
    for index, row in train.iterrows():
        info = (row["label"], row["text"])
        train_src.append(info)
    for index, row in test.iterrows():
        info = (row["label"], row["text"])
        test_src.append(info)



#对类型进行编号，传入一个类型，如果类型名不在class2dix里，就生成一个新映射
def to_idx(class_name):
    if class_name in class2idx:
        return class2idx[class_name]

    m = len(class2idx)
    class2idx[class_name] = m
    return m

#将数据转成svm模型，包括数据分词
def to_svm(text, class_name=None):
    feat = bigram(preprocess(text, None))
    if class_name is None:
        return feat
    return feat, to_idx(class_name)

#根据tokens生成字典
def unigram( tokens):
    feat = defaultdict(int)
    NG = ngram2fidx
    for x in tokens:
        if (x,) not in NG:
            NG[x,] = len(NG)
        feat[NG[x,]] += 1
    return feat
#字典数据
ngram2fidx = {'>>dummy<<': 0}
#字典数据反向
fidx2ngram = None
#我也不知道是在干啥
def bigram(tokens):
    feat = unigram(tokens)
    NG = ngram2fidx
    for x, y in zip(tokens[:-1], tokens[1:]):
        if (x, y) not in NG:
            NG[x, y] = len(NG)
        feat[NG[x, y]] += 1
    return feat
#把文本数据分词转成svm模型写入到svm文件
def convert_text( text_src, delimiter, output=None):
    if not output:
        output = '%s.svm' % text_src
    text_src = read_text_src(text_src, delimiter)
    with open(output, 'w') as w:
        for line in text_src:
            try:
                label, text = line
            except ValueError:
                continue
            feat, label = to_svm(text, label)
            w.write('%s %s\n' % (label, ''.join(' {0}:{1}'.format(f, feat[f]) for f in sorted(feat))))
#按分隔符拆分文字
def read_text_src(text_src, delimiter):
    if isinstance(text_src, str):
        with open(text_src, 'r') as f:
            text_src = [line.split(delimiter) for line in f]
    elif not isinstance(text_src, list):
        raise TypeError('text_src should be list or str')
    return text_src

#自定义数据分词方法，custom_tokenize可以指定分词函数，默认指定jieba.cut全模式
def preprocess(text, custom_tokenize):
    if custom_tokenize is not None:
        tokens = custom_tokenize(text)
    else:
        tokens = _default_tokenize(text)
    ret = []
    for idx, tok in enumerate(tokens):
        if tok not in tok2idx:
            tok2idx[tok] = len(tok2idx)
        ret.append(tok2idx[tok])
    return ret

#默认分词器，jieba.cut全模型
def _default_tokenize(text):
    count.append(1)
    if len(count) % 2000 == 0:
        print len(count)
    return jieba.cut(text, cut_all=True)

def _dict2list(d):
    if len(d) == 0:
        return []
    m = max(v for k, v in d.iteritems())
    ret = [''] * (m + 1)
    for k, v in d.iteritems():
        ret[v] = k
    return ret

#按传入ID获取类型名称
def to_class_name(idx):
    # if idx2class is None:
    #     idx2class = _dict2list(class2idx)
    if idx == -1:
        return "**not in training**"
    if idx >= len(idx2class):
        raise KeyError(
            'class idx ({0}) should be less than the number of classes ({0}).'.format(idx, len(idx2class)))
    return idx2class[idx]

#获取类型名称
def get_class_name(class_idx):
    return to_class_name(class_idx)
#按模型预测一行数据
def predict_one(xi, m):
    if isinstance(xi, (list, dict)):
        xi = liblinear.gen_feature_nodearray(xi)[0]
    learner_param = LearnerParameter(m.param_options[0], m.param_options[1])

    if m.bias >= 0:
        i = 0
        while xi[i].index != -1: i += 1

        # Already has bias, or bias reserved.
        # Actually this statement should be true if
        # the data is read by read_SVMProblem.
        if i > 0 and xi[i-1].index == m.nr_feature + 1:
            i -= 1

        xi[i] = liblinear.feature_node(m.nr_feature + 1, m.bias)
        xi[i+1] = liblinear.feature_node(-1, 0)

    LearnerProblem.normalize_one(xi, learner_param, m.idf)

    dec_values = (c_double * m.nr_class)()
    label = liblinear.liblinear.predict_values(m, xi, dec_values)

    return label, dec_values
#读取训练文件，生成训练集和测试集
read_data('/Users/boris/Downloads/dianxin03_news_data/t1.txt')
#svm数据文件
train_svm_file = '%s_train.svm' % "mypro"
#将训练数据进行分词转换，保存在svm文件中
convert_text(train_src, output=train_svm_file, delimiter='\t')
print "6"
#读取svm文件到内存
learner_prob = LearnerProblem(train_svm_file)
print "5"
#设置学习参数
learner_param = LearnerParameter('', '-s 4')

print "3"
#初始化参数
idf = None
if learner_param.inverse_document_frequency:
    idf = learner_prob.compute_idf()
learner_prob.normalize(learner_param, idf)
print "2"
#训练创建模型
m = liblinear_train(learner_prob, learner_param)
print "1"
#如果未指定交叉验证，使用LearnerModel
if not learner_param.cross_validation:
    m.x_space = None  # This is required to reduce the memory usage...
    m = LearnerModel(m, learner_param, idf)
print "训练完成"

# writefileopen = open('/Users/boris/Downloads/dianxin03_news_data/group3writefile2.txt', 'w+')
#将类型字典转换成【id,classname】
idx2class = _dict2list(class2idx)
#测试集结果
test_values = []
#测试集真实结果
real_values = []
#遍历测试集，每条放入模型中，返回预测结果
for line in test_src:
    textid=line[0]
    textinfo=line[1]
    text1 = to_svm(textinfo)
    y, dec = predict_one(text1, m)
    # print y
    y = get_class_name(int(y))
    real_values.append(line[0])
    test_values.append(y)

#用预测结果与真实结果比较，获取准确率与召回率
score = metrics.precision_score(real_values, test_values, average="micro")
recall = metrics.recall_score(real_values, test_values, average="micro")
print('precision: %.2f%%, recall: %.2f%%' % (100 * score, 100 * recall))
print "predict_"