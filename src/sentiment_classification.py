from pyquery import PyQuery as pq
import jieba 
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report



def process_file():
    """
    读取训练数据和测试数据，并对它们做一些预处理
    """    
    train_pos_file = "../data/train.positive.txt"
    train_neg_file = "../data/train.negative.txt"
    test_comb_file = "../data/test.combined.txt"
    pas_labels = None
    neg_labels = None
    pas = []
    neg = []
    test = []
    labels = []
    
    with open(train_pos_file, 'r', encoding='utf8') as f:
        html = pq(f.read())
        comments = html('review')
        for comment in comments:
            pas.append(comment.text.strip())
        
        pas_labels = ['1'] * len(comments)
        
    with open(train_neg_file, 'r', encoding='utf8') as f:
        html = pq(f.read())
        comments = html('review')
        for comment in comments:
            neg.append(comment.text.strip())
        
        neg_labels = ['0'] * len(comments)
    
    with open(test_comb_file, 'r', encoding='utf8') as f:
        html = pq(f.read())
        comments = html('review')
        for comment in comments:
            test.append(comment.text.strip())
            labels.append(comment.attrib['label'])       
   
    return pas, pas_labels, neg, neg_labels, test, labels
        
# TODO: 读取文件部分，把具体的内容写入到变量里面

pas_comments, pas_labels, neg_comments, neg_labels, test_comments, test_labels = process_file()

train_comments = pas_comments + neg_comments
train_labels = pas_labels + neg_labels

# 下面是做一个简单的分析，评论的长度不影响改评价是正面还是负面

# 长度不可能超过1000
pas_counts = [0]*1000
neg_counts = [0]*1000
length = 0
for comment in pas_comments:
    length = len(jieba.lcut(comment))
    pas_counts[length] = pas_counts[length] + 1
    
for comment in neg_comments:
    length = len(jieba.lcut(comment))
    neg_counts[length] = neg_counts[length] - 1
    
plt.bar(range(len(pas_counts)), pas_counts)
plt.bar(range(len(neg_counts)), neg_counts) 

plt.show()
# 从图中可以看出情感和长度无关，


# 生成后面生成tfidf使用的词典
#   1. 停用词过滤
#   2. 去掉特殊符号
#   3. 去掉数字（比如价格..)
#   4. 词频大于一定的程度
pas_vocabulary = {}
neg_vocabulary = {}

with open('../data/chinese_stop_words.txt', 'r', encoding='utf8') as f:
    stops = f.read().split()

stops.append(' ')
stops.append('\n')
stops.append('\u3000')
stops = set(stops)

for comment in pas_comments:
    for word in jieba.cut(comment):
        if word not in stops and not word.isdigit():
            pas_vocabulary[word] = pas_vocabulary.get(word, 0) + 1
            
for comment in neg_comments:
    for word in jieba.cut(comment):
        if word not in stops and not word.isdigit():
            neg_vocabulary[word] = neg_vocabulary.get(word, 0) + 1   

            
# 确定词典
vocabulary_ = pas_vocabulary_order + neg_vocabulary_order
vocabulary = set(x[0] for x in vocabulary_ if x[1]>1)       

# 对评论做分词处理
train_comments_new = [] 
test_comments_new = []
train_labels_new = []

for i, comment in enumerate(train_comments):
    comment = jieba.lcut(comment)
    newcomment = ' '.join([word for word in comment if word not in stops])
    #  由于评论数据本身很短，如果去掉的太多，很可能字符串长度变成0
    if len(newcomment) != 0:
        train_comments_new.append(newcomment)
        train_labels_new.append(train_labels[i])

for comment in test_comments:
    comment = jieba.lcut(comment)
    test_comments_new.append(' '.join([word for word in comment if word not in stops]))   
     
     
# 利用tf-idf从文本中提取特征,写到数组里面. 

tfidf_vec = TfidfVectorizer(vocabulary = vocabulary) 
labels_tfidf = TfidfVectorizer(analyzer='char')

X_train = tfidf_vec.fit_transform(train_comments_new) # 训练数据的特征
y_train = labels_tfidf.fit_transform(train_labels_new) # 训练数据的label
X_test =  tfidf_vec.fit_transform(test_comments_new) # 测试数据的特征
y_test =  labels_tfidf.fit_transform(test_labels) # 测试数据的label

# 利用逻辑回归来训练模型
#       1. 评估方式： F1-score
#       2. 超参数（hyperparater）的选择利用grid search 
#       3. 打印出在测试数据中的最好的结果

param_grid = {'C': [1,10,100,1000,10000]}
clf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid, cv=5, scoring='f1_micro')
clf.fit(X_train, y_train.indices)
print(clf.best_params_)
print(clf.score(X_test, y_test.indices))
print(classification_report(y_train.indices, clf.predict(X_train)))
print(classification_report(y_test.indices, clf.predict(X_test)))

# 利用SVM来训练模型
#       1. 评估方式： F1-score
#       2. 超参数（hyperparater）的选择利用grid search 
#       3. 打印出在测试数据中的最好的结果

# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
parameters = {'kernel':['rbf'], 'gamma':['auto'], 'C':[3900, 4000, 4100]}
clf = GridSearchCV(svm.SVC(), parameters, cv=5, scoring='f1_micro')
y_train_ = y_train.indices
y_train_[y_train_==0] = -1
clf.fit(X_train, y_train_)
print(clf.best_params_)
print(classification_report(y_train_, clf.predict(X_train)))
y_test_ = y_test.indices
y_test_[y_test_==0] = -1
print(classification_report(y_test_, clf.predict(X_test)))

# 仍然使用SVM模型，但在这里使用Bayesian Optimization来寻找最好的超参数。 
#       1. 评估方式： F1-score
#       2. 超参数（hyperparater）的选择利用Bayesian Optimization 
#       3. 打印出在测试数据中的最好的结果
#       参考Bayesian Optimization开源工具： https://github.com/fmfn/BayesianOptimization

from bayes_opt import BayesianOptimization
from sklearn import svm

def black_box_function(C):
    svm_ = svm.SVC(kernel='rbf', gamma='auto',  C=C)
    svm_.fit(X_train, y_train_)
    return svm_.score(X_train, y_train_)
# 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'gamma': ['auto'], 
pbounds = {'C':(3900, 4000)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1
)

optimizer.maximize(
    init_points=3,
    n_iter=2,
)

print(optimizer.max)
c = optimizer.max['params']['C']
print(c)

svm_ = svm.SVC(kernel='linear', C=c)
svm_.fit(X_train, y_train_)
print(classification_report(y_train_, svm_.predict(X_train)))
print(classification_report(y_test_, svm_.predict(X_test)))