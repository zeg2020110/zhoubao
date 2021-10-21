import pandas as pd #用于数据分析
import numpy as np
import tensorflow as tf
import jieba #用于中文文本分词
from gensim.models.word2vec import Word2Vec #自然语言处理工具模型
import multiprocessing #python自带的多线程库
from gensim.corpora.dictionary import Dictionary#语料库，使用字典可以生成训练用的语料
from keras.preprocessing import sequence #序列预处理
from sklearn.model_selection import train_test_split #对数据进行分割
import yaml #YAML是一个对所有编程语言都很友好的数据序列化标准(也可以认为是配置文件)；类比json

##加载文件
def loadfile():
     #当前路径为E:\program\python_code\tfTest。

    neg=pd.read_csv('.\\data\\neg.csv',header=None,index_col=None,error_bad_lines=False)
    pos=pd.read_csv('.\\data\\pos.csv',header=None,index_col=None,error_bad_lines=False)

    combined = np.concatenate((pos[0],neg[0]))#pos[0]是获取该dateframe的第一列数据，对于本例，只有一列。由于返回的series类型的，与数组一致，因此可以使用拼接。
    y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg),dtype=int)))#创建了两个对应长度的矩阵，1，0代表各自的标签

    return combined,y


##定义一些(超)参数：
# set parameters:
cpu_count = multiprocessing.cpu_count()  #电脑的线程数
vocab_dim = 100
n_iterations =5  # ideally more..
n_exposures = 10 # 所有频数超过10的词语
window_size = 7
n_epoch = 4
input_length = 100
maxlen = 100
batch_size = 32


##定义一些处理函数：
#(1)对句子经行分词，并去掉换行符
def tokenizer(text): #分词器，编译器
    '''
        解析器将每个文档转换为小写，然后删除换行符，最后在空格上进行分割
        txt=[]
        for document in text：
            t=jieba.lcut(document.replace('\n', ''))
            txt.append(t)
        return txt
    '''
    #replace就是字符串替换函数，将换行删除，然后再对每个评论进行分词。
    text = [jieba.lcut(document.replace('\n', '')) for document in text] #text是一个数组，for循环与语句写在一起，与分开写效果是一样的，更加简洁！
    return text


#(2)创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):

    #创建一个
    model = Word2Vec(size=vocab_dim, #100#特征向量（词向量）的维度。
                     min_count=n_exposures, #10:词频小于10的单词会被丢弃。
                     window=window_size, #7
                     workers=cpu_count, #8
                     iter=n_iterations) #1
    model.build_vocab(combined) # input: list #从一系列句子中构建词汇表，要求输入是一个字符串列表
    model.train(combined,total_examples = model.corpus_count,epochs =model.iter)#训练模型
    model.save('.\\data\\lstm_data_test\\Word2vec_model.pkl')#保存模型
    #index_dict为词频超过10的索引；word_vectors为每个词的词向量；combined为每条评论的分词所对应的索引数组。
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined

#创建字典
def create_dictionaries(model=None,
                        combined=None):
    '''
        Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()#
        #model.wv.vocab.keys()中的keys()应该是字典中keys()方法：返回一个列表，其中包含字典中的所有键
        gensim_dict.doc2bow(model.wv.vocab.keys(),allow_update=True) #使用词袋模型去表示文档

        # items():是python中的方法；items() 方法把字典中每对 key 和 value 组成一个元组，并把这些元组放在列表中返回
        #w2indx也是一个字典。字典中的键是gensim_dict中的v值，值是gensim_dict字典中的（k+1）值。
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#字典gensim_dict中存放的是所有频数超过10的词语。
        #字典w2indx存放了这些词语所在的索引。至于为什么+1：见下面，索引0要表示词频低于10的词。new_txt.append(0)
        '''
            相当于：
                    dic={}
                    for k,v in gensim_dict.items():#当两个参数时
                         dic[v]=k+1
                    dic就等同于w2indx。
        '''
        #model[word]应该是获取模型中词word的词向量。可是文档写的是model.wv[word],但是结果不一样啊！是每次结果都不一样，这可能也是根据网络模型计算出来的，所以会有偏差
        w2vec = {word: model.wv[word] for word in w2indx.keys()}#获取每个词频超过10的词的词向量
        '''
            字典中的keys()方法：返回一个列表，其中包含字典中的所有键
            dic={}
            for word in w2indx.keys()
                dic[word]=model[word]
            dic等同于w2vec
        '''

        def parse_dataset(combined): # 闭包-->临时使用
            '''
                Words become integers
            '''
            data=[]
            for sentence in combined: #这里combined传入的是分词后的句子，是一个列表，每一行都是一条评论
                new_txt = []
                for word in sentence: #每条句子（评论）是由若干分词组成的一个数组。
                    try:
                        new_txt.append(w2indx[word]) #对每个分词超过的10，将他们的词向量索引加入该数组
                    except:
                        new_txt.append(0) # #这里0的作用体现出来，因为要表示词频低于10的词，所以上面从1开始。
                data.append(new_txt)
            return data #创两个数组的原因：因为是两层循环
        combined=parse_dataset(combined)#等于这是得到了每一个评论（句子）对应的分词在词向量表中的索引。
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#通过预处理，将每条评论所生成的分词索引数组的长度变得一致：100
        return w2indx, w2vec,combined
    else:
        print ('No data provided...')
#得到数据
def get_data(index_dict,word_vectors,combined,y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim)) # 创建权重矩阵，维数为n_symbols*100；上面通过预处理将每条评论的分词的索引数组都变成了100

    for word, index in index_dict.items(): # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]#每个词向量的长度为100吗,是的，经过了验证！
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)#切分数据集
    y_train = tf.keras.utils.to_categorical(y_train)#把类别标签转换为onehot编码。
    y_test = tf.keras.utils.to_categorical(y_test)
    # print x_train.shape,y_train.shape
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test




##定义LSTM网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print ('Defining a Simple Keras Model')
    model = tf.keras.Sequential()  # or Graph or whatever
    model.add(tf.keras.layers.Embedding(
                        input_dim=n_symbols,#字典的长度
                        output_dim=vocab_dim,#词向量的维度，代表全连接嵌入的维度
                        mask_zero=True, #把 0 看作为一个应该被遮蔽的特殊的 "padding" 值。
                       # weights=[embedding_weights],#该参数已经被淘汰了吧！感觉没啥用
                        input_length=input_length))  #输入语句的长度
    model.add(tf.keras.layers.LSTM(units=50,activation="sigmoid",recurrent_activation="hard_sigmoid"))#增加lstm层
    model.add(tf.keras.layers.Dropout(0.5))#将 Dropout 应用于输入
    model.add(tf.keras.layers.Dense(2)) # 全连接层
    model.add(tf.keras.layers.Activation('sigmoid'))#将激活函数应用于输出。

    print ('Compiling the Model...')
    #指定训练时用的优化器、损失函数和准确率评测标准
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    print ("Train...") # batch_size=32
    #以固定数量的轮次（数据集上的迭代）训练模型。
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,verbose=1,validation_data=(x_test,y_test))

    print ("Evaluate...")
    #在测试模式，返回误差值和评估标准值。
    score = model.evaluate(x_test, y_test,batch_size=batch_size)

    #使用yaml文件保存模型
    yaml_string = model.to_yaml()
    with open('.\\data\\model\\lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    #保存权重
    model.save_weights('.\\data\\model\\lstm.h5')
    print ('Test score:', score)


##训练模型
print ('Loading Data...')
combined,y=loadfile()
print (len(combined),len(y))
print ('Tokenising...')
combined = tokenizer(combined)
print ('Training a Word2vec model...')
index_dict, word_vectors,combined=word2vec_train(combined)
print ('Setting up Arrays for Keras Embedding Layer...')
#n_symbols是所有词向量个数的长度；embedding_weights是词向量组成的权重矩阵；后面是将数据集分割成的测试集和训练集。
n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
print ("x_train.shape and y_train.shape:")
print (x_train.shape,y_train.shape)
train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)