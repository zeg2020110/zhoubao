import numpy as np
import tensorflow as tf
import jieba
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
import yaml


##超参数
maxlen = 100


##定义处理函数
#对测试的语句进行jieba
def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1) #将分词后的变成一个行向量
    model=Word2Vec.load('.\\data\\lstm_data_test\\Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined

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
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量, (word->model(word))

        def parse_dataset(combined): # 闭包-->临时使用
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0) # freqxiao10->0
                data.append(new_txt)
            return data # word=>index
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print('No data provided...')


##执行结果
def lstm_predict(string):
    print('loading model......')
    with open('.\\data\\model\\lstm.yml', 'r') as f:
        yaml_string = yaml.load(f,Loader=yaml.FullLoader)
    model = tf.keras.models.model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('.\\data\\model\\lstm.h5')
    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    data = input_transform(string)
    data.reshape(1, -1)

    # print data
    result = model.predict_classes(data)
    print (result)
    if result[0] == 1:
        print(string, ' positive')
    else:
        print(string, ' negative')




#string='酒店的环境非常好，价格也便宜，值得推荐'
string='手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'
#string = "这是我看过文字写得很糟糕的书，因为买了，还是耐着性子看完了，但是总体来说不好，文字、内容、结构都不好"
# string = "虽说是职场指导书，但是写的有点干涩，我读一半就看不下去了！"
#string = '书的质量还好，但是内容实在没意思。本以为会侧重心理方面的分析，但实际上是婚外恋内容。'
#string = "不是太好"
#string = "不错不错"
#string = "真的一般，没什么可以学习的"
lstm_predict(string)