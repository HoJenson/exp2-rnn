# 该文件应先于main.py执行，用以生成与处理好的数据
# 删除停用词，标点符号等

import pandas as pd
import string as st

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# 从json文件中提取出来保存为评价和评分列保存为.csv，文件所在的路径
PATH = 'data/yelp.csv'

df = pd.read_csv(PATH)
df['text'] = df['text'].astype(str).apply(str.lower)
df['stars'] = df['stars'].apply(lambda x: x - 1)
texts = df.text
labels = df.stars

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
punctuations = st.punctuation

X = []
for text in list(texts):
    temp_list = []
    tokens = word_tokenize(str(text))
    for token in tokens:
        if token not in punctuations:
            if token == 'not':
                temp_list.append(token)
            elif token not in stop_words and '...' not in token:
                stem = ps.stem(token)
                temp_list.append(stem)
    X.append(' '.join(temp_list))

data = pd.DataFrame({
    'X':X,
    'labels':labels})
# 预处理后文件保存的位置
data.to_csv('data/yelp_m.csv')