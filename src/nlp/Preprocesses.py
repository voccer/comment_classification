import os
import numpy as np
from src.nlp.NLP import NLP
from src.FileHandler import  FileReader
import operator
import src.Setting as setting

path1 = "/home/toanloi/Documents/comment_classification/Data/Data_Full/test/neg"
path2 = "/home/toanloi/Documents/comment_classification/Data/Data_Full/test/pos"
path3 = "/home/toanloi/Documents/comment_classification/Data/Data_Full/train/neg"
path4 = "/home/toanloi/Documents/comment_classification/Data/Data_Full/train/pos"
path5 = "/home/toanloi/Documents/comment_classification/Data/Data_Full/train/unsup"

def get_path(path):
    list = [path + "/" + file for file in os.listdir(path)]
    return(list)

def get_list():
    list = np.array([])
    # list = np.append(list, get_path(path1))
    list = np.append(list, get_path(path2))
    # list = np.append(list, get_path(path3))
    # list = np.append(list, get_path(path4))
    # list = np.append(list, get_path(path5))
    return list

def read_file(path):
    with open(path, mode = 'r') as f:
        data = f.read()
    return data

def write_file(data, path):
    with open(path, mode = 'w') as f:
        f.write(data)

def count_word():
    list = get_list()

    dict = {}
    for x in list[:5000]:
        data = read_file(x)
        data = NLP(text=data).get_words_feature()
        for word in data:
            keys = dict.get(word)
            if (keys == None):
                dict[word] = 1
            else:
                dict[word] = dict.get(word) + 1

    dict = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
    list = ""
    for x in dict:
        if x[1] < 50:
            break
        list += str(x[0]) + " : " + str(x[1])+ "\n"

    write_file(list, "/home/toanloi/Documents/comment_classification/Data/list_word")

def test():
    x = FileReader(path=setting.DIR_LEFT_WORD).read_left_word()
    print(x)

test()