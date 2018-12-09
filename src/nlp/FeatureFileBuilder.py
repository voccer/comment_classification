from src.FileHandler import FileReader
from src.nlp.NLP import NLP
import src.Setting as Setting
import os
import numpy as np

# Input : Folder_path
# Output : File feature của folder_path

class FeatureFileBuilder:
    def __init__(self, folder_path, number = None):
        self.__folderPath = folder_path
        self.__dictionary = FileReader(path=Setting.DIR_DICTIONARY).read_dictionary()
        self.__number = number

    # Lấy tập paths của các file trong folder
    def __get_filepath(self):
        file_paths = [self.__folderPath+ '/' + file for file in os.listdir(self.__folderPath)]
        return file_paths

    # Đọc number file trong folder cho trước, biểu thị dưới dạng BoW
    def build_feature_from_folder(self):
        file_paths = self.__get_filepath()
        nlp = NLP(text="")
        count = 0
        feature = np.array([]).astype(int)
        for filePath in file_paths:
            # Đọc text trong file và ghi lại vào list_word
            if count == self.__number: break
            nlp.text = FileReader(path=filePath).read()
            list_word = nlp.get_words_feature()

            feature = np.append(feature, self.__build_feature_from_file(list_word, count))
            count += 1
        return feature.reshape(-1,3)

    """
    Đếm số lần xuất hiện của các từ, sử dụng pp BoW
    Đối với lần lượt từng từ trong list_word, kiểm tra xem có trong dictionary không
    Nếu không có trong dictionary thì không được gán index
    Nếu có trong dictionary, kiểm tra xem đã xuất hiện trong bow không. Sau có cập nhật từ đó trong BoW
    """
    def __build_feature_from_file(self, list_word, count):
        bow = {}
        for word in list_word:
            index_dict = self.__dictionary.get(word)
            if (index_dict == None): continue
            if index_dict in bow:
                bow[index_dict] = bow.get(index_dict) + 1
            else: bow[index_dict] = 1

        # Lưu tệp sau khi đã mã hóa vào 1 numpy
        S = np.array([]).astype(int)
        for word in bow:
            S = np.append(S, np.array([count, word, bow.get(word)]))
        return S