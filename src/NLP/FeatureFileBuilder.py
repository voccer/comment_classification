import os
from src.NLP.FileHandler import FileReader, FileWriter
from src.NLP.NLP import NLP
import src.NLP.Setting as Setting


class FeatureFileBuilder:
    def __init__(self, folder_path):
        self.__folderPath = folder_path
        self.__dictionary = FileReader(filepath=Setting.DIR_DICTIONARY).read_dictionary()
        self.__build_feature_from_folder()

    def __get_filepath(self):
        neg_path = self.__folderPath + "/neg"
        pos_path = self.__folderPath + "/pos"
        file_paths = [neg_path + "/" + file for file in os.listdir(neg_path)]
        file_paths = file_paths + [pos_path + "/" + file for file in os.listdir(pos_path)]
        return file_paths

    # Đọc lần lượt các file train, test và xây dựng file feature
    def __build_feature_from_folder(self):
        # Đọc tất cả các file trong folder truyền vào
        file_reader = FileReader(filepath="")
        file_writer = FileWriter(filepath=self.__folderPath + '/feature.txt', Data="")
        nlp = NLP(text="")
        file_paths = self.__get_filepath()

        # Đọc các file trong folder
        count = 0
        for filePath in file_paths:
            S = ""
            bow = {}
            file_reader.filePath = filePath
            nlp.text = file_reader.read()
            list_word = nlp.get_words_feature()

            # Đếm số lần xuất hiện của các từ, sử dụng pp BoW
            for word in list_word:
                index_dict = self.__dictionary.get(word)
                if (index_dict==None): index_dict = -1
                if index_dict in bow:
                    bow[index_dict] = bow.get(index_dict) + 1
                else:
                    bow[index_dict] = 1

            # Lưu tệp sau khi đã mã hóa vào 1 xâu, sau đó lưu vào file feature
            for word in bow:
                S += str(count) + ":" + str(word) + ":" + str(bow.get(word)) + "\n"
            file_writer.Data = S
            file_writer.write_feature()
            count += 1


