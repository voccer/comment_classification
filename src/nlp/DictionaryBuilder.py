import os
from src.FileHandler import FileReader, FileWriter
from src.nlp.NLP import NLP
import src.Setting as Setting


class DictionaryBuilder:
    def __init__(self):
        self.__dictionary = set([])
        self.__build_dict()

    def __build_dict(self):
        self.__build_dict_from_folder(Setting.DIR_TEST_PATH + "/neg/")
        self.__build_dict_from_folder(Setting.DIR_TEST_PATH + "/pos/")
        self.__build_dict_from_folder(Setting.DIR_TRAIN_PATH + "/neg/")
        self.__build_dict_from_folder(Setting.DIR_TRAIN_PATH + "/pos/")
    # Đọc dữ liệu từ tất cả dữ liệu từ 1 folder rồi ghi vào từ điển

    def __build_dict_from_folder(self, folder_path):
        # Lấy danh sách file trong folderPath, đọc, tách từ và lưu vào từ điển
        file_reader = FileReader(path="")
        nlp = NLP(text="")
        files = [folder_path + '/' + files for files in os.listdir(folder_path)]

        # Đọc lần lượt từng file, ghi những từ còn thiếu vào trong từ điển
        for filePath in files:
            file_reader.filePath = filePath
            nlp.text = file_reader.read()
            list_word = set(nlp.get_words_feature())
            self.__dictionary = self.__dictionary | list_word

        file_writer = FileWriter(filepath=Setting.DIR_DICTIONARY, Data=self.__dictionary)
        file_writer.write_dictionary()