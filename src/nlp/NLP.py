from nltk.tokenize import word_tokenize
from src.FileHandler import FileReader
import src.Setting as Setting

class NLP:
    def __init__(self, text):
        self.text = text
        self.stopword, self.leftword, self.reverse = self.set_stopword_and_leftword_and_reverse()

    # Tách từ
    def segmentation(self):
        tokens = word_tokenize(self.text)
        return tokens

    # Đọc stopword, leftword, reverse
    def set_stopword_and_leftword_and_reverse(self):
        stop_word = FileReader(path=Setting.STOP_WORD).read_stopword()
        left_word = FileReader(path=Setting.LEFT_WORD).read_left_word()
        reverse = FileReader(path=Setting.REVERSE).read_reverse()
        return stop_word, left_word, reverse

    # def remove_negative_word(self):
    #     tokens = self.segmentation()


    # Xóa các ký tự đặc biệt và viết thường tất cả các chữ
    def split_words(self):
        # self.remove_negative_word()
        tokens = self.segmentation()
        try:
            return [x.strip(Setting.SPECIAL_CHARACTER).lower()
                    for x in tokens if len(x.strip(Setting.SPECIAL_CHARACTER)) > 0]
        except TypeError:
            return []

    # Loại bỏ stopword
    def get_words_feature(self):
        tokens = self.split_words()
        return [word for word in tokens if word not in self.stopword]