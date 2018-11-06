from nltk.tokenize import word_tokenize
from src.NLP.FileHandler import FileReader
import src.NLP.Setting as Setting

class NLP:
    def __init__(self, text):
        self.text = text
        self.stopword = self.set_stopWord()

    # Tách từ
    def segmentation(self):
        tokens = word_tokenize(self.text)
        return tokens

    # Đọc stopword
    def set_stopWord(self):
        stop_word = FileReader(filepath=Setting.STOP_WORD).read_stopword()
        return stop_word

    # Xóa các ký tự đặc biệt và viết thường tất cả các chữ
    def split_words(self):
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

class main:
    if __name__ == '__main__':
        text = "Story of a man who has unnatural feelings for a pig. Starts out with a opening scene that is a terrific example of absurd comedy. A formal orchestra audience is turned into an insane, violent mob by the crazy chantings of it's singers."
        nlp = NLP(text=text)
        print(nlp.get_words_feature())