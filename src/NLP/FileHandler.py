class FileReader:
    def __init__(self, path):
        self.filePath = path

    # Đọc file
    def read(self):
        with open(file=self.filePath, mode='r') as f:
            s = f.read()
        return s

    # Đọc stopword, trả ra set stopword
    def read_stopword(self):
        with open(file=self.filePath, mode='r') as f:
            stopwords = set([w.strip() for w in f.readlines()])
        return stopwords

    # Đọc từ điển
    def read_dictionary(self):
        count = 0
        dictionary = {}
        # Mỗi từ trong từ điển, loại bỏ ký từ \n ở cuối
        with open(file=self.filePath, mode='r') as f:
            for word in f.readlines():
                dictionary[''.join(word.split('\n'))] = count
                count += 1
        return dictionary


class FileWriter:
    def __init__(self, filepath, Data):
        self.filePath = filepath
        self.Data = Data

    # Ghi file
    def write_dictionary(self):
        with open(file=self.filePath, mode='w') as f:
            for word in self.Data:
                f.write(word + "\n")

    # Ghi file feature
    def write_feature(self):
        with open(file=self.filePath, mode='a') as f:
            f.write(self.Data)


