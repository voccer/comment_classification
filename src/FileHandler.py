class FileReader:
    def __init__(self, path):
        self.filePath = path

    """
    Đọc 1 file text bất kỳ
    @:return String chứa nội dung file 
    """
    def read(self):
        with open(file=self.filePath, mode='r') as f:
            s = f.read()
        return s

    """
    Đọc file stopword
    @:return set stopword
    """
    def read_stopword(self):
        with open(file=self.filePath, mode='r') as f:
            stopwords = set([w.strip() for w in f.readlines()])
        return stopwords

    """
    Đọc từ điển 
    @:return dictionary từ điển, key là từ, value là vị trí từ trong têp 
    """
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

    """
    Ghi 1 text vào 1 file 
    """
    def write_dictionary(self):
        with open(file=self.filePath, mode='w') as f:
            for word in self.Data:
                f.write(word + "\n")

    """
    Ghi 1 feature vào 1 file
    Mỗi dòng của file feature có dạng : order id count 
    order : thứ tự file được đọc
    id : vị trí của từ đang xét trong từ điển 
    count : số lượng từ đó xuất hiện trong comment 
    """
    def write_feature(self):
        with open(file=self.filePath, mode='a') as f:
            for x in range(0, self.Data.shape[0]):
                for y in range(0, self.Data.shape[1]):
                    f.write( str(self.Data[x][y]) + ' ')
                f.write('\n')



