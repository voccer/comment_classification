import src.Setting as setting
from src.FileHandler import FileReader
import numpy as np

from scipy import sparse
from sklearn import naive_bayes
import sklearn.metrics as sklm
import sklearn.model_selection as ms
import sklearn.feature_selection as fs

"""
Chuyển ma trận feature thành dạng coo_matrix
@:return coo_matrix feature 
"""
def transform_to_coo_matrix(dat, row, col):
    dat = sparse.coo_matrix((dat[:, 2], (dat[:, 0], dat[:, 1])), shape=(row, col))
    return dat

def print_metrics(labels, probs):
    metrics = sklm.precision_recall_fscore_support(labels, probs)
    conf = sklm.confusion_matrix(labels, probs)
    print('                 Confusion matrix')
    print('                 Pridicted positive    Pridicted negative')
    print('Actual positive    %6d' % conf[0, 0] + '             %5d' % conf[0, 1])
    print('Actual negative    %6d' % conf[1, 0] + '             %5d' % conf[1, 1])
    print('')
    print('Accuracy        %0.5f' % sklm.accuracy_score(labels, probs))
    print('AUC             %0.5f' % sklm.roc_auc_score(labels, probs[:]))
    print('Macro precision %0.5f' % float((float(metrics[0][0]) + float(metrics[0][1])) / 2.0))
    print('Macro recall    %0.5f' % float((float(metrics[1][0]) + float(metrics[1][1])) / 2.0))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])

def get_train_test():
    n_train = 2000
    n_test = 1000
    n_words = 140187

    train_data = Load_Data(number=n_train).load_feature(is_train=True)
    train_label = Load_Data(number=n_train).load_label()

    test_data = Load_Data(number=n_test).load_feature(is_train=False)
    test_label = Load_Data(number=n_test).load_label()

    train_data = transform_to_coo_matrix(train_data, 2 * n_train, n_words)
    test_data = transform_to_coo_matrix(test_data, 2 * n_test, n_words)

    return (train_data, train_label, test_data, test_label)

def print_format(f,x,y,z):
    print('Fold %2d    %4.3f        %4.3f      %4.3f' % (f, x, y, z))

def print_cv(scores):
    fold = [x + 1 for x in range(len(scores['test_precision_macro']))]
    print('         Precision     Recall       Accuracy')
    [print_format(f,x,y,z) for f,x,y,z in zip(fold, scores['test_precision_macro'],
                                          scores['test_recall_macro'],
                                          scores['test_accuracy'])]
    print('-' * 40)
    print('Mean       %4.3f        %4.3f      %4.3f' %
          (float(np.mean(scores['test_precision_macro'])),
           float(np.mean(scores['test_recall_macro'])),
           float(np.mean(scores['test_accuracy']))))
    print('Std        %4.3f        %4.3f      %4.3f' %
          (float(np.std(scores['test_precision_macro'])),
           float(np.std(scores['test_recall_macro'])),
           float(np.std(scores['test_accuracy']))))

class Load_Data:
    def __init__(self, number = None, file_path = None):
        self.number = number
        if file_path == None:
            self.file_path = setting.DIR_FEATURE_PATH + "/" + "feature_" + str(number)
        else: self.file_path = file_path

    """
    Đọc file feature từ path_feature
    @:return numpy feature
    """
    def load_feature(self, is_train):
        if (is_train):
            dir_neg = self.file_path + "/train_neg"
            dir_pos = self.file_path + "/train_pos"
        else:
            dir_neg = self.file_path + "/test_neg"
            dir_pos = self.file_path + "/test_pos"

        if self.number != None:
            dir_neg = dir_neg + "_" + str(self.number)
            dir_pos = dir_pos + "_" + str(self.number)

        neg = FileReader(path = dir_neg).read_feature()
        pos = FileReader(path = dir_pos).read_feature()

        leng = len(pos) + len(neg)
        dat = np.zeros((leng, 3), dtype=int)

        for i, line in enumerate(pos):
            a = line.split(' ')  # tách các dòng thành các chuỗi con, cắt theo dấu space
            dat[i, :] = np.array(
                [int(a[0]), int(a[1]), int(a[2])])  # gán giá trị tương ứng vào dòng thứ i của ma trận dat

        n_pos = int(pos[-1].split(' ')[0])
        for i, line in enumerate(neg):
            a = line.split(' ')  # tách các dòng thành các chuỗi con, cắt theo dấu space
            dat[i + len(pos), :] = np.array(
                [int(a[0]) + n_pos, int(a[1]), int(a[2])])  # gán giá trị tương ứng vào dòng thứ i của ma trận

        return dat

    """
    Đọc file label 
    @:return list label 
    """
    def load_label(self):
        labels = np.array([])
        if self.number != None:
            for i in range(2*self.number):
                if i < self.number:
                    labels = np.append(labels, 1)
                else:
                    labels = np.append(labels, 0)
        else:
            print("Không có label")
        return labels

class Bayes_classification:
    (train_data, train_label, test_data, test_label) = get_train_test()
    total_data = sparse.vstack((train_data, test_data))
    total_label = np.append(train_label, test_label)

    def train_test(self, train_data, train_label, test_data, test_label):
        clf = naive_bayes.MultinomialNB()
        clf.fit(train_data, train_label)

        probabilities = clf.predict(test_data)
        print_metrics(test_label, probabilities)

    """
    Tách ngẫu nhiên bộ train_test từ self.train_data và train_label 
    @:returns train_data, train_label, test_data, test_label thu được sau khi tách ngẫu nhiên bộ train
    """
    def split_data(self):
        self.total_data = self.total_data.tocsr()
        indx = range(self.total_data.shape[0])
        indx = ms.train_test_split(indx, test_size=2500)
        train_data = self.total_data[indx[0], :]
        train_label = np.ravel(self.total_label[indx[0]])
        test_data = self.total_data[indx[1], :]
        test_label = np.ravel(self.total_label[indx[1]])

        return (train_data, train_label, test_data, test_label)

    """
    Thực hiện cross validate
    """
    def cross_validate(self, total_data):
        scoring = ['precision_macro', 'recall_macro', 'accuracy']
        clf = naive_bayes.MultinomialNB()
        scores = ms.cross_validate(clf, total_data, self.total_label, scoring=scoring,
                                   cv=20, return_train_score=False)
        print_cv(scores)

    """
    Thực hiện selection feature 
    """
    def selection_feature(self, variance):
        sel = fs.VarianceThreshold(threshold=(variance*(1-variance)))
        total_data = sel.fit_transform(self.total_data)
        return(total_data)

if __name__ == '__main__':
    print("Thực hiện train-test với bộ dữ liệu có số lượng file neg-pos trong file train-test bằng nhau")
    b = Bayes_classification()
    b.train_test(b.train_data, b.train_label, b.test_data, b.test_label)

    print("\n"*5)
    print("Thực hiện train-test với bộ dữ liệu có các file train-test được lấy ngẫu nhiên từ tập ban đầu")
    (train_data, train_label, test_data, test_label) = b.split_data()
    b.train_test(train_data, train_label, test_data, test_label)

    print("\n"*5)
    print("Thực hiện cross validate k=20")
    b.cross_validate(b.total_data)

    print("\n"*5)
    print("Thực hiện loại bỏ các feature có phương sai < 0.95")
    total_data = b.selection_feature(variance=0.95)
    print("Shape of original model : " + str(b.total_data.shape))
    print("Shape of model after selected feature : " +str(total_data.shape))
    b.cross_validate(total_data)

