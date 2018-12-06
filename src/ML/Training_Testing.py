import numpy as np
from scipy.sparse import coo_matrix  # Sử dụng ma trận thưa thớt để lưu features
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score # Sử dụng để đánh giá kết quả mô hình
import sklearn.decomposition as RandomizedPCA
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import sklearn.metrics as sklm
from sklearn import feature_selection as fs
import sklearn.model_selection as ms
import numpy.random as nr
import src.Setting as setting


nwords = 140187  # số từ trong từ điển 
nTrain = 5000
nTest = 5000

# data path and file name

pathTrain = setting.DIR_PATH + '/Data/Feature/feature_' + str(nTrain) + '/'  # đường dẫn tới thư mục chứa file features
pathTest = setting.DIR_PATH + '/Data/Feature/feature_' + str(nTest) + '/'
train_pos_fn = 'train_pos_' + str(nTrain)   # tên file features train có nhãn thích
train_neg_fn = 'train_neg_' + str(nTrain)    # tên file features train có nhãn không thích
test_pos_fn = 'test_pos_' + str(nTest)    # tên file features test có nhãn thích
test_neg_fn = 'test_neg_' + str(nTest)   # tên file features test có nhãn không thích


# Hàm đọc file và tạo vector features cho các movie demo của bộ train và test
def read_data(path, data_pos, data_neg, n):
    # tạo list label với n movie review đầu là thích, n movie review sau là không thích
    label = np.array([])
    for i in range(2*n):
        if i < n:
            label = np.append(label,[1])
        else:
            label = np.append(label,[0])

    # đọc dữ liệu từ data_pos, sau đó xóa các khoảng trắng thừa hoặc các kí tự '\n'
    with open(path + data_pos) as f:
        pos = f.readlines()
    pos = [x.strip() for x in pos]

    # đọc dữ liệu từ data-neg, sau đó xóa các khoảng trắng thừa hoặc các kí tự '\n'
    with open(path + data_neg) as f:
        neg = f.readlines()
    neg = [x.strip() for x in neg]

    # Tạo ma trận 0 với n leng hàng 3 cột để lưu toàn bộ dữ liệu 2 file features
    leng = len(pos) + len(neg)
    dat = np.zeros((leng,3), dtype = int)

    for i,line in enumerate(pos):
        a = line.split(' ') # tách các dòng thành các chuỗi con, cắt theo dấu space
        dat[i, :] = np.array([int(a[0]),int(a[1]),int(a[2])]) # gán giá trị tương ứng vào dòng thứ i của ma trận dat
    
    for i,line in enumerate(neg):
        a = line.split(' ') # tách các dòng thành các chuỗi con, cắt theo dấu space
        dat[i+len(pos),:] = np.array([int(a[0])+n,int(a[1]),int(a[2])]) # gán giá trị tương ứng vào dòng thứ i của ma trận dat

    # Tạo ma trận thưa thớt để lưu các vector features(mỗi movie review là 1 vector, mỗi vector có nwords chiều):
        ## cột 0(dat[:, 0]) của ma trận dat cho biết đây là features của movie review thứ mấy
        ## cột 1(dat[:, 1]) của ma trận dat cho biết từ đang xét là từ thứ bao nhiêu trong bộ từ điển, từ đó suy ra vị trí trong vector feature
        ## cột 2(dat[:, 2]) của ma trận dat có chứa thông tin về tần suất xuất hiện của từ
        ### Tại 1 hàng của ma trận dat, nếu các giá trị của 3 cột lần lượt là x y z thì giá trị của data[x][y-1] = z

    data = coo_matrix((dat[:, 2], (dat[:, 0], dat[:, 1])), shape = (len(label),nwords))
    return (data,label)

def print_metrics(labels, probs):
    metrics = sklm.precision_recall_fscore_support(labels, probs)
    conf = sklm.confusion_matrix(labels, probs)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0, 0] + '             %5d' % conf[0, 1])
    print('Actual negative    %6d' % conf[1, 0] + '             %5d' % conf[1, 1])
    print('')
    print('Accuracy        %0.5f' % sklm.accuracy_score(labels, probs))
    print('AUC             %0.2f' % sklm.roc_auc_score(labels, probs[:]))
    print('Macro precision %0.2f' % float((float(metrics[0][0]) + float(metrics[0][1])) / 2.0))
    print('Macro recall    %0.2f' % float((float(metrics[1][0]) + float(metrics[1][1])) / 2.0))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])

(train_data, train_label)  = read_data(pathTrain, train_pos_fn, train_neg_fn,nTrain)
(test_data, test_label)  = read_data(pathTest, test_pos_fn, test_neg_fn,nTest)
# Feature = np.array([]).astype(int)
# Feature = np.append(Feature,train_data).reshape(-1,3)
# print(Feature.shape)
# Feature = np.append(Feature, test_data).reshape(-1,3)

# indx = range(Features_reduced.shape[0])
# X_train = Features_reduced[indx[0],:]
# y_train = np.ravel(Labels[indx[0]])
# X_test = Features_reduced[indx[1],:]
# y_test = np.ravel(Labels[indx[1]])


# Feature = coo_matrix((Feature[:, 2], (Feature[:, 0], Feature[:, 1])), shape = (4*nTest,nwords))
#
# sel = fs.VarianceThreshold(threshold=(0.95 * (1 - 0.95)))
# Features_reduced = sel.fit_transform(Feature)
# print(sel.get_support())
# print(Features_reduced.shape)
#
# Labels = np.append(train_label, test_label)

# print(train_data.shape)

## Define the variance threhold and fit the threshold to the feature array.
# nr.seed(1115)
# indx = range(Features_reduced.shape[0])
# indx = ms.train_test_split(indx, test_size = 10000)
# x_train = Features_reduced[indx[0],:]
# y_train = np.ravel(Labels[indx[0]])
# x_test = Features_reduced[indx[1],:]
# y_test = np.ravel(Labels[indx[1]])

## Define and fit the logistic regression model
# logstic_mod.fit(X_train, y_train)



## Print the support and shape for the transformed features
# print(Features_reduced.shape)

# clf = naive_bayes.BernoulliNB()
clf = naive_bayes.MultinomialNB()
clf.fit(train_data, train_label)


probabilities = clf.predict(test_data)
print_metrics(test_label, probabilities)
# plot_auc(test_label, probabilities)

# y_pred = clf.predict(test_data)
# print('Training size = %d, accuracy = %.2f%%' % \
#       (train_data.shape[0],accuracy_score(test_label, y_pred)*100))