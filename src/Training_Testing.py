import numpy as np
from scipy.sparse import coo_matrix  # Sử dụng ma trận thưa thớt để lưu features
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score # Sử dụng để đánh giá kết quả mô hình

nwords = 140187  # số từ trong từ điển 
nTrain = 5000
nTest = nTrain

# data path and file name
path = '../Data/Feature/feature_' + str(nTrain) + '/'  # đường dẫn tới thư mục chứa file features
train_pos_fn = 'train_pos_' + str(nTrain)   # tên file features train có nhãn thích
train_neg_fn = 'train_neg_' + str(nTrain)    # tên file features train có nhãn không thích
test_pos_fn = 'test_pos_' + str(nTest)    # tên file features test có nhãn thích
test_neg_fn = 'test_neg_' + str(nTest)   # tên file features test có nhãn không thích


# Hàm đọc file và tạo vector features cho các movie demo của bộ train và test
def read_data(data_pos, data_neg, n):
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
        dat[i+len(pos),:] = np.array([int(a[0])+nTrain,int(a[1]),int(a[2])]) # gán giá trị tương ứng vào dòng thứ i của ma trận dat

    # Tạo ma trận thưa thớt để lưu các vector features(mỗi movie review là 1 vector, mỗi vector có nwords chiều):
        ## cột 0(dat[:, 0]) của ma trận dat cho biết đây là features của movie review thứ mấy
        ## cột 1(dat[:, 1]) của ma trận dat cho biết từ đang xét là từ thứ bao nhiêu trong bộ từ điển, từ đó suy ra vị trí trong vector feature
        ## cột 2(dat[:, 2]) của ma trận dat có chứa thông tin về tần suất xuất hiện của từ
        ### Tại 1 hàng của ma trận dat, nếu các giá trị của 3 cột lần lượt là x y z thì giá trị của data[x][y-1] = z

    data = coo_matrix((dat[:, 2], (dat[:, 0], dat[:, 1])), shape = (len(label),nwords))
    return (data,label)


(train_data, train_label)  = read_data(train_pos_fn, train_neg_fn,nTrain)
(test_data, test_label)  = read_data(test_pos_fn, test_neg_fn,nTest)

# clf = naive_bayes.BernoulliNB()
clf = naive_bayes.MultinomialNB()
clf.fit(train_data, train_label)

y_pred = clf.predict(test_data)
print('Training size = %d, accuracy = %.2f%%' % \
      (train_data.shape[0],accuracy_score(test_label, y_pred)*100))