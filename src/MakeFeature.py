from _ast import Set

from src.NLP.DictionaryBuilder import DictionaryBuilder
from src.NLP.FeatureFileBuilder import FeatureFileBuilder
import src.NLP.Setting as Setting
import os

# Tạo file feature
# Feature_Train : trong đó mỗi dòng có 3 phần từ : Thứ tự vb , ID trong từ điển , Số lần xuất hiện


class MakeFeature:
    def makeFeature(self, number):
        path_feature_local = Setting.DIR_FEATURE_PATH+"/feature_"+str(number)
        if not (os.path.isdir(path_feature_local)):
            os.mkdir(path_feature_local)
        test_neg = path_feature_local + "/test_neg_" + str(number)
        test_pos = path_feature_local + "/test_pos_" + str(number)
        train_neg = path_feature_local + "/train_neg_" + str(number)
        train_pos = path_feature_local + "/train_pos_" + str(number)

        if not os.path.isfile(test_neg):
            print("Building Test Negative Feature " + str(number))
            FeatureFileBuilder(folder_path=Setting.DIR_TEST_PATH+"/neg/", number=number, file_target=test_neg)
        if not os.path.isfile(test_pos):
            print("Building Test Positive Feature " + str(number))
            FeatureFileBuilder(folder_path=Setting.DIR_TEST_PATH+"/pos/", number=number, file_target=test_pos)
        if not os.path.isfile(train_neg):
            print("Building Train Negative Feature " + str(number))
            FeatureFileBuilder(folder_path=Setting.DIR_TRAIN_PATH+"/neg/", number=number, file_target=train_neg)
        if not os.path.isfile(train_pos):
            print("Building Train Positive Feature " + str(number))
            FeatureFileBuilder(folder_path=Setting.DIR_TRAIN_PATH+"/pos/", number=number, file_target=train_pos)

    def makeMyFeature(self):
        path_feature_local = Setting.DIR_PATH + "/Data/MyFeature"
        if not (os.path.isdir(path_feature_local)):
            os.mkdir(path_feature_local)
        test_neg = path_feature_local + "/test_neg"
        test_pos = path_feature_local + "/test_pos"
        if not os.path.isfile(test_neg):
            print("Building Test My Negative Feature")
            FeatureFileBuilder(folder_path=Setting.DIR_PATH+"/Data/MyData/neg/", number=100000, file_target=test_neg)
        if not os.path.isfile(test_pos):
            print("Building Test My Positive Feature")
            FeatureFileBuilder(folder_path=Setting.DIR_PATH+"/Data/MyData/pos/", number=100000, file_target=test_pos)

class main:
    if __name__ == '__main__':
        # Load dữ liệu từ các file train
        # Xây dựng bộ từ điển từ
        # Nếu đã có sẵn từ điển thì không thực hiện bước này
        if not os.path.isfile(Setting.DIR_DICTIONARY):
            print("Building Dictionary")
            DictionaryBuilder()

        if not (os.path.isdir(Setting.DIR_FEATURE_PATH)):
            os.mkdir(Setting.DIR_FEATURE_PATH)

        # Xây dựng các file train, test với n bình luận
        mk = MakeFeature()
        array = {100}
        for x in array:
            print("Buiding feature " + str(x))
            mk.makeFeature(x)

        mk.makeMyFeature()



