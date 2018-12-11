from src.nlp.DictionaryBuilder import DictionaryBuilder
from src.nlp.FeatureFileBuilder import FeatureFileBuilder
from src.FileHandler import FileWriter
import src.Setting as Setting
import os

# Tạo file feature
# Feature_Train : trong đó mỗi dòng có 3 phần từ : Thứ tự vb , ID trong từ điển , Số lần xuất hiện


class MakeFeature:
    def makeFeature(self, number):
        path_feature_local = Setting.DIR_FEATURE_PATH + "/feature_" + str(number)
        if not (os.path.isdir(path_feature_local)):
            os.mkdir(path_feature_local)
        test_neg = path_feature_local + "/test_neg_" + str(number)
        test_pos = path_feature_local + "/test_pos_" + str(number)
        train_neg = path_feature_local + "/train_neg_" + str(number)
        train_pos = path_feature_local + "/train_pos_" + str(number)

        if not os.path.isfile(test_neg):
            print("Building Test Negative Feature " + str(number))
            f = FeatureFileBuilder(folder_path=Setting.DIR_TEST_PATH + "/neg/", number=number)
            f = f.build_feature_from_folder()
            FileWriter(filepath=test_neg, Data=f).write_feature()

        if not os.path.isfile(test_pos):
            print("Building Test Positive Feature " + str(number))
            f = FeatureFileBuilder(folder_path=Setting.DIR_TEST_PATH + "/pos/", number=number)
            f = f.build_feature_from_folder()
            FileWriter(filepath=test_pos, Data=f).write_feature()

        if not os.path.isfile(train_neg):
            print("Building Train Negative Feature " + str(number))
            f = FeatureFileBuilder(folder_path=Setting.DIR_TRAIN_PATH + "/neg/", number=number)
            f = f.build_feature_from_folder()
            FileWriter(filepath=train_neg, Data=f).write_feature()

        if not os.path.isfile(train_pos):
            print("Building Train Positive Feature " + str(number))
            f = FeatureFileBuilder(folder_path=Setting.DIR_TRAIN_PATH + "/pos/", number=number)
            f = f.build_feature_from_folder()
            FileWriter(filepath=train_pos, Data=f).write_feature()

    def makeMyFeature(self):
        path_feature_local = Setting.DIR_PATH + "/Data/MyFeature"
        if not (os.path.isdir(path_feature_local)):
            os.mkdir(path_feature_local)
        test_neg = path_feature_local + "/test_neg"
        test_pos = path_feature_local + "/test_pos"
        if not os.path.isfile(test_neg):
            print("Building Test My Negative Feature")
            f = FeatureFileBuilder(folder_path=Setting.DIR_PATH + "/Data/MyData/neg/")
            f = f.build_feature_from_folder()
            FileWriter(filepath=test_neg, Data=f).write_feature()

        if not os.path.isfile(test_pos):
            print("Building Test My Positive Feature")
            f = FeatureFileBuilder(folder_path=Setting.DIR_PATH + "/Data/MyData/pos/")
            f = f.build_feature_from_folder()
            FileWriter(filepath=test_pos, Data=f).write_feature()

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
        array = {100, 200, 500, 1000, 2000, 5000, 12500}
        for x in array:
            print("Buiding feature " + str(x))
            mk.makeFeature(x)

        mk.makeMyFeature()