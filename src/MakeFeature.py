from _ast import Set

from src.NLP.DictionaryBuilder import DictionaryBuilder
from src.NLP.FeatureFileBuilder import FeatureFileBuilder
import src.NLP.Setting as Setting
import os

# Tạo file feature
# Feature_Train : trong đó mỗi dòng có 3 phần từ : Thứ tự vb , ID trong từ điển , Số lần xuất hiện


class MakeFeature:
    def makeFeature(number):
        path_feature_local = Setting.DIR_FEATURE_PATH+"/feature_"+str(number)
        if not (os.path.isdir(path_feature_local)):
            os.mkdir(path_feature_local)
        test_neg = path_feature_local + "/test_neg_" + str(number) + ".txt"
        test_pos = path_feature_local + "/test_pos_" + str(number) + ".txt"
        train_neg = path_feature_local + "/train_neg_" + str(number) + ".txt"
        train_pos = path_feature_local + "train_pos_" + str(number) + ".txt"

        if not os.path.isfile(test_neg):
            print("Building Test Positive Feature " + str(number))
            FeatureFileBuilder(folder_path=Setting.DIR_TEST_PATH+"/neg/", number=number)
        if not os.path.isfile(test_pos):
            print("Building Test Negative Fearture " + str(number))
            FeatureFileBuilder(folder_path=Setting.DIR_TEST_PATH+"/pos/", number=number)
        if not os.path.isfile(train_neg):
            print("Building Train Positive Feature " + str(number))
            FeatureFileBuilder(folder_path=Setting.DIR_TRAIN_PATH+"/neg/", number=number)
        if not os.path.isfile(train_pos):
            print("Building Train Negative Feature " + str(number))
            FeatureFileBuilder(folder_path=Setting.DIR_TRAIN_PATH+"/pos/", number=number)

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
        array = {12500, 5000, 3000, 1000, 500, 300, 100}
        for number in array:
            print("Buiding feature " + str(number))
            makeFeature(number=number)

