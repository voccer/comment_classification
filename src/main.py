from src.NLP.DictionaryBuilder import DictionaryBuilder
from src.NLP.FeatureFileBuilder import FeatureFileBuilder
import src.NLP.Setting as Setting
import os

# Tạo file feature
# Feature_Train : trong đó mỗi dòng có 3 phần từ : Thứ tự vb , ID trong từ điển , Số lần xuất hiện


class MakeFeature:
    def makeFeature(number):
        test_neg = Setting.DIR_FEATURE_PATH + "/test_neg_" + str(number) + ".txt"
        test_pos = Setting.DIR_FEATURE_PATH + "/test_pos_" + str(number) + ".txt"
        train_neg = Setting.DIR_FEATURE_PATH + "/train_neg_" + str(number) + ".txt"
        train_pos = Setting.DIR_FEATURE_PATH + "train_pos_" + str(number) + ".txt"

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

        # Xây dựng các file train, test với n bình luận
        array = {2}
        for number in array:
            makeFeature(number=number)

