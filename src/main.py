from src.NLP.DictionaryBuilder import DictionaryBuilder
from src.NLP.FeatureFileBuilder import FeatureFileBuilder
import src.NLP.Setting as Setting
import os
class main:
    # Load dữ liệu từ các file train
    # Xây dựng bộ từ điển từ
    # Nếu đã có sẵn từ điển thì không thực hiện bước này
    if not os.path.isfile(Setting.DIR_DICTIONARY):
        print("Building Dictionary")
        DictionaryBuilder()

    # Load dữ liệu từ bộ train
    # Xây dựng bộ test từ các file train
    # Feature_Train : trong đó mỗi dòng có 3 phần từ : Thứ tự vb , ID trong từ điển , Số lần xuất hiện
    # Label_Train : mỗi dòng chứa 1 hoặc 0 là xác nhận neg hay pos
    if not os.path.isfile(Setting.DIR_TEST_PATH + "/feature.txt"):
        print("Building Test Feature")
        FeatureFileBuilder(folder_path=Setting.DIR_TEST_PATH)
    if not os.path.isfile(Setting.DIR_TRAIN_PATH + "/feature.txt"):
        print("Building Train Feature")
        FeatureFileBuilder(folder_path=Setting.DIR_TRAIN_PATH)

    # Load dữ liệu từ file test
    # Xây dựng bộ test từ các file test
    # Feature_Test : trong đó mỗi dòng có 3 phần tử : Thứ tự vb , ID trong từ điển , Số lần xuất hiện
    # Label_Test : mỗi dòng chứa 1 hoặc 0 là xác nhận neg hay pos
