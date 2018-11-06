import os

DIR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DIR_TRAIN_PATH = os.path.join(DIR_PATH, "Data/Data_Full/train")
DIR_TEST_PATH = os.path.join(DIR_PATH, "Data/Data_Full/test")
DIR_TRAIN_PATH_NEG = os.path.join(DIR_TRAIN_PATH, "neg")
DIR_TRAIN_PATH_POS = os.path.join(DIR_TRAIN_PATH, "pos")
DIR_TEST_PATH_NEG = os.path.join(DIR_TEST_PATH, "neg")
DIR_TEST_PATH_POS = os.path.join(DIR_TEST_PATH, "pos")
DIR_DICTIONARY = os.path.join(DIR_PATH, "Data/dictionary.txt")

SPECIAL_CHARACTER = '0123456789%@$.,=+-!;/()*"&^:#|~_{}[]\n\t\'\"`‘\\ <>?'
STOP_WORD = os.path.join(DIR_PATH, "Data/stopword.txt")