from docutils.nodes import problematic

from src.CrawlData import SeleniumCrawler
from src.nlp.FeatureFileBuilder import FeatureFileBuilder
from src.ml import Train_Test
import src.Setting as setting
from sklearn import naive_bayes
import pickle
import numpy as np

# crawler = SeleniumCrawler().run_crawler(link="https://www.imdb.com/title/tt5523010/reviews?ref_=tt_ov_rt")
feature = FeatureFileBuilder("/home/toanloi/Documents/comment_classification/Data/Data_Test/test/pos").build_feature_from_folder()
sel = pickle.load((open("../../sel.sav", 'rb')))

n_feature = feature[-1][0] + 1
n_word = 140200

loaded_model = pickle.load((open("../../final_model.sav", 'rb')))
feature = Train_Test.transform_to_coo_matrix(feature, n_feature, n_word)
feature = sel.transform(feature)
probabilities = loaded_model.predict(feature)
print(loaded_model.predict_proba(feature))

pos = np.count_nonzero(probabilities)
neg = np.size(probabilities) - pos
print("Positive : %d \nNegative : %d" %(pos , neg ))

