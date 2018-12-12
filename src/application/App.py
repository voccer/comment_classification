from src.nlp.FeatureFileBuilder import FeatureFileBuilder
from src.ml import Train_Test
import pickle
import numpy as np

# Train_Test.test()
# crawler = SeleniumCrawler().run_crawler(link="https://www.imdb.com/title/tt5523010/reviews?ref_=tt_ov_rt")
feature = FeatureFileBuilder("../../Data/MyData/neg").build_feature_from_folder()

n_feature = feature[-1][0] + 1
n_word = 140200

loaded_model = pickle.load((open("../../final_model.sav", 'rb')))
feature = Train_Test.transform_to_coo_matrix(feature, n_feature, n_word)
probabilities = loaded_model.predict(feature)

pos = np.count_nonzero(probabilities)
neg = np.size(probabilities) - pos
print("Positive : %d \nNegative : %d" %(pos , neg ))

