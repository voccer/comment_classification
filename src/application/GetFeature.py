from src.nlp.FeatureFileBuilder import FeatureFileBuilder
import src.Setting as setting

class GetFeature:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def get_feature(self):
        f = FeatureFileBuilder(folder_path=self.folder_path).build_feature_from_folder()
        print(f)

if __name__ == '__main__':
    GetFeature(setting.DIR_APPLY_PATH).get_feature()
