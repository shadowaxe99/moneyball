
import unittest
from models.player_value_model import PlayerValueModel
from utils.data_preprocessing import preprocess_data

class TestPlayerValueModel(unittest.TestCase):

    def setUp(self):
        self.model = PlayerValueModel()
        self.data = preprocess_data("data/player_stats.csv")

    def test_train(self):
        self.model.train(self.data)
        self.assertIsNotNone(self.model.model)

    def test_predict(self):
        self.model.train(self.data)
        prediction = self.model.predict(self.data.iloc[0])
        self.assertIsNotNone(prediction)

if __name__ == '__main__':
    unittest.main()
