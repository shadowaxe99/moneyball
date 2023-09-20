
import unittest
from models import player_performance_model
from utils import data_preprocessing

class TestPlayerPerformanceModel(unittest.TestCase):

    def setUp(self):
        self.data = data_preprocessing.load_data('data/player_stats.csv')
        self.model = player_performance_model.PlayerPerformanceModel()

    def test_model_train(self):
        self.model.train(self.data)
        self.assertIsNotNone(self.model.model)

    def test_model_predict(self):
        prediction = self.model.predict(self.data)
        self.assertIsNotNone(prediction)

if __name__ == '__main__':
    unittest.main()
