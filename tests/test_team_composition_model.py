
import unittest
from models import team_composition_model
from utils import data_preprocessing

class TestTeamCompositionModel(unittest.TestCase):

    def setUp(self):
        self.data = data_preprocessing.load_data('data/team_composition.csv')
        self.model = team_composition_model.TeamCompositionModel()

    def test_model_train(self):
        self.model.train(self.data)
        self.assertIsNotNone(self.model.model)

    def test_model_predict(self):
        prediction = self.model.predict(self.data)
        self.assertIsNotNone(prediction)

    def test_model_evaluate(self):
        score = self.model.evaluate(self.data)
        self.assertIsNotNone(score)

if __name__ == '__main__':
    unittest.main()

