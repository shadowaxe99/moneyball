
import tensorflow as tf
from models import player_performance_model, player_value_model, team_composition_model
from utils import data_preprocessing, model_evaluation

def main():
    # Load data
    player_stats = data_preprocessing.load_data("data/player_stats.csv")
    team_composition = data_preprocessing.load_data("data/team_composition.csv")

    # Preprocess data
    player_stats = data_preprocessing.preprocess_data(player_stats)
    team_composition = data_preprocessing.preprocess_data(team_composition)

    # Train models
    player_performance_model.train(player_stats)
    player_value_model.train(player_stats)
    team_composition_model.train(team_composition)

    # Evaluate models
    model_evaluation.evaluate(player_performance_model, player_stats)
    model_evaluation.evaluate(player_value_model, player_stats)
    model_evaluation.evaluate(team_composition_model, team_composition)

if __name__ == "__main__":
    main()
