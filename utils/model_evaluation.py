
import tensorflow as tf

def evaluate_model(model, test_data, test_labels):
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=2)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

def predict_model(model, new_data):
    predictions = model.predict(new_data)
    return predictions
