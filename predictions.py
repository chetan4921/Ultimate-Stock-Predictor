import numpy as np

def generate_future_predictions(model, last_seq, days_to_predict, feature_count):
    predictions_scaled = []
    input_seq = last_seq.reshape(1, last_seq.shape[0], feature_count)
    for _ in range(days_to_predict):
        pred = model.predict(input_seq)
        predictions_scaled.append(pred[0,0])
        next_row = np.append(input_seq[:,1:,:], [[np.append(input_seq[0,-1,:-1], pred[0,0])]], axis=1)
        input_seq = next_row
    return np.array(predictions_scaled).reshape(-1,1)

def inverse_transform(scaler, predictions, feature_index):
    min_val = scaler.data_min_[feature_index]
    max_val = scaler.data_max_[feature_index]
    return predictions * (max_val - min_val) + min_val
