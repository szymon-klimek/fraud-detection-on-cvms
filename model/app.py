import numpy as np
import onnxruntime as rt
import pickle
import os
import csv
import time
from colorama import Fore, Style

# Load ONNX model
sess = rt.InferenceSession("models/fraud/1/model.onnx", providers=rt.get_available_providers())

# Load scaler
with open('artifact/scaler.pkl', 'rb') as handle:
    scaler = pickle.load(handle)

input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

BATCH_SIZE = 5000
MAX_RECORDS = 100000

def ask_model_batch(queries):
    """Process a batch of queries at once for better performance."""
    queries_array = np.array(queries, dtype=np.float32)
    scaled = scaler.transform(queries_array).astype(np.float32)
    predictions = sess.run([output_name], {input_name: scaled})[0]
    
    threshold = float(os.getenv("TRESHOLD_PREDICTION", 0.999994))
    predictions_squeezed = np.squeeze(predictions)
    
    # Handle single prediction case
    if predictions_squeezed.ndim == 0:
        predictions_squeezed = np.array([predictions_squeezed])
    
    bool_answers = (predictions_squeezed > threshold) & (predictions_squeezed < 1)
    perc_answers = ["{:.5f}%".format(100 * p) for p in predictions_squeezed]
    
    return list(zip(bool_answers, perc_answers, predictions_squeezed))

def open_file(file_path):
    input_data = []

    if os.path.isfile(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                print(f"Loaded {file_path}")
                reader = list(csv.reader(f))[1:]
                input_data += reader
        except Exception as e:
            print(f"Could not read {file_path}: {e}")

    # random.shuffle(input_data)
    return input_data

def main():
    data = open_file(os.getenv("INPUT", "input/"))
    data = data[:MAX_RECORDS]  # Limit to MAX_RECORDS
    total_records = len(data)
    
    print(f"Inspecting {total_records} credit card transactions in batches of {BATCH_SIZE}...")

    fraud_count = 0
    start_time = time.time()
    
    # Process in batches
    for batch_start in range(0, total_records, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_records)
        batch_data = data[batch_start:batch_end]
        
        results = ask_model_batch(batch_data)

        # Process results for this batch
        batch_frauds = []
        for idx, (is_fraud, perc, raw_pred) in enumerate(results):
            global_idx = batch_start + idx
            if is_fraud:
                fraud_count += 1
                batch_frauds.append((global_idx, perc))

        # Print any fraudulent transactions found in this batch
        for fraud_idx, fraud_perc in batch_frauds:
            print(f"{Fore.RED}FRAUD DETECTED{Style.RESET_ALL} at record {fraud_idx}, likelihood: {fraud_perc}")
    
    print(f"Total fraudulent transactions detected: {Fore.RED}{fraud_count}{Style.RESET_ALL}/{total_records}")

if __name__ == "__main__":
    main()