import csv
import time
from colorama import Fore, Style

import os
import pickle
import numpy as np
import onnxruntime as rt

# Load threshold once
THRESHOLD = float(os.getenv("TRESHOLD_PREDICTION", 0.999994))

# ONNX Runtime session optimizations
so = rt.SessionOptions()
so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

# Optional tuning
so.execution_mode = rt.ExecutionMode.ORT_PARALLEL
so.intra_op_num_threads = 0
so.inter_op_num_threads = 0

# Create session
sess = rt.InferenceSession(
    "models/fraud/1/model.onnx",
    sess_options=so,
    providers=rt.get_available_providers()
)

# Load scaler once
with open("artifact/scaler.pkl", "rb") as handle:
    scaler = pickle.load(handle)

# Cache names once
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name


def ask_model(query):
    # Transform once
    transformed = scaler.transform(query).astype(np.float32, copy=False)

    # Run inference
    prediction = sess.run(
        [output_name],
        {input_name: transformed}
    )[0]

    # Extract scalar once
    pred = float(prediction.squeeze())

    # Fast boolean check
    bool_answer = THRESHOLD < pred < 1.0

    # Faster formatting
    perc_answer = f"{pred * 100:.5f}%"

    return bool_answer, perc_answer
def open_all_files_in_folder(folder_path):
    input_data = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', newline='') as f:
                    print(f"Loaded {file_path}")
                    reader = list(csv.reader(f))[1:]
                    input_data += reader
            except Exception as e:
                print(f"Could not read {filename}: {e}")

    # random.shuffle(input_data)
    return input_data

def main():
    data = open_all_files_in_folder(os.getenv("INPUT_FOLDER", "input/"))

    print("Inspecting credit card transactions... (Note: printing the progress slows the process down)")

    i = 0
    for query in data:
        b, p = ask_model([query])
        b_t = "FALSE"
        stop_print=False
        if b:
            b_t = Fore.RED + "TRUE" + Style.RESET_ALL
            stop_print=True
        print(f"\rIs query {i} fraudulent? {b_t}. Likelyhood of fraud: {p}", end='')
        # time.sleep(0.3)
        if stop_print:
            print("")
            time.sleep(1)
        i+=1

if __name__ == "__main__":
    main()