
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/lorashield_morse_correction_v2.keras"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Vocabulary: PAD=0, UNK=1, .=2, -=3, space=4
messages = [
    ".- -... -.-.",
    "... --- ...",
]

def encode(message, length=120):
    ids = [{"<PAD>": 0, "<UNK>": 1, ".": 2, "-": 3, " ": 4}.get(c, 1)
           for c in message]
    if len(ids) > length:
        raise ValueError("Message too long")
    return ids + [0] * (length - len(ids))

for message in messages:
    x = np.array([encode(message)], dtype=np.int32)
    predictions = model.predict(x, verbose=0)[0]
    predicted_ids = predictions.argmax(axis=-1)

    vocab = {0: "<PAD>", 1: "<UNK>", 2: ".", 3: "-", 4: " "}
    output = "".join(vocab[int(predicted_ids[i])] for i in range(len(message)))

    print("\nInput:     ", repr(message))
    print("Prediction:", repr(output))
    print("Exact match:", output == message)
    print("Mean confidence:", round(float(predictions[:len(message)].max(axis=-1).mean()), 4))
