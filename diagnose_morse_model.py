
import numpy as np
from correction.morse_corrector import MorseCorrector

corrector = MorseCorrector(enabled=True)

if not corrector.is_loaded:
    print("Model failed to load:", corrector._load_error)
    raise SystemExit(1)

tests = [
    ".- -... -.-.",
    ".- .--. -.-.",
    "... --- ...",
]

for message in tests:
    tokens = [
        corrector._char_to_id.get(ch, 1)
        for ch in message
    ]
    tokens += [0] * (corrector._seq_len - len(tokens))

    predictions = corrector._model.predict(
        np.array([tokens], dtype=np.int32),
        verbose=0,
    )[0]

    predicted_ids = predictions.argmax(axis=-1)
    vocabulary = corrector._id_to_char

    print("\nInput:", repr(message))
    print("Position | Input | Prediction | Confidence")

    for i, original in enumerate(message):
        predicted = vocabulary.get(int(predicted_ids[i]), "?")
        confidence = float(predictions[i].max())

        print(
            f"{i:8} | {original!r:5} | "
            f"{predicted!r:10} | {confidence:.3f}"
        )
