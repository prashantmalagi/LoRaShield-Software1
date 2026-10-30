
import numpy as np
from correction.morse_corrector import MorseCorrector

corrector = MorseCorrector(enabled=True)

if not corrector.is_loaded:
    print("Model failed to load:", corrector._load_error)
    raise SystemExit(1)

tests = [
    ".- -... -.-.",
    "... --- ...",
    ".... . .-.. .-.. ---",
    ".-.-.-",
    "-.-.",
]

for message in tests:
    token_ids = [
        corrector._char_to_id.get(char, 1)
        for char in message
    ]
    token_ids += [0] * (corrector._seq_len - len(token_ids))

    predictions = corrector._model.predict(
        np.array([token_ids], dtype=np.int32),
        verbose=0,
    )[0]

    predicted_ids = predictions.argmax(axis=-1)
    predicted = "".join(
        corrector._id_to_char.get(int(predicted_ids[i]), "?")
        for i in range(len(message))
    )

    print("Input:     ", repr(message))
    print("Prediction:", repr(predicted))
    print("Exact match:", message == predicted)
    print("-" * 35)
