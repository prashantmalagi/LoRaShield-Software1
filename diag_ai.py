"""
LoRaShield - One-shot AI diagnostic script (v2).
Run: .venv\Scripts\python.exe diag_ai.py
"""
import sys, os, traceback, pickle, warnings
sys.path.insert(0, '.')
from pathlib import Path

BASE_DIR       = Path(__file__).resolve().parent
MODEL_PATH     = BASE_DIR / 'models' / 'morse_decoder_v2.keras'
TOKENIZER_PATH = BASE_DIR / 'models' / 'tokenizer.pkl'

sep = '-' * 60
print(sep)
print('Loading AI Model (v2 - Seq2Seq Denoising)...')
print(f'Model path:       {MODEL_PATH}')
print(f'Tokenizer path:   {TOKENIZER_PATH}')
print(f'CWD:              {os.getcwd()}')
print()
print(f'Model exists:     {MODEL_PATH.exists()}')
print(f'Tokenizer exists: {TOKENIZER_PATH.exists()}')
if MODEL_PATH.exists():
    print(f'Model file size:  {MODEL_PATH.stat().st_size:,} bytes')
print(sep)

# TensorFlow
print('\nImporting TensorFlow...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
try:
    import tensorflow as tf
    print(f'TensorFlow Version: {tf.__version__}')
    print(f'Keras version:      {tf.keras.__version__}')
except Exception:
    print('FAILED to import TensorFlow:')
    traceback.print_exc()
    sys.exit(1)

# Tokenizer
print('\nLoading tokenizer...')
try:
    with open(TOKENIZER_PATH, 'rb') as fh:
        tok = pickle.load(fh)
    print(f'Tokenizer Type:    {type(tok).__name__}')
    print(f'Word Index:        {tok.word_index}')
    print(f'Vocabulary Size:   {len(tok.word_index)}')
except Exception:
    print('FAILED to load tokenizer:')
    traceback.print_exc()
    sys.exit(1)

# Keras model (.keras format)
print('\nLoading Keras model (.keras)...')
try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    print(f'Model Input Shape:  {model.input_shape}')
    print(f'Model Output Shape: {model.output_shape}')
    print(f'Model Type:         Seq2Seq Denoising (BiLSTM)')
    print('\nSUCCESS: Model loaded correctly.')
except Exception:
    print('\nFAILED to load model - full traceback:')
    traceback.print_exc()
    sys.exit(1)

# Quick inference test
print('\nRunning quick inference test...')
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

IDX_TO_MORSE = {0: '', 1: '.', 2: '-', 3: ' '}

MORSE_REVERSE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
}

def decode_pipeline(noisy_morse):
    seqs = tok.texts_to_sequences([noisy_morse])
    padded = pad_sequences(seqs, maxlen=108, padding='post', truncating='post')
    pred = model.predict(padded, verbose=0)
    pred_idx = np.argmax(pred[0], axis=-1)
    active = min(len(seqs[0]), 108)
    corrected = ''.join(IDX_TO_MORSE.get(int(i), '') for i in pred_idx[:active]).strip()
    # Decode corrected morse
    words = corrected.split(' / ') if ' / ' in corrected else [corrected]
    decoded = ' '.join(
        ''.join(MORSE_REVERSE.get(c.strip(), '?') for c in w.strip().split(' ') if c.strip())
        for w in words if w.strip()
    )
    conf = float(np.mean(np.max(pred[0][:active], axis=-1)))
    return corrected, decoded, conf

test_cases = [
    '... --- ...',
    '.- -... -.-.',
]
for tc in test_cases:
    corrected, decoded, conf = decode_pipeline(tc)
    print(f'  Input:     {repr(tc)}')
    print(f'  Corrected: {repr(corrected)}')
    print(f'  Decoded:   {repr(decoded)}')
    print(f'  Conf:      {conf:.3f}')
    print()

print('Diagnostic complete.')
