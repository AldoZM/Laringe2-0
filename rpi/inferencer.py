import numpy as np
import tflite_runtime.interpreter as tflite

PHONEMES = [
    'p', 'b', 't', 'd', 'k', 'g', 'f', 'v',
    'th_unvoiced', 'th_voiced', 's', 'z', 'sh', 'zh', 'h', 'ch', 'jh',
    'm', 'n', 'ng', 'l', 'r', 'w', 'y',
    'iy', 'ih', 'ey', 'eh', 'ae', 'aa', 'ao', 'ow', 'uh', 'uw',
    'ah', 'er', 'ax', 'ay', 'aw', 'oy',
    'silence', 'unknown', 'dx', 'nx',
]

CONFIDENCE_THRESHOLD = 0.7


class PhonemeInferencer:
    def __init__(self, model_path: str):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, mfccs: np.ndarray) -> tuple[str, float]:
        input_data = np.expand_dims(mfccs, axis=0).astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output     = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        idx        = int(np.argmax(output))
        confidence = float(output[idx])
        return PHONEMES[idx], confidence
