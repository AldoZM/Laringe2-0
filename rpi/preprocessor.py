import numpy as np
import librosa

SAMPLE_RATE = 8000
N_MFCC      = 26
N_FFT       = 512
HOP_LENGTH  = 80    # 10ms @ 8kHz
WIN_LENGTH  = 200   # 25ms @ 8kHz
ADC_MID     = 2047.5


class Preprocessor:
    def normalize(self, raw: np.ndarray) -> np.ndarray:
        return ((raw - ADC_MID) / ADC_MID).astype(np.float32)

    def extract_mfcc(self, samples: np.ndarray) -> np.ndarray:
        mfccs = librosa.feature.mfcc(
            y=samples,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
        )
        return mfccs.T.astype(np.float32)  # [time_steps, N_MFCC]

    def process(self, raw: np.ndarray) -> np.ndarray:
        return self.extract_mfcc(self.normalize(raw))
