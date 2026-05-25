#!/usr/bin/env python3
import logging
import sys

from serial_receiver import SerialReceiver
from preprocessor import Preprocessor
from inferencer import PhonemeInferencer, CONFIDENCE_THRESHOLD
from tts_engine import TTSEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)

SERIAL_PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
MODEL_PATH  = sys.argv[2] if len(sys.argv) > 2 else '../model/phoneme_classifier.tflite'
TTS_MODEL   = sys.argv[3] if len(sys.argv) > 3 else '/home/pi/piper/models/en_US-lessac-medium.onnx'
SILENCE     = 'silence'


def main():
    logger.info("Init — port=%s model=%s", SERIAL_PORT, MODEL_PATH)
    receiver     = SerialReceiver(SERIAL_PORT)
    preprocessor = Preprocessor()
    inferencer   = PhonemeInferencer(MODEL_PATH)
    tts          = TTSEngine(TTS_MODEL)
    phoneme_buffer = []

    logger.info("Listening. Vibrate laryngophone to start.")
    try:
        while True:
            frame = receiver.receive_frame()
            if frame is None:
                continue

            mfccs              = preprocessor.process(frame)
            phoneme, confidence = inferencer.predict(mfccs)

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            logger.info("Detected: %s (%.2f)", phoneme, confidence)

            if phoneme == SILENCE:
                if phoneme_buffer:
                    text = ' '.join(phoneme_buffer)
                    logger.info("Speaking: '%s'", text)
                    tts.speak(text)
                    phoneme_buffer.clear()
            else:
                phoneme_buffer.append(phoneme)

    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        receiver.close()


if __name__ == '__main__':
    main()
