import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from inferencer import PhonemeInferencer, PHONEMES


@pytest.fixture
def inferencer():
    with patch('inferencer.tflite.Interpreter') as MockInterp:
        inst = MockInterp.return_value
        inst.get_input_details.return_value  = [{'index': 0}]
        inst.get_output_details.return_value = [{'index': 1}]
        inf = PhonemeInferencer.__new__(PhonemeInferencer)
        inf.interpreter    = inst
        inf.input_details  = inst.get_input_details()
        inf.output_details = inst.get_output_details()
        yield inf


def make_output(winner_idx, value=0.9):
    out = np.zeros((1, len(PHONEMES)), dtype=np.float32)
    out[0][winner_idx] = value
    return out


def test_predict_returns_phoneme_and_float(inferencer):
    inferencer.interpreter.get_tensor.return_value = make_output(0)
    phoneme, conf = inferencer.predict(np.random.rand(12, 26).astype(np.float32))
    assert isinstance(phoneme, str) and phoneme in PHONEMES
    assert isinstance(conf, float)


def test_predict_argmax_class(inferencer):
    inferencer.interpreter.get_tensor.return_value = make_output(5, 0.95)
    phoneme, conf = inferencer.predict(np.random.rand(12, 26).astype(np.float32))
    assert phoneme == PHONEMES[5]
    assert abs(conf - 0.95) < 1e-5


def test_predict_confidence_in_range(inferencer):
    inferencer.interpreter.get_tensor.return_value = make_output(2, 0.7)
    _, conf = inferencer.predict(np.random.rand(12, 26).astype(np.float32))
    assert 0.0 <= conf <= 1.0
