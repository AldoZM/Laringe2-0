import numpy as np
import pytest
from preprocessor import Preprocessor, N_MFCC, ADC_MID


@pytest.fixture
def pp():
    return Preprocessor()


def test_normalize_midpoint_is_zero(pp):
    samples = np.full(1024, ADC_MID, dtype=np.float32)
    assert np.allclose(pp.normalize(samples), 0.0, atol=1e-3)


def test_normalize_range_bounded(pp):
    samples = np.random.uniform(0, 4095, 1024).astype(np.float32)
    result = pp.normalize(samples)
    assert result.min() >= -1.0 and result.max() <= 1.0


def test_normalize_max_maps_near_one(pp):
    samples = np.full(1024, 4095.0, dtype=np.float32)
    assert np.all(pp.normalize(samples) > 0.99)


def test_normalize_min_maps_near_minus_one(pp):
    samples = np.full(1024, 0.0, dtype=np.float32)
    assert np.all(pp.normalize(samples) < -0.99)


def test_normalize_output_dtype(pp):
    samples = np.random.uniform(0, 4095, 1024).astype(np.float32)
    assert pp.normalize(samples).dtype == np.float32


def test_extract_mfcc_shape(pp):
    samples = np.random.uniform(-1, 1, 1024).astype(np.float32)
    mfccs = pp.extract_mfcc(samples)
    assert mfccs.ndim == 2
    assert mfccs.shape[1] == N_MFCC


def test_process_pipeline_shape_and_dtype(pp):
    raw = np.random.uniform(0, 4095, 1024).astype(np.float32)
    result = pp.process(raw)
    assert result.ndim == 2
    assert result.shape[1] == N_MFCC
    assert result.dtype == np.float32
