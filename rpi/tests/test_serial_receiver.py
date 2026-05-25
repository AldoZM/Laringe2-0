import numpy as np
import struct
import pytest
from unittest.mock import MagicMock, patch
from serial_receiver import SerialReceiver, FRAME_SAMPLES, FRAME_BYTES


def make_valid_frame(samples=None):
    if samples is None:
        samples = np.random.randint(0, 4095, FRAME_SAMPLES, dtype=np.uint16)
    raw = samples.tobytes()
    checksum = sum(raw) & 0xFFFF
    return raw, struct.pack('<H', checksum), samples


@pytest.fixture
def receiver():
    with patch('serial_receiver.serial.Serial'):
        r = SerialReceiver.__new__(SerialReceiver)
        r.ser = MagicMock()
        yield r


def test_receive_valid_frame_returns_float32_array(receiver):
    raw, checksum, _ = make_valid_frame()
    receiver.ser.read.side_effect = [bytes([0xAA]), bytes([0x55]), raw, checksum]
    result = receiver.receive_frame()
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == FRAME_SAMPLES


def test_bad_checksum_returns_none(receiver):
    raw, _, _ = make_valid_frame()
    receiver.ser.read.side_effect = [bytes([0xAA]), bytes([0x55]), raw, struct.pack('<H', 0x0000)]
    assert receiver.receive_frame() is None


def test_short_read_returns_none(receiver):
    receiver.ser.read.side_effect = [bytes([0xAA]), bytes([0x55]), b'\x00' * 10, b'\x00\x00']
    assert receiver.receive_frame() is None


def test_timeout_on_header_returns_none(receiver):
    receiver.ser.read.return_value = b''
    result = receiver.receive_frame()
    assert result is None


def test_frame_values_preserved(receiver):
    samples = np.arange(FRAME_SAMPLES, dtype=np.uint16)
    raw, checksum, _ = make_valid_frame(samples)
    receiver.ser.read.side_effect = [bytes([0xAA]), bytes([0x55]), raw, checksum]
    result = receiver.receive_frame()
    np.testing.assert_array_equal(result, samples.astype(np.float32))
