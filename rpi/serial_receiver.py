import struct
import logging
import serial
import numpy as np

FRAME_SAMPLES    = 1024
BYTES_PER_SAMPLE = 2
FRAME_BYTES      = FRAME_SAMPLES * BYTES_PER_SAMPLE
CHECKSUM_BYTES   = 2

logger = logging.getLogger(__name__)


class SerialReceiver:
    def __init__(self, port: str, baudrate: int = 921600, timeout: float = 1.0):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)

    def _wait_for_header(self) -> bool:
        while True:
            b = self.ser.read(1)
            if not b:
                return False
            if b[0] == 0xAA:
                b2 = self.ser.read(1)
                if b2 and b2[0] == 0x55:
                    return True

    def receive_frame(self) -> np.ndarray | None:
        if not self._wait_for_header():
            return None
        raw = self.ser.read(FRAME_BYTES)
        if len(raw) != FRAME_BYTES:
            logger.warning("Short read: got %d bytes, expected %d", len(raw), FRAME_BYTES)
            return None
        checksum_raw = self.ser.read(CHECKSUM_BYTES)
        if len(checksum_raw) != CHECKSUM_BYTES:
            return None
        expected = sum(raw) & 0xFFFF
        received = struct.unpack('<H', checksum_raw)[0]
        if expected != received:
            logger.warning("Checksum mismatch: expected %04X got %04X", expected, received)
            return None
        return np.frombuffer(raw, dtype=np.uint16).astype(np.float32)

    def close(self):
        self.ser.close()
