#!/usr/bin/env python3
"""Smoke test: verifica que frames del ESP32 llegan y se procesan correctamente."""
import sys
from serial_receiver import SerialReceiver
from preprocessor import Preprocessor

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'

receiver     = SerialReceiver(PORT)
preprocessor = Preprocessor()

print(f"Listening on {PORT}... Ctrl+C to stop.")
print("Vibrate laryngophone to generate signal.\n")

for i in range(5):
    frame = receiver.receive_frame()
    if frame is None:
        print(f"Frame {i+1}: FAILED (None) — check circuit or serial port")
        continue
    mfccs = preprocessor.process(frame)
    print(f"Frame {i+1}: OK  raw=[{frame.min():.0f}–{frame.max():.0f}]  "
          f"MFCC shape={mfccs.shape}  mean={mfccs.mean():.3f}")

receiver.close()
print("\nDone.")
