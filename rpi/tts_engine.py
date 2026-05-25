import os
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def synthesize_to_file(self, text: str) -> str:
        out_path = tempfile.mktemp(suffix='.wav')
        result = subprocess.run(
            ['piper', '--model', self.model_path, '--output_file', out_path],
            input=text.encode(),
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Piper failed: {result.stderr.decode()}")
        return out_path

    def speak(self, text: str) -> None:
        wav_path = self.synthesize_to_file(text)
        try:
            subprocess.run(['aplay', wav_path], capture_output=True, timeout=30)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
