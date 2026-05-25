import pytest
from unittest.mock import patch, MagicMock
from tts_engine import TTSEngine


@pytest.fixture
def tts():
    return TTSEngine('/fake/model.onnx')


def test_synthesize_calls_piper_with_model(tts):
    with patch('tts_engine.subprocess.run') as mock_run, \
         patch('tts_engine.tempfile.mktemp', return_value='/tmp/t.wav'):
        mock_run.return_value = MagicMock(returncode=0)
        path = tts.synthesize_to_file('hello')
        assert path == '/tmp/t.wav'
        args = mock_run.call_args[0][0]
        assert 'piper' in args
        assert '/fake/model.onnx' in args


def test_synthesize_raises_on_piper_failure(tts):
    with patch('tts_engine.subprocess.run') as mock_run, \
         patch('tts_engine.tempfile.mktemp', return_value='/tmp/t.wav'):
        mock_run.return_value = MagicMock(returncode=1, stderr=b'error')
        with pytest.raises(RuntimeError, match='Piper failed'):
            tts.synthesize_to_file('hello')


def test_speak_calls_aplay(tts):
    with patch('tts_engine.subprocess.run') as mock_run, \
         patch('tts_engine.tempfile.mktemp', return_value='/tmp/t.wav'), \
         patch('tts_engine.os.path.exists', return_value=True), \
         patch('tts_engine.os.unlink'):
        mock_run.return_value = MagicMock(returncode=0)
        tts.speak('hello')
        all_calls = [c[0][0] for c in mock_run.call_args_list]
        assert any('aplay' in c for c in all_calls)
