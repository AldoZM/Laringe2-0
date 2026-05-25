# Laringófono — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working vocal assistance prototype that captures neck vibrations from a Baofeng K-Type laryngophone, classifies English phonemes via TFLite on RPi 5, and synthesizes speech via Piper TTS through an integrated speaker.

**Architecture:** ESP32 handles analog signal capture (ADC @ 8kHz via hardware timer) and transmits raw PCM frames over USB Serial to the RPi 5. The RPi 5 runs a Python pipeline: MFCC extraction → TFLite phoneme classifier → Piper TTS → audio output via PAM8403 amplifier.

**Tech Stack:** Arduino C++ (ESP32), Python 3.11 (RPi 5), librosa, tflite-runtime, piper-tts, pyserial, pyaudio, pytest, Edge Impulse Studio (dataset + training).

---

## File Map

```
D:\Codigo Abierto\Laringofono\
├── firmware/
│   └── esp32/
│       ├── esp32.ino          # Main sketch: ADC timer + serial TX
│       └── config.h           # Pin, sample rate, frame constants
├── rpi/
│   ├── requirements.txt       # Python dependencies
│   ├── serial_receiver.py     # Read frames from ESP32 over USB Serial
│   ├── preprocessor.py        # Normalize ADC + extract MFCC
│   ├── inferencer.py          # TFLite phoneme classifier wrapper
│   ├── tts_engine.py          # Piper TTS wrapper → aplay
│   ├── main.py                # Orchestrator: receive → infer → speak
│   └── tests/
│       ├── test_serial_receiver.py
│       ├── test_preprocessor.py
│       ├── test_inferencer.py
│       └── test_tts_engine.py
└── model/
    └── phoneme_classifier.tflite   # Exported after training (Task 11)
```

---

## Task 1: Initialize Git Repository and Project Structure

**Files:**
- Create: `firmware/esp32/config.h`
- Create: `firmware/esp32/esp32.ino` (skeleton)
- Create: `rpi/requirements.txt`
- Create: `.gitignore`

- [ ] **Step 1: Init git repo**

```bash
cd "D:\Codigo Abierto\Laringofono"
git init
```

Expected: `Initialized empty Git repository in D:/Codigo Abierto/Laringofono/.git/`

- [ ] **Step 2: Create .gitignore**

Create `D:\Codigo Abierto\Laringofono\.gitignore`:
```
__pycache__/
*.pyc
*.tflite
.venv/
*.wav
*.egg-info/
.pytest_cache/
model/phoneme_classifier.tflite
```

- [ ] **Step 3: Create config.h**

Create `firmware/esp32/config.h`:
```cpp
#pragma once

#define ADC_PIN         34        // GPIO34 = ADC1_CH6 (input-only, safe)
#define SAMPLE_RATE     8000      // Hz
#define FRAME_SIZE      1024      // samples per frame
#define BAUD_RATE       921600
#define HEADER_BYTE1    0xAA
#define HEADER_BYTE2    0x55
```

- [ ] **Step 4: Create empty esp32.ino skeleton**

Create `firmware/esp32/esp32.ino`:
```cpp
#include "config.h"
#include "driver/adc.h"

// Populated in Task 4
void setup() {}
void loop() {}
```

- [ ] **Step 5: Create requirements.txt**

Create `rpi/requirements.txt`:
```
pyserial==3.5
numpy==1.26.4
librosa==0.10.2
tflite-runtime==2.14.0
pyaudio==0.2.14
pytest==8.1.0
pytest-mock==3.14.0
```

- [ ] **Step 6: Initial commit**

```bash
git add .
git commit -m "chore: initialize project structure"
```

---

## Task 2: Assemble Signal Conditioning Circuit

**Hardware task — no code. All connections on breadboard.**

**Components needed:** LM358 DIP-8, R1 10kΩ, R2 10kΩ, Rf 390kΩ, Rin 10kΩ, Rout 1kΩ, C1 10µF electrolytic, Cout 47nF ceramic, Baofeng K-Type cable, ESP32 DevKit, breadboard, jumper cables.

- [ ] **Step 1: Place LM358 on breadboard**

Straddle center gap. Pin 1 top-left. DIP-8 pinout:
```
Pin 1 = OUT1    Pin 8 = VCC
Pin 2 = IN1-    Pin 7 = OUT2
Pin 3 = IN1+    Pin 6 = IN2-
Pin 4 = GND     Pin 5 = IN2+
```
Only use amplifier 1 (pins 1-4 + 8).

- [ ] **Step 2: Power the LM358**

```
LM358 pin 8 (VCC) → ESP32 3V3 pin
LM358 pin 4 (GND) → ESP32 GND pin
```

- [ ] **Step 3: Build bias voltage divider**

```
ESP32 3V3 → R1 (10kΩ) → nodo A
nodo A    → R2 (10kΩ) → GND
nodo A    → LM358 pin 3 (+)
```
Nodo A = ~1.65V (VCC/2). Medir con multímetro antes de continuar.

- [ ] **Step 4: Connect AC coupling capacitor**

```
Baofeng MIC+ cable → C1 (10µF, positivo hacia MIC+) → nodo A
Baofeng GND cable  → GND (común con ESP32)
```
**⚠ Polaridad C1:** positivo (+) hacia el laringófono, negativo (–) hacia nodo A.

- [ ] **Step 5: Build feedback network (ganancia 40x)**

```
LM358 pin 1 (OUT) → Rf (390kΩ) → LM358 pin 2 (-)
LM358 pin 2 (-)   → Rin (10kΩ) → GND
```

- [ ] **Step 6: Connect anti-aliasing output filter**

```
LM358 pin 1 (OUT) → Rout (1kΩ) → ESP32 GPIO34
ESP32 GPIO34      → Cout (47nF) → GND
```

---

## Task 3: Verify Circuit with Multimeter

**Hardware verification — no code.**

- [ ] **Step 1: Verificar voltaje de bias**

Con ESP32 conectado y alimentado (USB):
- Multímetro en DC Voltage
- Medir nodo A respecto a GND
- Esperado: **1.60V – 1.70V**
- Si <1.5V o >1.8V: revisar R1/R2, verificar conexión 3V3

- [ ] **Step 2: Verificar alimentación LM358**

- Medir LM358 pin 8 vs GND → esperado: 3.3V
- Medir LM358 pin 4 vs GND → esperado: 0V

- [ ] **Step 3: Verificar output en reposo**

- Medir LM358 pin 1 (OUT) vs GND sin hablar
- Esperado: ~1.65V ± 0.2V (centrado sin señal)
- Si satura (0V o 3.3V): revisar Rf/Rin, verificar que C1 esté bien polarizado

- [ ] **Step 4: Verificar señal con voz/vibración**

- Hablar o tocar el laringófono físicamente
- Medir GPIO34 con multímetro en AC
- Debe mostrar variación (aunque sea pequeña)

---

## Task 4: ESP32 Firmware — ADC Sampling con Timer

**Files:**
- Modify: `firmware/esp32/esp32.ino`

- [ ] **Step 1: Escribir firmware completo**

Reemplazar contenido de `firmware/esp32/esp32.ino`:
```cpp
#include "config.h"
#include "driver/adc.h"

uint16_t frame_buffer[FRAME_SIZE];
volatile uint16_t write_idx  = 0;
volatile bool     frame_ready = false;

hw_timer_t *timer = NULL;
portMUX_TYPE timer_mux = portMUX_INITIALIZER_UNLOCKED;

void IRAM_ATTR onTimer() {
  portENTER_CRITICAL_ISR(&timer_mux);
  if (!frame_ready) {
    frame_buffer[write_idx++] = (uint16_t)adc1_get_raw(ADC1_CHANNEL_6);
    if (write_idx >= FRAME_SIZE) {
      frame_ready = true;
      write_idx   = 0;
    }
  }
  portEXIT_CRITICAL_ISR(&timer_mux);
}

static void send_frame() {
  uint32_t checksum = 0;
  for (int i = 0; i < FRAME_SIZE; i++) checksum += frame_buffer[i];
  uint16_t chk16 = (uint16_t)(checksum & 0xFFFF);

  Serial.write(HEADER_BYTE1);
  Serial.write(HEADER_BYTE2);
  Serial.write((uint8_t*)frame_buffer, FRAME_SIZE * 2);
  Serial.write((uint8_t*)&chk16, 2);
}

void setup() {
  Serial.begin(BAUD_RATE);

  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(ADC1_CHANNEL_6, ADC_ATTEN_DB_11);

  // Timer 0, prescaler 80 → 1MHz tick, cuenta hasta 125 → 8kHz
  timer = timerBegin(0, 80, true);
  timerAttachInterrupt(timer, &onTimer, true);
  timerAlarmWrite(timer, 125, true);
  timerAlarmEnable(timer);
}

void loop() {
  if (frame_ready) {
    send_frame();
    portENTER_CRITICAL(&timer_mux);
    frame_ready = false;
    portEXIT_CRITICAL(&timer_mux);
  }
}
```

- [ ] **Step 2: Abrir proyecto en Arduino IDE**

- Arduino IDE → File → Open → `firmware/esp32/esp32.ino`
- Board: "ESP32 Dev Module"
- Tools → Board → ESP32 Arduino → ESP32 Dev Module
- Upload Speed: 921600
- Port: el que aparezca al conectar ESP32

- [ ] **Step 3: Compilar (sin subir)**

- Sketch → Verify/Compile
- Esperado: compilación exitosa sin errores
- Si error `adc1_get_raw not found`: agregar en Arduino IDE → File → Preferences → Additional boards y asegurarse de tener ESP32 board package instalado

- [ ] **Step 4: Subir firmware al ESP32**

- Sketch → Upload
- Esperar: "Done uploading"

- [ ] **Step 5: Verificar en Serial Plotter**

- Tools → Serial Plotter, 921600 baud
- Conectar laringófono al circuito
- Tocar/vibrar el laringófono con la garganta
- Esperado: curva que oscila, no línea plana ni saturada en 0 o 4095
- Si línea plana en ~2047: señal llega pero sin amplificación suficiente → verificar Rf (390kΩ)
- Si saturada en 4095: ganancia excesiva → probar Rf=100kΩ

- [ ] **Step 6: Commit**

```bash
git add firmware/
git commit -m "feat: ESP32 ADC sampling at 8kHz via hw timer + serial frame TX"
```

---

## Task 5: Setup Raspberry Pi 5

**RPi 5 setup — comandos a correr en la RPi via SSH o terminal.**

- [ ] **Step 1: Instalar OS**

- Usar Raspberry Pi Imager → Raspberry Pi OS (64-bit, Bookworm)
- Flashear SD card 32GB
- En Imager: configurar hostname, SSH, WiFi antes de flashear

- [ ] **Step 2: Conectar y actualizar**

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv portaudio19-dev git
```

- [ ] **Step 3: Clonar/copiar proyecto a RPi**

En Windows:
```bash
# Copiar carpeta rpi/ a RPi via SCP
scp -r "D:\Codigo Abierto\Laringofono\rpi" pi@raspberrypi.local:~/laringofono/
```

- [ ] **Step 4: Crear virtualenv e instalar dependencias**

En RPi:
```bash
cd ~/laringofono
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nota: `tflite-runtime` para RPi ARM64:
```bash
pip install tflite-runtime
```
Si falla, usar wheel específico:
```bash
pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp311-cp311-linux_aarch64.whl
```

- [ ] **Step 5: Instalar Piper TTS**

```bash
cd ~
wget https://github.com/rhasspy/piper/releases/latest/download/piper_linux_aarch64.tar.gz
tar -xf piper_linux_aarch64.tar.gz
sudo mv piper /usr/local/bin/piper
# Descargar modelo de voz
mkdir -p ~/piper/models
cd ~/piper/models
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

- [ ] **Step 6: Probar Piper TTS**

```bash
echo "hello world" | piper --model ~/piper/models/en_US-lessac-medium.onnx --output_file /tmp/test.wav
aplay /tmp/test.wav
```
Esperado: escuchar "hello world" por el audio jack + PAM8403 + bocina.

---

## Task 6: serial_receiver.py con Tests

**Files:**
- Create: `rpi/serial_receiver.py`
- Create: `rpi/tests/test_serial_receiver.py`

- [ ] **Step 1: Escribir test que falla**

Crear `rpi/tests/test_serial_receiver.py`:
```python
import numpy as np
import struct
import pytest
from unittest.mock import MagicMock, patch
from serial_receiver import SerialReceiver, FRAME_SAMPLES

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
    raw, checksum, original = make_valid_frame()
    receiver.ser.read.side_effect = [
        bytes([0xAA]), bytes([0x55]), raw, checksum
    ]
    result = receiver.receive_frame()
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert len(result) == FRAME_SAMPLES

def test_bad_checksum_returns_none(receiver):
    raw, _, _ = make_valid_frame()
    bad_checksum = struct.pack('<H', 0x0000)
    receiver.ser.read.side_effect = [
        bytes([0xAA]), bytes([0x55]), raw, bad_checksum
    ]
    assert receiver.receive_frame() is None

def test_short_read_returns_none(receiver):
    receiver.ser.read.side_effect = [
        bytes([0xAA]), bytes([0x55]), b'\x00' * 10, b'\x00\x00'
    ]
    assert receiver.receive_frame() is None
```

- [ ] **Step 2: Correr test — verificar que falla**

```bash
cd ~/laringofono && source .venv/bin/activate
pytest rpi/tests/test_serial_receiver.py -v
```
Esperado: `ImportError: No module named 'serial_receiver'`

- [ ] **Step 3: Implementar serial_receiver.py**

Crear `rpi/serial_receiver.py`:
```python
import struct
import logging
import serial
import numpy as np

FRAME_SAMPLES   = 1024
BYTES_PER_SAMPLE = 2
FRAME_BYTES     = FRAME_SAMPLES * BYTES_PER_SAMPLE
CHECKSUM_BYTES  = 2

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
        expected  = sum(raw) & 0xFFFF
        received  = struct.unpack('<H', checksum_raw)[0]
        if expected != received:
            logger.warning("Checksum mismatch: expected %04X got %04X", expected, received)
            return None
        return np.frombuffer(raw, dtype=np.uint16).astype(np.float32)

    def close(self):
        self.ser.close()
```

- [ ] **Step 4: Correr tests — verificar que pasan**

```bash
pytest rpi/tests/test_serial_receiver.py -v
```
Esperado:
```
PASSED test_receive_valid_frame_returns_float32_array
PASSED test_bad_checksum_returns_none
PASSED test_short_read_returns_none
```

- [ ] **Step 5: Commit**

```bash
git add rpi/serial_receiver.py rpi/tests/test_serial_receiver.py
git commit -m "feat: serial_receiver reads and validates ESP32 frames"
```

---

## Task 7: preprocessor.py con Tests

**Files:**
- Create: `rpi/preprocessor.py`
- Create: `rpi/tests/test_preprocessor.py`

- [ ] **Step 1: Escribir tests que fallan**

Crear `rpi/tests/test_preprocessor.py`:
```python
import numpy as np
import pytest
from preprocessor import Preprocessor, N_MFCC, ADC_MID

@pytest.fixture
def pp():
    return Preprocessor()

def test_normalize_midpoint_is_zero(pp):
    samples = np.full(1024, ADC_MID, dtype=np.float32)
    result = pp.normalize(samples)
    assert np.allclose(result, 0.0, atol=1e-3)

def test_normalize_range_bounded(pp):
    samples = np.random.uniform(0, 4095, 1024).astype(np.float32)
    result = pp.normalize(samples)
    assert result.min() >= -1.0 and result.max() <= 1.0

def test_normalize_max_value(pp):
    samples = np.full(1024, 4095.0, dtype=np.float32)
    result = pp.normalize(samples)
    assert np.all(result > 0.99)

def test_normalize_min_value(pp):
    samples = np.full(1024, 0.0, dtype=np.float32)
    result = pp.normalize(samples)
    assert np.all(result < -0.99)

def test_extract_mfcc_shape(pp):
    samples = np.random.uniform(-1, 1, 1024).astype(np.float32)
    mfccs = pp.extract_mfcc(samples)
    assert mfccs.ndim == 2
    assert mfccs.shape[1] == N_MFCC

def test_process_pipeline(pp):
    raw = np.random.uniform(0, 4095, 1024).astype(np.float32)
    result = pp.process(raw)
    assert result.ndim == 2
    assert result.shape[1] == N_MFCC
    assert result.dtype == np.float32
```

- [ ] **Step 2: Correr tests — verificar que fallan**

```bash
pytest rpi/tests/test_preprocessor.py -v
```
Esperado: `ImportError: No module named 'preprocessor'`

- [ ] **Step 3: Implementar preprocessor.py**

Crear `rpi/preprocessor.py`:
```python
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
```

- [ ] **Step 4: Correr tests — verificar que pasan**

```bash
pytest rpi/tests/test_preprocessor.py -v
```
Esperado: 6 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add rpi/preprocessor.py rpi/tests/test_preprocessor.py
git commit -m "feat: preprocessor normalizes ADC samples and extracts MFCCs"
```

---

## Task 8: Verificar Pipeline ESP32 → RPi (datos reales)

**Integration test — ESP32 conectado a RPi via USB.**

- [ ] **Step 1: Conectar ESP32 a RPi vía USB**

- Cable USB del ESP32 → puerto USB de RPi 5
- En RPi: `ls /dev/ttyUSB*` o `ls /dev/ttyACM*`
- Debe aparecer `/dev/ttyUSB0` o `/dev/ttyACM0`

- [ ] **Step 2: Escribir script de verificación**

Crear `rpi/verify_pipeline.py`:
```python
#!/usr/bin/env python3
import sys
from serial_receiver import SerialReceiver
from preprocessor import Preprocessor

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'

receiver = SerialReceiver(PORT)
preprocessor = Preprocessor()

print(f"Listening on {PORT}... Press Ctrl+C to stop.")
print("Vibrate the laryngophone against throat to generate signal.")

for i in range(5):
    frame = receiver.receive_frame()
    if frame is None:
        print(f"Frame {i+1}: FAILED (None)")
        continue
    mfccs = preprocessor.process(frame)
    print(f"Frame {i+1}: OK — raw min={frame.min():.0f} max={frame.max():.0f} "
          f"| MFCC shape={mfccs.shape} mean={mfccs.mean():.3f}")

receiver.close()
print("Done.")
```

- [ ] **Step 3: Correr verificación**

```bash
cd ~/laringofono && source .venv/bin/activate
python rpi/verify_pipeline.py /dev/ttyUSB0
```
Esperado (con laringófono activo):
```
Frame 1: OK — raw min=1200 max=2900 | MFCC shape=(12, 26) mean=0.234
Frame 2: OK — raw min=... ...
```
- Si `raw min=2047 max=2048`: señal plana → revisar circuito (ver Task 3)
- Si frames FAILED: revisar baudrate o puerto

---

## Task 9: Recolección de Dataset — Edge Impulse

**Data collection — requiere cuenta en edge-impulse.com.**

- [ ] **Step 1: Crear cuenta y proyecto**

- Ir a https://studio.edgeimpulse.com
- New Project → nombre: "laringofono-phonemes"
- Target device: Raspberry Pi 4/5 (o Custom board)

- [ ] **Step 2: Instalar Edge Impulse CLI en RPi**

```bash
curl -sL https://deb.nodesource.com/setup_18.x | sudo bash -
sudo apt install -y nodejs
npm install -g edge-impulse-cli
```

- [ ] **Step 3: Conectar ESP32 como data forwarder**

El ESP32 ya envía datos por serial a 921600 baud.
Edge Impulse data forwarder lee de serial y sube a la nube:
```bash
edge-impulse-data-forwarder --port /dev/ttyUSB0 --baud-rate 921600 --frequency 8000
```
- Ingresar credenciales Edge Impulse cuando pida
- Seleccionar proyecto "laringofono-phonemes"
- Frequency: 8000 (Hz)
- Sensor name: "laryngophone"

- [ ] **Step 4: Grabar muestras por fonema**

En Edge Impulse Studio → Data Acquisition:
- Label: nombre del fonema (ej. "p", "b", "iy", "silence")
- Sample length: 2000ms
- Grabar mínimo 50 muestras por fonema por persona
- Repetir con mínimo 5 personas diferentes
- Prioridad inicial: empezar con 10 fonemas más distinguibles:
  `silence, iy, ae, aa, p, b, s, z, m, n`
- Escalar a 44 fonemas después de validar pipeline

- [ ] **Step 5: Verificar balance de clases**

En Edge Impulse → Dashboard → Data summary:
- Cada clase debe tener ≥50 muestras
- Split automático 80/20 train/test

---

## Task 10: Entrenamiento del Modelo

**Training — en Edge Impulse Studio (web).**

- [ ] **Step 1: Configurar Impulse Design**

En Edge Impulse Studio → Impulse Design:
- Input block: Time series (window 2000ms, stride 200ms, frequency 8000Hz)
- Processing block: MFCCs
  - n_mfcc: 26
  - frame_length: 0.025 (25ms)
  - frame_stride: 0.010 (10ms)
  - num_filters: 32
  - FFT length: 512
- Learning block: Classification (Keras)

- [ ] **Step 2: Configurar y entrenar modelo**

En Edge Impulse → Neural Network:
```
Input layer (MFCCs shape)
→ Conv1D(32, kernel_size=3, activation='relu')
→ BatchNormalization()
→ Conv1D(64, kernel_size=3, activation='relu')
→ GlobalAveragePooling1D()
→ Dense(128, activation='relu')
→ Dropout(0.3)
→ Dense(N_classes, activation='softmax')
```
- Epochs: 50
- Learning rate: 0.001
- Batch size: 32
- Click "Start Training"

- [ ] **Step 3: Evaluar modelo**

En Edge Impulse → Model Testing:
- Run all tests
- Objetivo: accuracy ≥ 70% (con 10 fonemas iniciales)
- Si <60%: aumentar dataset (más muestras o más personas)

- [ ] **Step 4: Exportar modelo TFLite**

En Edge Impulse → Deployment:
- Seleccionar: TensorFlow Lite (float32)
- Click "Build"
- Descargar `.tflite`

- [ ] **Step 5: Copiar modelo a RPi**

```bash
# En Windows:
scp phoneme_classifier.tflite pi@raspberrypi.local:~/laringofono/model/
```

---

## Task 11: inferencer.py con Tests

**Files:**
- Create: `rpi/inferencer.py`
- Create: `rpi/tests/test_inferencer.py`

- [ ] **Step 1: Escribir tests que fallan**

Crear `rpi/tests/test_inferencer.py`:
```python
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

def test_predict_returns_phoneme_string_and_float(inferencer):
    inferencer.interpreter.get_tensor.return_value = make_output(0)
    phoneme, conf = inferencer.predict(np.random.rand(12, 26).astype(np.float32))
    assert isinstance(phoneme, str)
    assert phoneme in PHONEMES
    assert isinstance(conf, float)

def test_predict_returns_argmax_class(inferencer):
    inferencer.interpreter.get_tensor.return_value = make_output(5, 0.95)
    phoneme, conf = inferencer.predict(np.random.rand(12, 26).astype(np.float32))
    assert phoneme == PHONEMES[5]
    assert abs(conf - 0.95) < 1e-5

def test_predict_confidence_between_0_and_1(inferencer):
    inferencer.interpreter.get_tensor.return_value = make_output(2, 0.7)
    _, conf = inferencer.predict(np.random.rand(12, 26).astype(np.float32))
    assert 0.0 <= conf <= 1.0
```

- [ ] **Step 2: Correr tests — verificar que fallan**

```bash
pytest rpi/tests/test_inferencer.py -v
```
Esperado: `ImportError: No module named 'inferencer'`

- [ ] **Step 3: Implementar inferencer.py**

Crear `rpi/inferencer.py`:
```python
import numpy as np
import tflite_runtime.interpreter as tflite

PHONEMES = [
    'p','b','t','d','k','g','f','v',
    'th_unvoiced','th_voiced','s','z','sh','zh','h','ch','jh',
    'm','n','ng','l','r','w','y',
    'iy','ih','ey','eh','ae','aa','ao','ow','uh','uw',
    'ah','er','ax','ay','aw','oy',
    'silence','unknown','dx','nx',
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
```

- [ ] **Step 4: Correr tests — verificar que pasan**

```bash
pytest rpi/tests/test_inferencer.py -v
```
Esperado: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add rpi/inferencer.py rpi/tests/test_inferencer.py
git commit -m "feat: TFLite phoneme inferencer with 44-class output"
```

---

## Task 12: tts_engine.py con Tests

**Files:**
- Create: `rpi/tts_engine.py`
- Create: `rpi/tests/test_tts_engine.py`

- [ ] **Step 1: Escribir tests que fallan**

Crear `rpi/tests/test_tts_engine.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from tts_engine import TTSEngine

@pytest.fixture
def tts():
    return TTSEngine('/fake/model.onnx')

def test_synthesize_calls_piper_with_text(tts):
    with patch('tts_engine.subprocess.run') as mock_run, \
         patch('tts_engine.tempfile.mktemp', return_value='/tmp/t.wav'):
        mock_run.return_value = MagicMock(returncode=0)
        path = tts.synthesize_to_file('hello')
        assert path == '/tmp/t.wav'
        call_args = mock_run.call_args[0][0]
        assert 'piper' in call_args
        assert '/fake/model.onnx' in call_args

def test_synthesize_raises_on_failure(tts):
    with patch('tts_engine.subprocess.run') as mock_run, \
         patch('tts_engine.tempfile.mktemp', return_value='/tmp/t.wav'):
        mock_run.return_value = MagicMock(returncode=1, stderr=b'err')
        with pytest.raises(RuntimeError, match='Piper failed'):
            tts.synthesize_to_file('hello')

def test_speak_calls_aplay(tts):
    with patch('tts_engine.subprocess.run') as mock_run, \
         patch('tts_engine.tempfile.mktemp', return_value='/tmp/t.wav'), \
         patch('tts_engine.os.path.exists', return_value=True), \
         patch('tts_engine.os.unlink'):
        mock_run.return_value = MagicMock(returncode=0)
        tts.speak('hello')
        calls = [c[0][0] for c in mock_run.call_args_list]
        assert any('aplay' in c for c in calls)
```

- [ ] **Step 2: Correr tests — verificar que fallan**

```bash
pytest rpi/tests/test_tts_engine.py -v
```
Esperado: `ImportError: No module named 'tts_engine'`

- [ ] **Step 3: Implementar tts_engine.py**

Crear `rpi/tts_engine.py`:
```python
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
```

- [ ] **Step 4: Correr tests — verificar que pasan**

```bash
pytest rpi/tests/test_tts_engine.py -v
```
Esperado: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add rpi/tts_engine.py rpi/tests/test_tts_engine.py
git commit -m "feat: Piper TTS engine wrapper with aplay output"
```

---

## Task 13: main.py — Pipeline Completo

**Files:**
- Create: `rpi/main.py`

- [ ] **Step 1: Correr todos los tests antes de integrar**

```bash
pytest rpi/tests/ -v
```
Esperado: todos PASSED. Si alguno falla, resolverlo antes de continuar.

- [ ] **Step 2: Implementar main.py**

Crear `rpi/main.py`:
```python
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

SERIAL_PORT  = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
MODEL_PATH   = sys.argv[2] if len(sys.argv) > 2 else '../model/phoneme_classifier.tflite'
TTS_MODEL    = sys.argv[3] if len(sys.argv) > 3 else '/home/pi/piper/models/en_US-lessac-medium.onnx'
SILENCE      = 'silence'

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

            mfccs             = preprocessor.process(frame)
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
        logger.info("Stopped by user.")
    finally:
        receiver.close()

if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Commit**

```bash
git add rpi/main.py
git commit -m "feat: main pipeline orchestrator — receive→infer→speak"
```

---

## Task 14: Prueba End-to-End

**Integration test — sistema completo.**

- [ ] **Step 1: Verificar todos los componentes conectados**

```
✓ Laringófono Baofeng → circuito acondicionamiento → ESP32 GPIO34
✓ ESP32 → USB → RPi 5
✓ RPi 5 audio jack → PAM8403 → bocina 8Ω
✓ modelo phoneme_classifier.tflite en ~/laringofono/model/
✓ Piper instalado y modelo en ~/piper/models/
```

- [ ] **Step 2: Lanzar pipeline**

```bash
cd ~/laringofono && source .venv/bin/activate
python rpi/main.py /dev/ttyUSB0 model/phoneme_classifier.tflite ~/piper/models/en_US-lessac-medium.onnx
```

- [ ] **Step 3: Probar con fonemas conocidos**

- Presionar laringófono firmemente contra laringe
- Intentar emitir fonema /iy/ (como "ee")
- Esperar log: `Detected: iy (0.85)`
- Hacer pausa (silence) → escuchar TTS reproducir

- [ ] **Step 4: Medir latencia**

Agregar timing al log en main.py:
```python
import time
# En el loop, antes de receive_frame:
t0 = time.time()
# Después de tts.speak():
logger.info("Latency: %.0fms", (time.time() - t0) * 1000)
```
Objetivo: <500ms. Si >500ms: reducir FRAME_SIZE a 512 en config.h y en preprocessor.py.

- [ ] **Step 5: Commit final**

```bash
git add -A
git commit -m "feat: end-to-end vocal assistant prototype working"
```

---

## Resumen de Riesgos y Mitigaciones

| Síntoma | Diagnóstico | Fix |
|---|---|---|
| Serial Plotter plano en ~2047 | Laringófono no conectado o C1 invertido | Verificar polaridad C1 y cable K-Type |
| Serial Plotter saturado en 4095 | Ganancia excesiva | Cambiar Rf a 100kΩ |
| frames None constantemente | Puerto serial incorrecto | Verificar `ls /dev/ttyUSB*` |
| Accuracy modelo <60% | Dataset insuficiente | Más grabaciones, más personas |
| Latencia >500ms | Frame demasiado grande | Reducir FRAME_SIZE a 512 |
| Piper no encontrado | PATH incorrecto | `sudo mv piper /usr/local/bin/piper` |
