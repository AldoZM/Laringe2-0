# Laringófono — Dispositivo Asistencia Vocal
**Fecha:** 2026-05-24  
**Estado:** Aprobado  
**Autor:** Aldo ZM + Claude

---

## 1. Problema

Personas sin voz no pueden comunicarse oralmente. Laringófono Baofeng K-Type capta vibraciones residuales del cuello — señal débil, ruidosa, inutilizable sin procesamiento. Objetivo: convertir esas vibraciones en voz sintetizada inteligible en tiempo real.

---

## 2. Alcance del Prototipo

- Reconoce ~44 fonemas del inglés
- Modelo generalizado (múltiples usuarios, no personalizado)
- Salida: bocina integrada
- Fase futura (fuera de este spec): Bluetooth a celular

---

## 3. Hardware

### 3.1 Inventario (ya disponible)
| Componente | Uso |
|---|---|
| Laringófono Baofeng K-Type | Sensor entrada |
| ESP32 DevKit | Frontend analógico / ADC |
| Raspberry Pi 5 | Backend ML + TTS |
| Arduino Nano | Reserva |

### 3.2 Compras pendientes (UNITELECTRONICS)
| Componente | Valor | Costo aprox. |
|---|---|---|
| LM358 op-amp DIP-8 | — | $12 MXN |
| Resistencias | 10kΩ ×3, 390kΩ ×1, 1kΩ ×1 | $8 MXN |
| Capacitor electrolítico | 10µF | $3 MXN |
| Capacitor cerámico | 47nF | $2 MXN |
| PAM8403 (amp audio clase D) | — | $30 MXN |
| Bocina | 8Ω 0.5W | $25 MXN |
| SD card | 32GB | $120 MXN |
| Cables jumper + breadboard | — | $80 MXN |
| **Total** | | **~$280 MXN** |

---

## 4. Arquitectura del Sistema

```
[Baofeng K-Type]
      │ señal AC ~5-50mV
      ▼
[Circuito Acondicionamiento]
  C1 → Bias (R1/R2) → LM358 40x → RC filter
      │ señal limpia 0–3.3V
      ▼
[ESP32]
  ADC 12-bit @ 8kHz → buffer 1024 muestras → USB Serial
      │ frames PCM raw
      ▼
[Raspberry Pi 5]
  Python → MFCC → TFLite inferencia → Piper TTS → PyAudio
      │ audio WAV
      ▼
[PAM8403 → Bocina 8Ω]
```

---

## 5. Circuito Acondicionamiento de Señal

### 5.1 Propósito
Señal del laringófono: AC, ~5-50mV, oscila positivo/negativo.  
ESP32 ADC acepta solo 0–3.3V. Sin circuito → daño permanente al ADC.

### 5.2 Etapas
| Etapa | Componentes | Función |
|---|---|---|
| Acoplamiento AC | C1: 10µF | Bloquea voltaje negativo |
| Bias DC | R1+R2: 10kΩ c/u | Centra señal en 1.65V (VCC/2) |
| Amplificación | LM358, Rf: 390kΩ, Rin: 10kΩ | Ganancia 40x |
| Anti-aliasing | Rout: 1kΩ, Cout: 47nF | Corta frecuencias >3.4kHz |

### 5.3 Parámetros
- Ganancia: `1 + (390k / 10k) = 40x`
- Frecuencia de corte filtro: `1 / (2π × 1kΩ × 47nF) ≈ 3.4kHz`
- Voltaje de salida: 0 – 3.3V centrado en ~1.65V

### 5.4 Conexiones breadboard
```
1.  Baofeng MIC+  → C1 (10µF)       → nodo A
2.  Nodo A        → R1 (10kΩ)       → 3.3V (ESP32)
3.  Nodo A        → R2 (10kΩ)       → GND
4.  Nodo A        → LM358 pin 3 (+)
5.  LM358 pin 8   → 3.3V
6.  LM358 pin 4   → GND
7.  LM358 pin 1   → Rf (390kΩ)      → LM358 pin 2 (-)
8.  LM358 pin 2   → Rin (10kΩ)      → GND
9.  LM358 pin 1   → Rout (1kΩ)      → GPIO34 ESP32
10. GPIO34        → Cout (47nF)      → GND
11. Baofeng GND   → GND (común)
```

---

## 6. Bloque ESP32

### 6.1 Responsabilidades
- Muestrear GPIO34 (ADC) a 8kHz, resolución 12-bit
- Mantener buffer circular de 1024 muestras (~128ms)
- Al completar frame: enviar por USB Serial CDC a RPi 5
- Protocolo frame: `[HEADER 2B][1024 bytes PCM raw][CHECKSUM 2B]`

### 6.2 Parámetros ADC
| Parámetro | Valor |
|---|---|
| Pin | GPIO34 (ADC1_CH6, input-only) |
| Sample rate | 8,000 Hz |
| Resolución | 12-bit (0–4095) |
| Frame size | 1,024 muestras |
| Duración frame | ~128ms |
| Baudrate USB Serial | 921,600 bps |

---

## 7. Bloque Raspberry Pi 5

### 7.1 Stack de software
| Capa | Herramienta |
|---|---|
| Recepción serial | `pyserial` |
| Arrays numéricos | `numpy` |
| Extracción MFCC | `librosa` o `python_speech_features` |
| Inferencia ML | `tflite-runtime` |
| TTS | `piper-tts` (neural) |
| Audio output | `pyaudio` / `aplay` |

### 7.2 Pipeline de inferencia
```
frame bytes (USB Serial)
  → numpy float32, normalizar [-1, 1]
  → MFCC: 26 coeficientes, ventana 25ms, hop 10ms
  → tensor shape: [1, tiempo, 26]
  → TFLite interpreter.invoke()
  → argmax sobre 44 clases → índice fonema
  → buffer fonemas → formar palabra
  → Piper TTS.synthesize(texto)
  → PyAudio.write(wav_bytes)
```

### 7.3 Latencia objetivo
| Etapa | Objetivo |
|---|---|
| Captura frame (ESP32) | 128ms |
| MFCC + inferencia (RPi 5) | <100ms |
| TTS síntesis | <200ms |
| **Total end-to-end** | **<500ms** |

---

## 8. Modelo ML

### 8.1 Clases objetivo
44 fonemas del inglés americano (IPA):
Consonantes: /p b t d k g f v θ ð s z ʃ ʒ h tʃ dʒ m n ŋ l r w j/  
Vocales: /iː ɪ eɪ ɛ æ ɑː ɔː oʊ ʊ uː ʌ ɜː ə aɪ aʊ ɔɪ/

### 8.2 Arquitectura modelo (candidato)
```
Input: MFCCs [batch, time_steps, 26]
  → Conv1D(32, kernel=3, relu)
  → BatchNorm
  → Conv1D(64, kernel=3, relu)
  → GlobalAveragePooling1D
  → Dense(128, relu)
  → Dropout(0.3)
  → Dense(44, softmax)
Output: probabilidad 44 fonemas
```

### 8.3 Dataset
- Plataforma: Edge Impulse Studio
- Mínimo: 50 muestras × 44 fonemas × 5 personas (mínimo viable para generalización)
- Captura: ESP32 en modo recolección → Edge Impulse data forwarder
- Features: MFCC (mismo pipeline que inferencia para consistencia)
- Split: 80% train / 20% test

### 8.4 Exportación
- Entrenar en Edge Impulse o Colab
- Exportar como `.tflite` (int8 quantized)
- Copiar a RPi 5, cargar con `tflite-runtime`

---

## 9. Audio Output

```
RPi 5 (3.5mm jack) → PAM8403 (amp clase D) → Bocina 8Ω 0.5W
```
- PAM8403: alimentación 5V (desde RPi), ganancia ~26dB, stereo
- Bocina: cualquier 8Ω ≥0.5W disponible en UNITELECTRONICS

---

## 10. Fases de Desarrollo

| Fase | Entregable | Prioridad |
|---|---|---|
| 1 | Circuito acondicionamiento en breadboard funcional | Alta |
| 2 | ESP32 muestrea y envía frames a RPi por serial | Alta |
| 3 | RPi recibe frames y extrae MFCCs | Alta |
| 4 | Recolección dataset (Edge Impulse) | Alta |
| 5 | Entrenamiento + exportación modelo .tflite | Alta |
| 6 | Pipeline inferencia completo en RPi | Alta |
| 7 | Integración TTS (Piper) + bocina | Alta |
| 8 | Prueba end-to-end completa | Alta |
| 9 | BT a celular | Futura |

---

## 11. Riesgos

| Riesgo | Mitigación |
|---|---|
| Señal laringófono demasiado débil | Aumentar Rf para mayor ganancia; probar con osciloscopio |
| Dataset insuficiente para 44 fonemas | Empezar con 10 fonemas, escalar gradualmente |
| Latencia >500ms inaceptable | Reducir frame size, optimizar pipeline MFCC |
| Ruido eléctrico en breadboard | Apantallar cables, usar GND común sólido |
| Piper TTS muy lento en RPi 5 | Usar modelo Piper tiny/fast o fallback a espeak |
