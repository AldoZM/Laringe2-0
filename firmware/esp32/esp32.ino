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

  // Timer 0, prescaler 80 -> 1MHz tick, alarma 125 -> 8kHz
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
