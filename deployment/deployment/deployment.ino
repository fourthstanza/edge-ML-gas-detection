
#include <iostream>
#include <TFT_eSPI.h> //has a hardware-specific setup file.
#include <SPI.h>
//#include "../../data/data.h"
#include "driver/adc.h"
#include "esp_adc_cal.h"
#include "Free_Fonts.h"


TFT_eSPI tft = TFT_eSPI();

#define ANALOG_IN_PIN  15 // ESP32 pin GPIO15 (ADC1) connected to sensor
#define REF_VOLTAGE    1.3
#define ADC_RESOLUTION 4095.0
#define R1             30000.0 // resistor values in voltage sensor (in ohms)
#define R2             7500.0  // resistor values in voltage sensor (in ohms)
#define PUSHBUTTON     36 // GPIO pin of pushbutton

int xpos = 5;
int ypos = 40;

void setup() {
  Serial.begin(9600);
  Serial.println("Begining setup");

  tft.init();
  tft.fillScreen(TFT_BLACK);
  tft.setFreeFont(FSB9);

  Serial.println("Screen initialized.");

  analogSetAttenuation(ADC_6db); // for the voltage at max resistance of the pot

  pinMode(PUSHBUTTON, INPUT);

  drawtext("Device initialized.\nBeginning run.", TFT_WHITE, 0, ypos);
  Serial.println("Device initialized.\nBeginning run.");
  
  delay(2000);

}

void loop() {
  float voltage = readADC();
  int buttonState = digitalRead(PUSHBUTTON);

  if (buttonState == 1){
    int NUM_SAMPLES = 100;
    float measurement[NUM_SAMPLES];

    for (int i = 0; i < NUM_SAMPLES; i++) {
    measurement[i] = readADC();
    delay(10);
    }

    Serial.print("[");
    for (int i = 0; i < NUM_SAMPLES; i++) {
      Serial.print(measurement[i]);
      Serial.print(", ");
      if ((i+1)%10 == 0){
        Serial.println("");
      }
    }
    Serial.println("]");
  }

  
  tft.fillScreen(TFT_BLACK);
  tft.drawFloat(voltage, 2, xpos, ypos);
  delay(100);
}

float readADC() {
  int adc_value = analogRead(ANALOG_IN_PIN);
  float voltage_adc = ((float)adc_value * REF_VOLTAGE) / ADC_RESOLUTION;
  return(voltage_adc);
}

void drawtext(char *text, uint16_t color, int xp, int yp){
  tft.setCursor(xp,yp);
  tft.setTextColor(color);
  tft.setTextWrap(true);
  tft.print(text);
}
