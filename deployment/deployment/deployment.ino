
#include <iostream>
#include <TFT_eSPI.h> //has a hardware-specific setup file.
#include <SPI.h>
#include "../../data/data.h"

TFT_eSPI tft = TFT_eSPI();

void setup() {
  Serial.begin(9600);
  Serial.println("Begining setup");

  tft.init();

  Serial.println("Screen initialized.");

  tft.fillScreen(TFT_BLACK);
  drawtext("Device initialized.\nBeginning run.", TFT_WHITE);
  Serial.println("Device initialized.\nBeginning run.");
  delay(2000);
}

void loop() {
  triangles();
}

void drawtext(char *text, uint16_t color){
  tft.setCursor(0,0);
  tft.setTextColor(color);
  tft.setTextWrap(true);
  tft.print(text);
}

void triangles() {
  tft.fillScreen(TFT_BLACK);
  int color = 0xF800;
  int t;
  int w = tft.width()/2;
  int x = tft.height()-1;
  int y = 0;
  int z = tft.width();
  for(t = 0 ; t <= 15; t+=1) {
    tft.drawTriangle(w, y, y, x, z, x, color);
    x-=4;
    y+=4;
    z-=4;
    color+=100;
    delay(100);
  }
}
