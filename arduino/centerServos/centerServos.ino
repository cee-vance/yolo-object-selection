#include <Servo.h>

Servo xServo;  // Horizontal axis
Servo yServo;  // Vertical axis

void setup() {
  xServo.attach(9);   // Connect horizontal servo to pin 9
  yServo.attach(10);  // Connect vertical servo to pin 10

  xServo.write(90);   // Center horizontal servo
  yServo.write(90);   // Center vertical servo
}

void loop() {
  // Nothing needed here for static centering
}
