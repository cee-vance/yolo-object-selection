#include <Servo.h>

Servo xServo;  // Horizontal servo
Servo yServo;  // Vertical servo

const int laserPin = 5;  // Laser pointer control
String inputString = "";
bool inputComplete = false;

int currentX = 90;
int currentY = 90;

void setup() {
  Serial.begin(9600);
  xServo.attach(9);
  yServo.attach(10);
  pinMode(laserPin, OUTPUT);
  digitalWrite(laserPin, LOW);

  inputString.reserve(20);

  // Center servos
  xServo.write(currentX);
  yServo.write(currentY);
  
  Serial.println("System ready. Waiting for target coordinates...");
}

void loop() {
  if (inputComplete) {
    Serial.print("Received command: ");
    Serial.println(inputString);

    int commaIndex = inputString.indexOf(',');
    if (commaIndex > 0) {
      int targetX = inputString.substring(0, commaIndex).toInt();
      int targetY = inputString.substring(commaIndex + 1).toInt();

      Serial.print("Target angles → X: ");
      Serial.print(targetX);
      Serial.print(" | Y: ");
      Serial.println(targetY);

      // Smooth sweep
  

      currentX = targetX;
      currentY = targetY;

      sweepServos(currentX, targetX, currentY, targetY);

      // Laser pulse
      Serial.println("Laser ON");
      digitalWrite(laserPin, HIGH);
      delay(5000);
      digitalWrite(laserPin, LOW);
      Serial.println("Laser OFF");
    } else {
      Serial.println("⚠️ Invalid format. Expected 'x,y'");
    }

    inputString = "";
    inputComplete = false;
  }
}

void sweepServos(int fromX, int toX, int fromY, int toY) {
  int stepX = (toX > fromX) ? 1 : -1;
  int stepY = (toY > fromY) ? 1 : -1;

  int posX = fromX;
  int posY = fromY;

  while (posX != toX || posY != toY) {
    if (posX != toX) posX += stepX;
    if (posY != toY) posY += stepY;

    xServo.write(posX);
    yServo.write(posY);
    delay(5);  // Adjust for smoothness
  }

  // Final position
  xServo.write(toX);
  yServo.write(toY);
}


void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      inputComplete = true;
    } else {
      inputString += inChar;
    }
  }
}
