/**
 * AI-Enhanced Atmospheric Electricity Harvesting
 * Arduino/ESP32 MQTT Publisher
 * Matches AI model input parameters
 * 
 * Components in diagram:
 *  - Voltage Sensor (e.g., INA219 or similar)
 *  - Current Sensor (e.g., INA219 or ACS712)
 *  - DHT22 (Temperature & Humidity)
 *  - Electric Field Sensor (Analog input)
 *  - Battery Level Sensor (Voltage divider)
 *  - Sensor Active Flag (Digital Input or set to 1 if always on)
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <Adafruit_INA219.h>
#include "DHT.h"

// ------------------- CONFIG -------------------
#define WIFI_SSID "YOUR_WIFI_SSID"
#define WIFI_PASS "YOUR_WIFI_PASSWORD"
#define MQTT_SERVER "broker.hivemq.com"
#define MQTT_PORT 1883
#define MQTT_TOPIC "aiharvest/iot"

// Sensor pins
#define DHTPIN 4
#define DHTTYPE DHT22
#define FIELD_SENSOR_PIN 34     // Analog input for field strength
#define BATTERY_SENSOR_PIN 35   // Analog input for battery level
#define SENSOR_ACTIVE_PIN 5     // Digital input or fixed

// Battery reference
#define BATTERY_MAX_VOLTAGE 4.2
#define BATTERY_MIN_VOLTAGE 3.0

// ------------------- OBJECTS -------------------
WiFiClient espClient;
PubSubClient client(espClient);
Adafruit_INA219 ina219; 
DHT dht(DHTPIN, DHTTYPE);

// ------------------- HELPERS -------------------
void connectWiFi() {
  Serial.print("Connecting to WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Connected!");
}

void connectMQTT() {
  while (!client.connected()) {
    Serial.print("Connecting to MQTT...");
    if (client.connect("ESP32Client_" + String(random(0xffff), HEX))) {
      Serial.println("Connected!");
    } else {
      Serial.print("Failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying...");
      delay(2000);
    }
  }
}

// Map battery voltage to %
float batteryPercentage(float v) {
  float pct = (v - BATTERY_MIN_VOLTAGE) / (BATTERY_MAX_VOLTAGE - BATTERY_MIN_VOLTAGE) * 100.0;
  return constrain(pct, 0.0, 100.0);
}

// ------------------- SETUP -------------------
void setup() {
  Serial.begin(115200);
  delay(2000);

  connectWiFi();
  client.setServer(MQTT_SERVER, MQTT_PORT);

  if (!ina219.begin()) {
    Serial.println("Failed to find INA219 chip");
    while (1) { delay(10); }
  }

  dht.begin();
  pinMode(SENSOR_ACTIVE_PIN, INPUT_PULLUP); // Or OUTPUT if manual
}

// ------------------- LOOP -------------------
void loop() {
  if (!client.connected()) {
    connectMQTT();
  }
  client.loop();

  // Voltage & current from INA219
  float busVoltage = ina219.getBusVoltage_V();  // volts
  float current_mA = ina219.getCurrent_mA();    // mA
  float current_A = current_mA / 1000.0;
  float power_W = busVoltage * current_A;
  float energy_J = power_W * 60;  // Approx per minute

  // DHT22 readings
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  // Electric field sensor (scale based on calibration)
  int field_raw = analogRead(FIELD_SENSOR_PIN);
  float field_strength = map(field_raw, 0, 4095, 0, 500); // Example: 0–500 V/m

  // Battery level
  int battery_raw = analogRead(BATTERY_SENSOR_PIN);
  float battery_voltage = ((float)battery_raw / 4095.0) * 2.0 * 3.3; // Voltage divider x2
  float storage_level = batteryPercentage(battery_voltage);

  // Sensor active flag
  int sensor_active = digitalRead(SENSOR_ACTIVE_PIN) == HIGH ? 1 : 0;

  // Build JSON payload
  String payload = "{";
  payload += "\"timestamp\":\"" + String(getISO8601Time().c_str()) + "\",";
  payload += "\"voltage_V\":" + String(busVoltage, 3) + ",";
  payload += "\"current_A\":" + String(current_A, 3) + ",";
  payload += "\"power_W\":" + String(power_W, 3) + ",";
  payload += "\"energy_usage_J\":" + String(energy_J, 3) + ",";
  payload += "\"temperature_C\":" + String(temperature, 1) + ",";
  payload += "\"humidity_%\":" + String(humidity, 1) + ",";
  payload += "\"field_strength_V_m\":" + String(field_strength, 1) + ",";
  payload += "\"storage_level_%\":" + String(storage_level, 1) + ",";
  payload += "\"sensor_active\":" + String(sensor_active);
  payload += "}";

  // Publish
  client.publish(MQTT_TOPIC, payload.c_str());
  Serial.println("Published: " + payload);

  delay(5000); // every 5 seconds
}

// ------------------- TIME -------------------
// Dummy ISO8601 — replace with RTC or NTP sync for accurate timestamps
String getISO8601Time() {
  unsigned long nowMillis = millis() / 1000;
  unsigned long sec = nowMillis % 60;
  unsigned long min = (nowMillis / 60) % 60;
  unsigned long hour = (nowMillis / 3600) % 24;
  char buf[25];
  sprintf(buf, "2025-08-15T%02lu:%02lu:%02luZ", hour, min, sec);
  return String(buf);
}
