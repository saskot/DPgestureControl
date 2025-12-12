#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>

#define LED_PIN 2

// NUS UUIDs
#define SERVICE_UUID "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define RX_CHAR_UUID "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  // WRITE / WRITE_NR
#define TX_CHAR_UUID "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  // NOTIFY

BLEServer* pServer = nullptr;
BLECharacteristic* pRxChar = nullptr;
BLECharacteristic* pTxChar = nullptr;
bool deviceConnected = false;

// ----- Stavový automat pre LED -----
enum LedMode { MODE_OFF, MODE_ON, MODE_BLINK_SLOW, MODE_BLINK_FAST };
volatile LedMode currentMode = MODE_OFF;

unsigned long lastToggleMs = 0;
bool ledState = false;  // aktuálny stav pinu (HIGH/LOW)

// intervaly blikania (uprav podľa chuti)
const unsigned long SLOW_INTERVAL_MS = 500;
const unsigned long FAST_INTERVAL_MS = 100;

// pomocná – pošli text cez notify (ak je central pripojený)
void notifyText(const String& s) {
  if (deviceConnected && pTxChar) {
    pTxChar->setValue((uint8_t*)s.c_str(), s.length());
    pTxChar->notify();
  }
}

String modeToString(LedMode m) {
  switch (m) {
    case MODE_OFF:        return "off";
    case MODE_ON:         return "on";
    case MODE_BLINK_SLOW: return "blinkslow";
    case MODE_BLINK_FAST: return "blinkfast";
  }
  return "unknown";
}

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* s) override {
    deviceConnected = true;
    Serial.println("✅ Central pripojený");
  }
  void onDisconnect(BLEServer* s) override {
    deviceConnected = false;
    Serial.println("🔌 Central odpojený, reklamujem...");
    s->getAdvertising()->start();
  }
};

class RxCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic* pChar) override {
    String rx = pChar->getValue();
    rx.trim();
    rx.toLowerCase();
    if (rx.length() == 0) return;

    Serial.print("Prijaté: ");
    Serial.println(rx);

    if (rx == "on") {
      currentMode = MODE_ON;
      digitalWrite(LED_PIN, HIGH);
      ledState = true;
      notifyText("OK on");
    } else if (rx == "off") {
      currentMode = MODE_OFF;
      digitalWrite(LED_PIN, LOW);
      ledState = false;
      notifyText("OK off");
    } else if (rx == "blinkslow") {
      currentMode = MODE_BLINK_SLOW;
      // reset časovača, aby sa okamžite prejavilo
      lastToggleMs = millis();
      notifyText("OK blinkslow");
    } else if (rx == "blinkfast") {
      currentMode = MODE_BLINK_FAST;
      lastToggleMs = millis();
      notifyText("OK blinkfast");
    } else if (rx == "state?") {
      notifyText("state: " + modeToString(currentMode));
    } else {
      Serial.println("Neznámy príkaz");
      notifyText("ERR unknown");
    }
  }
};

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  BLEDevice::init("ESP32_Gesture");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new ServerCallbacks());

  BLEService* pService = pServer->createService(SERVICE_UUID);

  // RX – príjem príkazov
  pRxChar = pService->createCharacteristic(
    RX_CHAR_UUID,
    BLECharacteristic::PROPERTY_WRITE | BLECharacteristic::PROPERTY_WRITE_NR
  );
  pRxChar->setCallbacks(new RxCallbacks());

  // TX – notifikácie (odpovede, echo)
  pTxChar = pService->createCharacteristic(
    TX_CHAR_UUID,
    BLECharacteristic::PROPERTY_NOTIFY
  );
  pTxChar->addDescriptor(new BLE2902());

  pService->start();
  BLEDevice::startAdvertising();

  Serial.println("BLE NUS beží. Píš na 0002 (on/off/blinkslow/blinkfast/state?).");
}

void loop() {
  // neblokujúce blikanie podľa módu
  unsigned long now = millis();

  switch (currentMode) {
    case MODE_OFF:
      if (ledState) { digitalWrite(LED_PIN, LOW); ledState = false; }
      break;

    case MODE_ON:
      if (!ledState) { digitalWrite(LED_PIN, HIGH); ledState = true; }
      break;

    case MODE_BLINK_SLOW:
      if (now - lastToggleMs >= SLOW_INTERVAL_MS) {
        ledState = !ledState;
        digitalWrite(LED_PIN, ledState ? HIGH : LOW);
        lastToggleMs = now;
      }
      break;

    case MODE_BLINK_FAST:
      if (now - lastToggleMs >= FAST_INTERVAL_MS) {
        ledState = !ledState;
        digitalWrite(LED_PIN, ledState ? HIGH : LOW);
        lastToggleMs = now;
      }
      break;
  }

  // žiadne delay – nech je BLE stabilné
  delay(1);
}
