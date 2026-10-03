// AtomS3 非常停止ボタン ファームウェア (PlatformIO)
//
// ボタンを押すたびに STOP <-> RESUME をトグルし、estop_node.py へ UDP で送る
// (画面: STOP=赤、RESUME=緑、WiFi 接続待ち=黄)。UDP は到達保証が無いので
// 同じコマンドを複数回送る。
// 書き込み: `pio run -t upload`。WiFi と送信先は include/wifi_secrets.h
// (git 管理外、wifi_secrets.h.example 参照) で設定する。

#include <Arduino.h>
#include <M5Unified.h>
#include <WiFi.h>
#include <WiFiUdp.h>

#include "wifi_secrets.h"

// estop_node.py の --listen-port 既定値と合わせる。
#ifndef ESTOP_PORT
#define ESTOP_PORT 5555
#endif

// 取りこぼし対策の連続送信回数。
static const int SEND_REPEAT_COUNT = 5;
static const int SEND_REPEAT_INTERVAL_MS = 30;

static const uint32_t COLOR_STOPPED = 0xFF0000;      // 赤
static const uint32_t COLOR_RUNNING = 0x00FF00;      // 緑
static const uint32_t COLOR_CONNECTING = 0xFFFF00;   // 黄 (WiFi接続待ち)

// チャタリング対策: 前回のトグルからこの時間内の再トグルは無視する
// (M5Unified の debounce だけでは 1 回の押下で複数回発火することがある)。
static const uint32_t DEBOUNCE_MS = 300;

void fillScreen(uint32_t color);
void connectWiFi();
void sendCommand(const char *command);
void applyState();

WiFiUDP udp;
bool stopped = false;  // 起動直後は RESUME (通常動作) 扱い
uint32_t lastToggleMs = 0;

void fillScreen(uint32_t color) {
  M5.Display.fillScreen(color);
}

void connectWiFi() {
  fillScreen(COLOR_CONNECTING);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("[atoms3_estop] Connecting to WiFi SSID=%s ...\n", WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(200);
    M5.update();
    // 接続待ち中はボタンを監視しない。
  }
  Serial.printf("[atoms3_estop] WiFi connected. IP=%s\n",
                WiFi.localIP().toString().c_str());
}

void sendCommand(const char *command) {
  for (int i = 0; i < SEND_REPEAT_COUNT; i++) {
    udp.beginPacket(ESTOP_NODE_HOST, ESTOP_PORT);
    udp.write(reinterpret_cast<const uint8_t *>(command), strlen(command));
    udp.write('\n');
    udp.endPacket();
    delay(SEND_REPEAT_INTERVAL_MS);
  }
  Serial.printf("[atoms3_estop] sent \"%s\" x%d to %s:%d\n",
                command, SEND_REPEAT_COUNT, ESTOP_NODE_HOST, ESTOP_PORT);
}

void applyState() {
  if (stopped) {
    sendCommand("STOP");
    fillScreen(COLOR_STOPPED);
  } else {
    sendCommand("RESUME");
    fillScreen(COLOR_RUNNING);
  }
}

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  M5.Display.setRotation(0);

  Serial.begin(115200);

  connectWiFi();

  // 起動直後は RESUME (通常動作) 状態で開始する。
  stopped = false;
  fillScreen(COLOR_RUNNING);
}

void loop() {
  M5.update();

  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
    // 再接続後は現在の状態を再送する。
    applyState();
  }

  if (M5.BtnA.wasPressed()) {
    uint32_t now = millis();
    if (now - lastToggleMs >= DEBOUNCE_MS) {
      lastToggleMs = now;
      stopped = !stopped;
      applyState();
    }
  }

  delay(10);
}
