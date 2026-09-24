// AtomS3 非常停止ボタン ファームウェア (PlatformIO)
//
// ボタン (G41, M5Unified では M5.BtnA) を押すたびに STOP <-> RESUME の
// 状態をトグルし、対応するコマンド (改行区切りのテキスト) を PC 上の
// estop_node.py (scripts/ros/estop_node.py) へ UDP で送る。
// あわせて画面全体の色を状態に応じて変える
// (STOP = 赤、RESUME = 緑、WiFi接続待ち = 黄)。
//
// UDP は到達保証がないので、状態が変わるたびに同じパケットを複数回連続
// 送信する (estop_node.py 側は 1 回受信すれば cancel を送るが、パケット
// 自体が 1 つも届かないと意味が無いため)。
//
// ビルド/書き込み: platformio.ini 参照 (`pio run -t upload`)。
// WiFi の SSID/パスワードと PC 側 (estop_node.py) の IP/ポートは
// include/wifi_secrets.h (git 管理外、wifi_secrets.h.example を参照) で
// 設定する。

#include <Arduino.h>
#include <M5Unified.h>
#include <WiFi.h>
#include <WiFiUdp.h>

#include "wifi_secrets.h"

// estop_node.py 側の既定ポート (scripts/ros/estop_node.py の
// --listen-port 既定値) と合わせる。
#ifndef ESTOP_PORT
#define ESTOP_PORT 5555
#endif

// UDP パケットの取りこぼし対策で、状態が変わるたびに同じコマンドを
// この回数だけ連続送信する。
static const int SEND_REPEAT_COUNT = 5;
static const int SEND_REPEAT_INTERVAL_MS = 30;

static const uint32_t COLOR_STOPPED = 0xFF0000;      // 赤
static const uint32_t COLOR_RUNNING = 0x00FF00;      // 緑
static const uint32_t COLOR_CONNECTING = 0xFFFF00;   // 黄 (WiFi接続待ち)

void fillScreen(uint32_t color);
void connectWiFi();
void sendCommand(const char *command);
void applyState();

WiFiUDP udp;
bool stopped = false;  // 起動直後は RESUME (通常動作) 扱い

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
    // 接続待ち中でもボタン長押しでリセットできるように、ここでは
    // ボタン監視はしない (単純化のため)。
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
    // 再接続後は現在の状態を再送しておく (再接続中に取りこぼした場合の保険)。
    applyState();
  }

  if (M5.BtnA.wasPressed()) {
    stopped = !stopped;
    applyState();
  }

  delay(10);
}
