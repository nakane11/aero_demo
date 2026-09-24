# atoms3_estop

AtomS3R のボタンで `scripts/ros/estop_node.py` へ UDP で STOP/RESUME を送る
ファームウェア (PlatformIO プロジェクト)。ボタンを押すたびに STOP <->
RESUME をトグルし、画面全体の色を状態に応じて変える (STOP=赤,
RESUME=緑, WiFi接続待ち=黄)。

## セットアップ

```
cd firmware/atoms3_estop
cp include/wifi_secrets.h.example include/wifi_secrets.h
```

`include/wifi_secrets.h` を編集し、WiFi の SSID/パスワードと
`estop_node.py` を動かす PC の IP アドレスを設定する
(`wifi_secrets.h` は `.gitignore` 対象なのでコミットされない)。

```c
#define WIFI_SSID "your-wifi-ssid"
#define WIFI_PASSWORD "your-wifi-password"
#define ESTOP_NODE_HOST "192.168.xxx.xxx"
```

ポート番号 (既定 5555) は `estop_node.py` 側の `--listen-port` と揃える
必要がある。変更する場合は `src/main.cpp` の `ESTOP_PORT` を書き換える。

## 書き込み

```
pio run -t upload
pio device monitor -b 115200   # ログ確認 (任意)
```

`board = m5stack-atoms3r` が手元の `platformio-espressif32` のバージョンに
存在しない場合は `pio boards | grep -i atoms3` で正しいボード名を確認し、
`platformio.ini` を書き換えること (見つからない場合、ボタン/画面のピン
配置が共通の `m5stack-atoms3` でも動く見込みだが未確認)。

## 動作確認

1. PC 側で `rosrun aero_demo estop_node.py` を起動しておく。
2. AtomS3 の電源を入れる (WiFi 接続待ちは画面が黄色、接続後は緑 = 通常動作)。
3. ボタンを押す → 画面が赤になり、`estop_node.py` のログに STOP 受信が
   出て `/estop` が True になることを確認する。
4. もう一度ボタンを押す → 画面が緑に戻り、RESUME が送られて `/estop` が
   False に戻ることを確認する。
