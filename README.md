# Exness XAU & US Stocks Signal

Bot nhẹ, chỉ đọc dữ liệu từ terminal **Exness MetaTrader 5**. Không có Binance, Twelve Data, WebSocket giá 10/30 giây, AI tạo mức giá, hoặc nguồn market-data dự phòng.

Bot không đặt lệnh. Tín hiệu là công cụ hỗ trợ định lượng, không phải cam kết lợi nhuận. Hãy paper-test trước khi dùng tiền thật.

## Luồng hoạt động

| Tài sản | Khi active | Nhịp quét | Nến được cập nhật |
|---|---|---:|---|
| XAUUSD | Giờ Exness mở kim loại, trừ maintenance/cuối tuần | 1 phút | 1m mỗi phút; 5m/15m/1H/4H đúng lúc đóng |
| US stocks | 09:40–15:45 New York, thứ Hai–thứ Sáu | 5 phút | 1m/5m mỗi 5 phút; 15m/1H/4H đúng lúc đóng |
| D1 | Lần active đầu tiên của ngày giao dịch mới | 1 lần/ngày | Chỉ nến ngày đã đóng |

Ngoài giờ giao dịch, scanner không gọi MT5 và không sinh tín hiệu. Giờ New York dùng `zoneinfo`, nên DST tự đổi. Bot còn chặn quote cũ để tránh phân tích ngày nghỉ lễ hoặc lúc Exness không phát giá.

Exness công bố stock CFD theo giờ Wall Street và XAU có maintenance hằng ngày; lịch cụ thể có thể thay đổi vào ngày lễ. Xem [Exness trading hours](https://www.exness.com/blog/product/exness-trading-hours/). Python đọc tick/nến qua terminal bằng API chính thức của MetaTrader 5: [MetaTrader5 Python integration](https://www.mql5.com/en/docs/python_metatrader5).

## Logic tín hiệu

Bốn cửa, phải qua **tất cả** mới ra lệnh. Thiếu bất kỳ cửa nào là WAIT.

1. **Thiên hướng khung ngày (bộ lọc cứng).** D1 phải rõ hướng — giá trên EMA200 *và* EMA20>EMA50 để cho phép LONG, ngược lại cho SHORT. D1 đi ngang hoặc mâu thuẫn thì **NEUTRAL và không vào lệnh nào cả**. Đây là cửa quan trọng nhất: thiếu nó, bot đã mở 150 lệnh SHORT trong sóng tăng 70% và lỗ 34,75R.
2. **Đồng thuận đa khung.** Ít nhất 2/3 khung 15m–1H–4H cùng hướng vượt `alignment_floor`, và điểm tổng hợp vượt `actionable_score`. RSI(14) quanh mốc 50 + độ dốc là thành phần nặng nhất, xác nhận cùng EMA20/50/200, MACD histogram, ADX, ATR.
3. **Vùng RSI khung vào lệnh.** Tránh mua khi đã quá nóng, bán khi đã quá bán.
4. **Retest có nến đóng.** Breakout–retest cấu trúc, retest EMA20, hoặc retest biên mở phiên Mỹ — kèm nến từ chối.

Khung vào lệnh mặc định là **15m, không phải 5m**: ở 5m spread Exness ăn 3,47% risk mỗi lệnh (SL tối thiểu chỉ ~3,2 USD), ở 15m còn 1,90%. Đo trên 15 tháng tick thật, chênh lệch này một mình đã lật kết quả từ −23,25R thành +23,76R.
**Phiên Mỹ** vẫn chạy đầy đủ bốn cửa trên. Song song có lớp opening-range riêng: không vào lúc tạo biên, ưu tiên retest biên sau mở cửa, tắt entry giữa phiên thanh khoản thấp và 15 phút cuối.

**Rủi ro:** SL đặt ngoài nến/cấu trúc retest cộng buffer ATR; nếu SL cấu trúc quá xa thì bỏ setup thay vì kéo SL vào gần. TP1 mặc định 1.5R, TP2 2.5R, chốt một phần ở TP1 rồi dời SL về hoà vốn.

Điểm `setup /100` là độ đầy đủ điều kiện, **không phải xác suất thắng**.

Thông báo 4H và D1 được gom ngắn gọn. Thông báo Entry chỉ phát khi fingerprint setup thay đổi và đã qua cooldown.

## Thông báo Telegram

- XAUUSD và toàn bộ cổ phiếu được bật trong `assets` đều tự gửi Telegram khi có Entry retest mới.
- Cả hai nhóm đều có bản tin 4H và D1. Nếu nhiều mã làm tin dài quá giới hạn Telegram, bot tự chia thành nhiều tin và không cắt mất mã ở cuối danh sách.
- Cổ phiếu chỉ active trong giờ stock CFD New York; ngoài giờ bot không gọi MT5 và không gửi dự báo giả từ quote cũ.
- Tất cả asset hiện bị khóa `mode: signal_only`. Đặc biệt cổ phiếu **chỉ thông báo**, không có đường code gửi lệnh MT5.

## Khi setup đang mở bị phá vỡ

`position_exit.py` định nghĩa chính sách quản trị để execution engine/backtest dùng chung:

- SL cứng luôn được kiểm tra bằng đúng phía giá đóng lệnh: BUY thoát theo Bid, SHORT thoát theo Ask.
- Không đóng chỉ vì một wick xuyên mức hoặc RSI đơn lẻ. Phá cấu trúc cần **2 nến 5m** hoặc **1 nến 15m** đóng ngoài mức hủy setup, đồng thời RSI/score 5m đã đảo bất lợi.
- Đóng toàn bộ nếu ít nhất 2/3 khung 15m–1H–4H đảo đồng thuận ngược vị thế.
- Time-stop sau 24 nến 5m nếu chưa đi được 0.5R và động lượng đã mất.
- TP1 chốt 50%, phần còn lại dời SL về hòa vốn; TP2 đóng phần còn lại.

Bot Telegram hiện vẫn **chỉ phát tín hiệu, chưa gửi lệnh hay đóng lệnh trên MT5**. Module trên là logic đã kiểm thử để tích hợp vào execution engine sau này; không nên bật auto-trading trước khi có paper/forward test đạt chuẩn.

## Replay Tick History Exness

Replay chỉ nhận ZIP/CSV Bid/Ask từ [Exness Tick History](https://www.exness.com/tick-history/). Tick được đọc theo chunk để không nạp toàn bộ CSV vào RAM, gom thành nến UTC rồi chạy đúng nến đã đóng. BUY vào Ask/thoát Bid; SHORT vào Bid/thoát Ask. Nếu một nến 1m chạm cả SL và TP mà không còn thứ tự tick, replay giả định SL trước.

Ví dụ:

```powershell
python run_exness_backtest.py `
  --symbol XAUUSD --asset-type metal `
  --period 2025 --period 2025-08 --period 2025-09 --period 2025-10 --period 2025-11 --period 2025-12 `
  --bars-cache backtest/cache/XAUUSD_2025_1m.pkl `
  --output logs/exness_backtest_XAUUSD_2025.json
```

`--symbol` phải khớp suffix tài khoản/dataset Exness, ví dụ `XAUUSDm` cho Standard. Luôn đọc dòng `Nguồn ... start → end`; archive theo năm có thể không phủ hết năm, khi đó cần bổ sung archive theo tháng. ZIP và cache nằm trong `.gitignore`, không đưa vào production image.

Kết quả kiểm tra hiện tại, giữ nguyên tham số và có cooldown 60 phút:

| Tập | Baseline | Có đóng sớm | Kết luận |
|---|---:|---:|---|
| XAUUSD 2025 | +10.455R, PF 1.152, MDD 11R, 120 lệnh | +11.293R, PF 1.166, MDD 11R | Dương nhưng biên lợi thế mỏng |
| XAUUSD 01–08/2026 ngoài mẫu | -34.250R, PF 0.810, MDD 46.25R, 276 lệnh | -30.969R, PF 0.824, MDD 41.06R | Không đạt chuẩn auto-entry; đóng sớm chỉ giảm thiệt hại |
| AAPL 2025–08/2026 | +2.986R, 3 lệnh | +2.986R, 3 lệnh | Mẫu quá nhỏ, không có ý nghĩa thống kê |

Các số trên đã tính spread lịch sử Bid/Ask nhưng chưa có commission/slippage thực tế. Vì 2026 thất bại ngoài mẫu, mặc định dự án phải giữ chế độ signal/paper-trading; không được diễn giải kết quả 2025 hay 3 lệnh AAPL thành xác suất thắng.

## Cài đặt trên Windows

MetaTrader5 Python giao tiếp trực tiếp với terminal desktop, vì vậy cần Windows hoặc Windows VPS; Render/Linux container thông thường không dùng được IPC này.

1. Cài Exness MetaTrader 5, đăng nhập tài khoản Exness (nên dùng demo trước), bật các mã cần theo dõi trong Market Watch và để terminal chạy.
2. Cài Python 3.10+ và dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Sao chép `.env.example` thành `.env`, điền `TELEGRAM_BOT_TOKEN` và `TELEGRAM_CHAT_ID`. Nếu máy chỉ có một terminal đang đăng nhập, có thể để trống cấu hình MT5. Nếu có nhiều terminal, đặt `EXNESS_MT5_PATH` tới `terminal64.exe` của Exness.
4. Kiểm tra một lần:

   ```powershell
   python main.py --symbol XAUUSD
   python main.py --symbol AAPL
   ```

5. Chạy Telegram bot:

   ```powershell
   python lightweight_bot.py
   ```

Provider hard-fail nếu terminal đăng nhập broker không phải Exness. Nếu một mã như `FTNT` không có trên loại tài khoản/khu vực Exness hiện tại, bot báo khi startup rồi bỏ qua mã đó; không thay bằng giá ở nơi khác.

## Lệnh Telegram

- `/xau`, `/gia`, `/signal`: phân tích XAUUSD ngay nếu thị trường mở.
- `/cp AAPL`: phân tích một cổ phiếu trong danh sách lớn.
- `/stocks`: bản quét gọn toàn bộ nhóm cổ phiếu đang mở.
- `/status`: trạng thái phiên và cache, không gọi dữ liệu giá.
- `/help`: trợ giúp.

Danh sách mặc định: AAPL, NVDA, MSFT, AMZN, GOOGL, META, TSLA, AVGO, JPM, WMT và FTNT. Có thể bật/tắt trong `config.yaml`.

## Cấu hình quan trọng

- `assets[].scan_minutes`: XAU mặc định 1, stocks mặc định 5.
- `assets[].mode`: hiện chỉ chấp nhận `signal_only`; cấu hình giá trị khác sẽ làm startup thất bại an toàn.
- `scanner.timeframes`: các khung nến cache.
- `alerts.telegram_enabled` / `entry_alerts`: bật toàn bộ Telegram và bật riêng cảnh báo Entry.
- `alerts.four_hour_summary` / `daily_summary`: bật bản tin 4H và D1 cho cả XAU lẫn stocks.
- `strategy.rsi_long_zone` / `rsi_short_zone`: vùng RSI cho retest, tránh đuổi giá.
- `strategy.entry_interval`: khung vào lệnh. **Đừng hạ về 5min** — chi phí spread ở đó nuốt hết edge.
- `strategy.require_bias_alignment` / `bias_interval`: bộ lọc thiên hướng khung lớn. Tắt nó là mở lại cửa cho lệnh ngược sóng.
- `strategy.actionable_score` / `alignment_floor`: độ chặt của đồng thuận đa khung.
- `strategy.minimum_stop_atr` / `maximum_stop_atr`: biên an toàn SL.
- `take_profit_1_r` / `take_profit_2_r`: mục tiêu theo R.

Không tăng độ “chuẩn” bằng cách hạ ngưỡng cho ra nhiều lệnh. Muốn tối ưu tiếp, hãy lưu forward journal đủ lớn cho từng tài sản/phiên rồi hiệu chỉnh riêng trên dữ liệu Exness ngoài mẫu.
