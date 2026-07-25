# Gold Signal Tool (XAU/USD)

Bot phân tích kỹ thuật cho XAU/USD, sinh tín hiệu **BUY / SELL / HOLD** từ tổng hợp có trọng số của RSI, MACD, EMA crossover và Bollinger Bands, kèm backtest và cảnh báo qua Telegram.

> **Disclaimer:** Đây là công cụ hỗ trợ phân tích kỹ thuật, **KHÔNG phải dự đoán chắc chắn** và không phải lời khuyên đầu tư. Không có gì đảm bảo lợi nhuận. Giao dịch future dùng đòn bẩy cao — rủi ro mất vốn rất lớn. Không tự động hoá đặt lệnh dựa hoàn toàn vào tín hiệu này mà không kiểm tra thủ công và tự chịu trách nhiệm về quyết định giao dịch của mình.

## 1. Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env
```

Điền vào `.env`:
- `TWELVEDATA_API_KEY` — đăng ký free tại [twelvedata.com](https://twelvedata.com) (free tier: 800 request/ngày, 8 request/phút — đủ dùng cho polling 15 phút/lần).
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` — chưa cần ngay, chỉ cần khi bật alerting thật (xem bước 4).
- `GROQ_API_KEY` — AI chính đọc biểu đồ bằng `qwen/qwen3.6-27b`; tạo khóa tại [Groq Console](https://console.groq.com/keys).
- `GEMINI_API_KEY` — không bắt buộc; dùng làm AI dự phòng khi Groq lỗi hoặc đạt giới hạn.

## 2. Chạy backtest TRƯỚC KHI dùng thật

Đây là bước bắt buộc để biết cấu hình chỉ báo hiện tại có đáng tin hay không:

```bash
python run_backtest.py --interval 4h --years 2
```

Kết quả in ra: `total_trades`, `win_rate`, `profit_factor`, `max_drawdown`, `sharpe`, `net_pnl`, và file `backtest/equity_curve.png`.

Lần chạy đầu sẽ tải dữ liệu lịch sử từ Twelve Data (tốn vài request) rồi cache ra `backtest/cache/*.csv`. Lần sau sẽ đọc từ cache, dùng `--force-refresh` nếu muốn tải lại.

**Nguyên tắc quan trọng:** nếu `win_rate < 50-55%` hoặc `max_drawdown` quá lớn so với `net_pnl`, KHÔNG bật alerting thật — quay lại chỉnh `weights` / `threshold_buy` / `threshold_sell` trong `config.yaml` (hoặc thử interval khác) rồi backtest lại.

## 3. Kiểm tra tín hiệu live (chưa gửi Telegram)

```bash
python main.py --once
```

In ra tín hiệu hiện tại (giá, từng chỉ báo, điểm tổng hợp) mà không cần Telegram, không chạy vòng lặp.

## 4. Bot Telegram hỏi-đáp theo yêu cầu (khuyến nghị dùng cách này)

Thay vì tự poll và đẩy tin (main.py), cách chính để dùng tool là **hỏi khi cần**: nhắn `/signal` hoặc `/gia` cho bot bất cứ lúc nào để xem tình hình hiện tại.

1. Tạo bot qua [@BotFather](https://t.me/BotFather) trên Telegram → lấy `TELEGRAM_BOT_TOKEN`.
2. Nhắn bất kỳ gì cho bot của bạn (VD "hi") để Telegram ghi nhận cuộc hội thoại.
3. Lấy `chat_id`: mở trình duyệt, vào `https://api.telegram.org/bot<TOKEN>/getUpdates` (thay `<TOKEN>` bằng token thật, dính liền không dấu ngoặc) → tìm `"chat":{"id":XXXXXXXXX,...}`.
4. Điền cả `TELEGRAM_BOT_TOKEN` và `TELEGRAM_CHAT_ID` vào `.env`.
5. Chạy:
   ```bash
   python telegram_query_bot.py
   ```
6. Trong Telegram, nhắn `/signal` hoặc `/gia` cho bot → nhận ngay:
   - Giá và trạng thái nguồn (`LIVE`, `DỮ LIỆU TRỄ` hoặc `ĐÓNG CỬA`), không giả dữ liệu cuối phiên là realtime.
   - Một biểu đồ gộp **15m / 1H / 4H**; dữ liệu Twelve Data được yêu cầu ở UTC và chỉ nến đã đóng mới được dùng để xác nhận.
   - Vùng hỗ trợ/kháng cự động từ swing-high, swing-low đa khung và ATR. Hai vùng được tách rời; vùng nhiễu không được dùng làm entry.
   - Trạng thái breakout/breakdown và retest. Bot chỉ lập Entry/SL/TP/khối lượng/đòn bẩy khi đủ chuỗi: ba khung đồng thuận → nến 15m đóng phá vùng → nến sau retest xác nhận → tín hiệu backtest gần đây cùng hướng → giá hiện vẫn ở vùng vào.
   - Phân tích AI mới từ Groq, tự chuyển sang Gemini nếu cần. Luồng `/gia` không tái sử dụng cache AI hoặc cache tin tức.
   - AI nhận ảnh gộp cùng 8 nến OHLC gần nhất của mỗi khung, vùng giá, động lượng và tin tiếng Việt. AI không tự gọi thêm dữ liệu và không được tự tạo hay sửa Entry/SL/TP; nếu code chưa xác nhận entry thì kết luận AI bị ép thành `ĐỨNG NGOÀI`.

7. Nhắn `/dinh` để nhận bản đồ đỉnh chuyên biệt:
   - Quét nến đã đóng 15m/1H/4H/D1 bằng Williams Fractal 5 nến.
   - Lọc nhiễu bằng ZigZag với deviation riêng cho từng khung; điểm ZigZag cuối chưa có swing ngược xác nhận không được cộng uy tín.
   - Gom các đỉnh gần nhau thành vùng, phân biệt đỉnh cũ/mới và chấm uy tín theo khung, số đỉnh hội tụ, phản ứng sau đỉnh và volume nếu nguồn thực sự có.
   - Đỉnh nằm trên giá được trình bày như cản; đỉnh cũ đã được nến 15m đóng vượt được trình bày riêng như hỗ trợ retest tiềm năng.

   Chỉ chat có `chat_id` khớp trong `.env` mới được bot trả lời (người khác nhắn bot sẽ bị bỏ qua, tránh lộ và tốn quota API).

**Quan trọng:** hỗ trợ/kháng cự và retest chỉ làm giảm việc đuổi giá, không biến tín hiệu thành dự đoán chắc chắn. Kết quả backtest gần nhất vẫn **chưa có edge ổn định** (win rate ~47%, profit factor ~0.5, xem mục 2). Không dùng riêng bot hoặc AI làm căn cứ vào lệnh đòn bẩy.

### (Tuỳ chọn) Cách cũ: bot tự poll và đẩy tin khi tín hiệu đổi

```bash
python main.py --dry-run   # test trước, chỉ log không gửi thật
python main.py             # chạy thật, đẩy tin khi tín hiệu đổi trạng thái
```

## 5. Cấu hình (`config.yaml`)

- `weights` — trọng số từng chỉ báo, dùng chung cho cả tín hiệu backtest (BUY/SELL/HOLD) và bias liên tục.
- `threshold_buy` / `threshold_sell` — ngưỡng điểm để ra tín hiệu BUY/SELL (chỉ áp dụng cho tín hiệu backtest, không áp dụng cho bias liên tục).
- `atr_stop_multiplier` — hệ số nhân ATR(14) để gợi ý khoảng cách stop-loss (không ảnh hưởng tín hiệu hướng).
- `live.interval` / `live.poll_minutes` / `live.outputsize` — khung nến và tần suất polling khi chạy live.
- `signal_confirmation` — ngưỡng đồng thuận ba khung, số nến tìm retest, độ rộng vùng retest và số nến tìm tín hiệu backtest xác nhận.
- `resistance_test` — độ gom cụm swing theo ATR, độ rộng vùng và các ngưỡng kiểm tra từ chối giá.
- `peak_map` — các khung của `/dinh`, span Fractal, deviation ZigZag, độ gom vùng và số vùng tối đa cần hiển thị.

## 6. Cấu trúc project

```
data_provider/          # nguồn dữ liệu giá (Twelve Data), có thể thêm provider khác qua interface DataProvider
indicators/              # signal_engine.py — chỉ báo, tín hiệu backtest (compute_signal) + bias liên tục (compute_momentum_bias)
backtest/                # tải dữ liệu lịch sử, mô phỏng giao dịch, báo cáo hiệu quả
alerting/                # gửi cảnh báo Telegram (dùng bởi main.py)
main.py                  # (tuỳ chọn) vòng lặp polling live, tự đẩy tin khi tín hiệu đổi
telegram_query_bot.py    # bot hỏi-đáp /signal, /gia — cách dùng chính
run_backtest.py          # chạy backtest
```
