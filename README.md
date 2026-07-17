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
   - **Bias xu hướng ngắn hạn** (-100% đến +100%, liên tục — luôn có số cụ thể để tham khảo, không chỉ HOLD/0.00): tổng hợp RSI/MACD/EMA trend/Bollinger %B như các thước đo động lượng (momentum), không phải chờ sự kiện cắt hiếm gặp.
   - Biên độ biến động ước lượng cho ~1h và ~4h tới (dựa trên ATR, quy tắc căn bậc hai thời gian).
   - Tín hiệu backtest cũ (BUY/SELL/HOLD) để tham khảo thêm — tín hiệu này thận trọng hơn nên ít khi kích hoạt.

   Chỉ chat có `chat_id` khớp trong `.env` mới được bot trả lời (người khác nhắn bot sẽ bị bỏ qua, tránh lộ và tốn quota API).

**Quan trọng:** con số "bias" là chỉ số kỹ thuật đọc trạng thái động lượng hiện tại, **chưa được backtest riêng và không đảm bảo đúng hướng**. Tín hiệu BUY/SELL/HOLD (mục 2) mới là cái đã backtest — và kết quả gần nhất cho thấy **chưa có edge ổn định** (win rate ~47%, profit factor ~0.5, xem mục 2). Dùng để tham khảo thêm góc nhìn, không phải căn cứ duy nhất để vào lệnh đòn bẩy trên Binance.

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
