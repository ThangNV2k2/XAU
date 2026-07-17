# Gold Trend Signal Tool — Spec cho Claude Code

## 1. Mục tiêu

Xây dựng một tool phân tích kỹ thuật (technical analysis) cho **XAU/USD (giá vàng thế giới)**, sinh tín hiệu **BUY / SELL / HOLD** dựa trên tổng hợp nhiều chỉ báo, có backtest để đánh giá độ tin cậy trước khi dùng thật, và gửi cảnh báo qua Telegram khi có tín hiệu mạnh.

**Lưu ý quan trọng (ghi rõ trong README của project):** Đây là công cụ hỗ trợ phân tích kỹ thuật, KHÔNG phải dự đoán chắc chắn. Không có gì đảm bảo lợi nhuận. Không dùng toàn bộ vốn để giao dịch tự động dựa trên tín hiệu này mà không kiểm tra thủ công.

---

## 2. Nguồn dữ liệu

Dùng dữ liệu XAU/USD từ một trong các nguồn free-tier sau (implement theo dạng adapter, dễ đổi nguồn sau này):

- **Twelve Data** (twelvedata.com) — có endpoint `time_series` cho XAU/USD, free tier ~800 requests/day
- **Alpha Vantage** — có `CURRENCY_EXCHANGE_RATE` / FX intraday, free tier giới hạn 25 requests/day (khá ít, chỉ nên dùng backup)
- **GoldAPI.io** — chuyên giá vàng, free tier giới hạn request/tháng

→ Implement class `DataProvider` với interface chung:
```python
class DataProvider(ABC):
    def get_historical(self, interval: str, outputsize: int) -> pd.DataFrame: ...
    def get_latest_price(self) -> float: ...
```
Bắt đầu với Twelve Data làm provider chính (free tier đủ dùng cho polling mỗi 5-15 phút).

API key đọc từ file `.env` (dùng `python-dotenv`), KHÔNG hardcode.

---

## 3. Chỉ báo kỹ thuật (indicators)

Dùng thư viện `pandas-ta` hoặc `ta` (Python) để tính:

1. **RSI (14)** — quá mua (>70) / quá bán (<30)
2. **MACD (12, 26, 9)** — tín hiệu cắt đường signal line
3. **EMA crossover** — EMA(9) cắt EMA(21): golden cross / death cross
4. **Bollinger Bands (20, 2)** — giá chạm/vượt band trên/dưới
5. **ATR (14)** — dùng để đo volatility, hỗ trợ tính stop-loss gợi ý (không phải chỉ báo hướng)

---

## 4. Logic tổng hợp tín hiệu (signal engine)

Mỗi indicator trả về điểm: `+1` (bullish), `-1` (bearish), `0` (neutral).

Tổng hợp có trọng số (weight có thể config trong file `config.yaml`):

```yaml
weights:
  rsi: 1.0
  macd: 1.2
  ema_crossover: 1.0
  bollinger: 0.8
```

- `score = sum(indicator_score * weight)`
- `score >= threshold_buy` → tín hiệu **BUY**
- `score <= threshold_sell` → tín hiệu **SELL**
- còn lại → **HOLD**

`threshold_buy` / `threshold_sell` để trong config, mặc định `+2.0` / `-2.0`.

Output mỗi lần chạy: JSON gồm từng chỉ báo + điểm + tín hiệu tổng hợp + giá hiện tại + timestamp.

---

## 5. Backtest module

**Bắt buộc phải có trước khi dùng tool thật** — đây là phần quan trọng nhất để biết tool có đáng tin không.

- Input: dữ liệu lịch sử (ví dụ 1-2 năm, khung H1 hoặc H4)
- Chạy signal engine trên từng nến lịch sử (walk-forward, không nhìn tương lai — tránh look-ahead bias)
- Giả lập giao dịch: mở lệnh khi có tín hiệu BUY/SELL, đóng khi có tín hiệu ngược hoặc sau N nến
- Output báo cáo:
  - Win rate (%)
  - Tổng số lệnh
  - Profit factor
  - Max drawdown
  - Sharpe ratio (đơn giản hóa)
  - Biểu đồ equity curve (matplotlib, lưu ra PNG)

→ Đây là bước quyết định: nếu win rate backtest < 50-55% hoặc drawdown quá lớn, KHÔNG nên deploy alert thật, cần điều chỉnh weight/threshold hoặc bỏ chỉ báo yếu.

---

## 6. Alerting (Telegram)

- Dùng Telegram Bot API (`python-telegram-bot` hoặc gọi trực tiếp HTTP)
- Bot token + chat_id đọc từ `.env`
- Gửi alert khi tín hiệu đổi từ HOLD → BUY/SELL (tránh spam mỗi lần poll)
- Format message: giá hiện tại, tín hiệu, điểm từng chỉ báo, thời gian

---

## 7. Cấu trúc project đề xuất

```
gold-signal-tool/
├── .env.example
├── config.yaml
├── requirements.txt
├── data_provider/
│   ├── base.py
│   └── twelvedata_provider.py
├── indicators/
│   └── signal_engine.py
├── backtest/
│   ├── backtester.py
│   └── report.py
├── alerting/
│   └── telegram_bot.py
├── main.py          # polling loop chạy live
├── run_backtest.py  # script chạy backtest riêng
└── README.md
```

---

## 8. Task breakdown (implement theo thứ tự)

1. Setup project, requirements.txt, `.env.example`, đọc config từ `config.yaml`
2. Implement `DataProvider` cho Twelve Data, test lấy được historical + latest price
3. Implement `signal_engine.py` tính đủ 4 indicator + tổng hợp điểm
4. Implement `backtester.py` + `report.py`, chạy thử trên data lịch sử, xuất báo cáo + equity curve
5. **Dừng lại đánh giá kết quả backtest trước khi làm bước 6** — nếu win rate quá thấp, quay lại tinh chỉnh weight/threshold hoặc thêm/bớt indicator
6. Implement Telegram alerting
7. Implement `main.py` polling loop (chạy mỗi 5-15 phút tuỳ interval, có logging ra file)
8. Viết README hướng dẫn setup + **disclaimer rủi ro rõ ràng**

---

## 9. Tech stack

- Python 3.11+
- `pandas`, `pandas-ta` (hoặc `ta`), `requests`, `python-dotenv`, `pyyaml`
- `matplotlib` cho backtest report
- `python-telegram-bot` cho alerting
- (Tuỳ chọn) `APScheduler` cho polling loop thay vì while-loop + sleep thô
