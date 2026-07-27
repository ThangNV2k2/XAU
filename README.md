# Binance Futures Gold Signal Tool (XAUUSDT)

Bot phân tích kỹ thuật cho hợp đồng **Binance Futures XAUUSDT perpetual**, sinh tín hiệu từ nến, volume, order book, Mark/Index Price và dữ liệu phái sinh của đúng hợp đồng đang giao dịch.

> **Disclaimer:** Đây là công cụ hỗ trợ phân tích kỹ thuật, **KHÔNG phải dự đoán chắc chắn** và không phải lời khuyên đầu tư. Không có gì đảm bảo lợi nhuận. Giao dịch future dùng đòn bẩy cao — rủi ro mất vốn rất lớn. Không tự động hoá đặt lệnh dựa hoàn toàn vào tín hiệu này mà không kiểm tra thủ công và tự chịu trách nhiệm về quyết định giao dịch của mình.

## 1. Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env
```

Điền vào `.env`:
- Dữ liệu live Binance Futures là public nên **không cần Binance API key**; bot không có quyền đặt lệnh hay đọc tài khoản.
- `TWELVEDATA_API_KEY` — chỉ còn dùng cho backtest XAU/USD tham chiếu ở mục 2, không tham gia Entry/SL live Binance.
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` — chưa cần ngay, chỉ cần khi bật alerting thật (xem bước 4).
- `GROQ_API_KEY` — AI chính đọc biểu đồ bằng `qwen/qwen3.6-27b`; tạo khóa tại [Groq Console](https://console.groq.com/keys).
- `GEMINI_API_KEY` — không bắt buộc; dùng làm AI dự phòng khi Groq lỗi hoặc đạt giới hạn.

## 2. Chạy backtest TRƯỚC KHI dùng thật

Backtest cũ dùng XAU/USD Twelve Data để tham khảo hành vi chỉ báo:

```bash
python run_backtest.py --interval 4h --years 2
```

Kết quả in ra: `total_trades`, `win_rate`, `profit_factor`, `max_drawdown`, `sharpe`, `net_pnl`, và file `backtest/equity_curve.png`.

Lần chạy đầu sẽ tải dữ liệu lịch sử từ Twelve Data (tốn vài request) rồi cache ra `backtest/cache/*.csv`. Lần sau sẽ đọc từ cache, dùng `--force-refresh` nếu muốn tải lại.

**Giới hạn:** kết quả XAU/USD không chứng minh edge cho hợp đồng XAUUSDT mới trên Binance. Chỉ xem đây là kiểm tra logic; muốn đánh giá chiến lược live phải backtest lại bằng lịch sử Binance từ ngày hợp đồng được niêm yết.

## 3. Kiểm tra tín hiệu live (chưa gửi Telegram)

```bash
python main.py --once
```

In ra tín hiệu hiện tại (giá, từng chỉ báo, điểm tổng hợp) mà không cần Telegram, không chạy vòng lặp.

## 4. Bot Telegram hỏi-đáp và tự canh Entry (khuyến nghị)

Một tiến trình `telegram_query_bot.py` vừa trả lời lệnh, vừa tự canh XAUUSDT nền và đẩy cảnh báo khi setup đủ điều kiện; không cần chạy thêm `main.py`.

1. Tạo bot qua [@BotFather](https://t.me/BotFather) trên Telegram → lấy `TELEGRAM_BOT_TOKEN`.
2. Nhắn bất kỳ gì cho bot của bạn (VD "hi") để Telegram ghi nhận cuộc hội thoại.
3. Lấy `chat_id`: mở trình duyệt, vào `https://api.telegram.org/bot<TOKEN>/getUpdates` (thay `<TOKEN>` bằng token thật, dính liền không dấu ngoặc) → tìm `"chat":{"id":XXXXXXXXX,...}`.
4. Điền cả `TELEGRAM_BOT_TOKEN` và `TELEGRAM_CHAT_ID` vào `.env`.
5. Chạy:
   ```bash
   python telegram_query_bot.py
   ```
   Ngay khi khởi động, bot gửi `CANH TỰ ĐỘNG XAUUSDT ĐÃ BẬT`. Mặc định bot kiểm tra mỗi 30 giây nhưng chỉ cảnh báo khi cùng logic với `/dinh` đã xác nhận, giá sát/vào vùng Entry và kế hoạch có R:R hợp lệ.
   - `/canh` — xem lần kiểm tra, gate/lý do và lỗi gần nhất.
   - `/canhbat` / `/canhtat` — bật hoặc tắt canh nền mà không dừng bot.
   - Cùng một setup được chống gửi lặp; có thể báo hai giai đoạn `SÁT ENTRY` và `ĐÃ VÀO ENTRY`.
   - Cảnh báo tự động không gọi AI nên không tốn quota Groq/Gemini; nó gửi Entry/SL/TP/khối lượng/cách đặt lệnh và biểu đồ 15m/1H/D1 từ code định lượng.
6. Trong Telegram, nhắn `/signal` hoặc `/gia` cho bot → nhận ngay:
   - Giá bid/ask realtime từ Binance WebSocket; Last, Mark, Index, funding và open interest từ public Futures API.
   - Entry/vùng giá dựa trên nến giao dịch XAUUSDT; Mark Price dùng để theo dõi rủi ro PnL chưa thực hiện/thanh lý; Index và basis dùng để phát hiện lệch giá.
   - Một biểu đồ gộp **15m / 1H / 4H** từ Binance Futures, chỉ dùng nến đã đóng để xác nhận.
   - Volume và order book thật của XAUUSDT, không còn dùng PAXG làm proxy.
   - Vùng hỗ trợ/kháng cự động từ swing-high, swing-low đa khung và ATR. Hai vùng được tách rời; vùng nhiễu không được dùng làm entry.
   - Trạng thái breakout/breakdown và retest. Bot chỉ lập Entry/SL/TP/khối lượng/đòn bẩy khi đủ chuỗi: ba khung đồng thuận → nến 15m đóng phá vùng → nến sau retest xác nhận → tín hiệu backtest gần đây cùng hướng → giá hiện vẫn ở vùng vào.
   - `/gia` và `/dinh` cùng dùng bộ lọc thanh khoản 1H; cuối tuần hoặc khi volume/spread không đạt ngưỡng thì cả hai đều khóa Entry.
   - Phân tích AI mới từ Groq, tự chuyển sang Gemini nếu cần. Luồng `/gia` không tái sử dụng cache AI hoặc cache tin tức.
   - AI nhận ảnh gộp cùng 8 nến OHLC gần nhất của mỗi khung, vùng giá, động lượng và tin tiếng Việt. AI không tự gọi thêm dữ liệu và không được tự tạo hay sửa Entry/SL/TP; nếu code chưa xác nhận entry thì kết luận AI bị ép thành `ĐỨNG NGOÀI`.

7. Nhắn `/dinh` để nhận bản đồ đỉnh chuyên biệt:
   - Bot gửi biểu đồ **15m / 1H / D1 nến đã đóng** với EMA9/EMA21, volume thật và các vùng kháng cự/hỗ trợ: 15m tìm Entry, 1H xác nhận, D1 kiểm tra xu hướng lớn.
   - Quét nến đã đóng 15m/1H/4H/D1 bằng Williams Fractal 5 nến.
   - Lọc nhiễu bằng ZigZag với deviation riêng cho từng khung; điểm ZigZag cuối chưa có swing ngược xác nhận không được cộng uy tín.
   - Gom các đỉnh gần nhau thành vùng, phân biệt đỉnh cũ/mới và chấm uy tín theo khung, số đỉnh hội tụ, phản ứng sau đỉnh và volume nếu nguồn thực sự có.
   - Đỉnh nằm trên giá được trình bày như cản; đỉnh cũ đã được nến 15m đóng vượt được trình bày riêng như hỗ trợ retest tiềm năng.
   - Sau bản đồ, AI review ngắn theo nến đóng và dữ liệu phái sinh hiện tại. Code khóa kết luận ở `CHỜ`, `CANH LONG` hoặc `CANH SHORT`; AI không được đảo hướng, tự tạo mốc giá hay Entry/SL/TP.
   - `CANH LONG` chỉ được mở khi 15m/1H/4H cùng tăng, 15m phá/retest, 1H đóng trên cản và D1 không giảm mạnh. `CANH SHORT` dùng điều kiện đối xứng và D1 không được tăng mạnh. Thiếu một điều kiện thì kết luận là `CHỜ`.
   - Khi đã xác nhận, code mới tính vùng Entry retest, SL ngoài vùng theo ATR, TP1 ở 1R và TP2 tại cản/hỗ trợ cấu trúc kế tiếp. Nếu TP2 không đạt tối thiểu 1.5R thì bot ghi `KHÔNG VÀO`.
   - Kế hoạch kèm khối lượng theo rủi ro tài khoản, đòn bẩy isolated, cách chốt 50% tại TP1, dời SL về Entry và time-stop. Khi chưa xác nhận, `/dinh` chỉ đưa mốc cần chờ và yêu cầu gọi lại sau nến 15m đóng.
   - Hướng dẫn thực thi ghi tuần tự cách đặt Limit Entry, Stop-Market theo Mark Price, TP1 50%, TP2 phần còn lại và hủy các lệnh đóng vị thế còn treo.
   - Thanh khoản được đo bằng median volume 1H gần nhất so với median giờ ngày thường. Mặc định thứ Bảy/Chủ nhật chỉ vẽ vùng và không sinh Entry (`peak_liquidity.block_weekend_entries: true`).

   Chỉ chat có `chat_id` khớp trong `.env` mới được bot trả lời (người khác nhắn bot sẽ bị bỏ qua, tránh lộ và tốn quota API).

**Quan trọng:** hỗ trợ/kháng cự và retest chỉ làm giảm việc đuổi giá, không biến tín hiệu thành dự đoán chắc chắn. Dữ liệu đúng sàn loại bỏ sai lệch nguồn nhưng không tạo ra edge; không dùng riêng bot hoặc AI làm căn cứ vào lệnh đòn bẩy.

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
- `market_data` — endpoint REST/WebSocket public của Binance Futures.
- `order_book` — depth thật của XAUUSDT.
- `signal_confirmation` — ngưỡng đồng thuận ba khung, số nến tìm retest, độ rộng vùng retest và số nến tìm tín hiệu backtest xác nhận.
- `resistance_test` — độ gom cụm swing theo ATR, độ rộng vùng và các ngưỡng kiểm tra từ chối giá.
- `peak_map` — các khung của `/dinh`, span Fractal, deviation ZigZag, độ gom vùng và số vùng tối đa cần hiển thị.
- `peak_execution` — đệm Entry/SL theo ATR và R:R cấu trúc tối thiểu trước khi `/dinh` được phép sinh kế hoạch.
- SL cấu trúc bị giới hạn tối đa khoảng 7 giá tính từ mọi điểm khớp trong vùng Entry; nếu cần xa hơn bot bỏ kèo. Sau khi khớp, nến 1m đóng sai phía vùng retest kèm áp lực ngược sẽ phát cảnh báo `RETEST THẤT BẠI — CẮT NGAY`, không chờ hard SL.
- `peak_liquidity` — tỷ lệ volume tối thiểu, spread tối đa và chính sách chặn Entry cuối tuần.
- `auto_alerts` — quét cấu trúc nền mỗi 30 giây; khi có setup LONG/SHORT bot gửi cảnh báo code ngay, gọi AI ở tác vụ riêng để gửi thông báo xác thực thứ hai, rồi canh giá mỗi 10 giây. Nến 1m cập nhật mỗi phút; vòng canh nhanh không gọi AI.

## 6. Cấu trúc project

```
data_provider/          # Binance Futures cho live; Twelve Data chỉ cho backtest tham chiếu
indicators/              # signal_engine.py — chỉ báo, tín hiệu backtest (compute_signal) + bias liên tục (compute_momentum_bias)
backtest/                # tải dữ liệu lịch sử, mô phỏng giao dịch, báo cáo hiệu quả
alerting/                # gửi cảnh báo Telegram (dùng bởi main.py)
main.py                  # (tuỳ chọn) vòng lặp polling live, tự đẩy tin khi tín hiệu đổi
telegram_query_bot.py    # bot hỏi-đáp /signal, /gia, /dinh — cách dùng chính
run_backtest.py          # chạy backtest
```
