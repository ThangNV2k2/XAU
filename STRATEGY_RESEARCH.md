# Ghi chú nghiên cứu chiến lược XAUUSDT

## Kết luận áp dụng

- Điểm `70/80/90+` là điểm **chất lượng điều kiện**, không phải xác suất thắng. Tier hiện tại: B `70–79`, A `80–89`, S `90+`; tier C dưới 70 không tạo setup.
- Hệ thống mặc định `paper_only: true`. Không suy ra edge từ vài lệnh hoặc tự tăng khối lượng vì điểm cao.
- Thứ tự quyết định: khóa tin vĩ mô/thanh khoản → đồng thuận 15m/1H/4H và D1 không đối nghịch → phá vùng bằng giá đóng → retest → kiểm tra sweep/FOMO → R:R sau chi phí → chấm tier.
- “Quét hai đầu”, FOMO và bẫy thanh khoản chỉ được ghi là **dấu hiệu từ giá/volume**. OHLC, order book và open interest không đủ để quy kết hành vi cho một quỹ hay market maker cụ thể.

## Những dữ liệu chính thức đã tham khảo

### Mỹ

- [Federal Reserve — FOMC calendars](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm): dùng ngày kết thúc cuộc họp làm mốc quyết định và chặn setup trong cửa sổ rộng quanh sự kiện. Fed thường có tám cuộc họp định kỳ mỗi năm.
- [CME Group — Gold futures volume](https://www.cmegroup.com/markets/metals/precious/gold.volume.html): dùng để kiểm tra bối cảnh thanh khoản thị trường futures; bot vẫn chỉ dùng volume đúng hợp đồng Binance cho quyết định live.

### Anh / thị trường London

- [LBMA — Clearing data](https://www.lbma.org.uk/prices-and-data/clearing-data): cung cấp số liệu trung bình ngày về volume, giá trị và số giao dịch thanh toán vàng London; phù hợp để hiểu thanh khoản nền, không phải trigger vào lệnh 15m.
- [ONS — Release calendar](https://www.ons.gov.uk/releasecalendar): nguồn lịch công bố kinh tế Anh. Các sự kiện liên quan GBP/risk sentiment cần nhập vào `macro_guard.manual_events` sau khi xác nhận giờ UTC.
- [World Gold Council — Gold liquidity](https://www.gold.org/goldhub/research/relevance-of-gold-as-a-strategic-asset/key-attributes-liquidity): cho thấy thanh khoản vàng phân bổ giữa OTC, futures và ETF; vì vậy volume của một sàn không đại diện toàn bộ thị trường vàng.

### Trung Quốc

- [Shanghai Gold Exchange — Trading hours](https://en.sge.com.cn/eng_trading_ProductsIntroduce): phiên đêm 20:00–02:30 và phiên ngày 09:00–15:30 giờ Trung Quốc (có ngoại lệ ngày lễ). Đây là cơ sở để so sánh volume theo cùng giờ thay vì một median chung cả ngày.
- [National Bureau of Statistics of China — Press releases](https://www.stats.gov.cn/english/PressRelease/): nguồn xác nhận lịch/thời điểm dữ liệu kinh tế Trung Quốc trước khi thêm sự kiện thủ công.

## Quy tắc triển khai

1. Volume hiện tại được so với median **cùng giờ của ngày thường**, cần tối thiểu 24 mẫu; cuối tuần hoặc spread quá rộng thì khóa Entry.
2. Buy-side sweep: giá xuyên vùng cản nhưng đóng trở lại bên dưới với râu đủ lớn. Sell-side sweep dùng điều kiện đối xứng tại hỗ trợ đã chuyển đổi. Nếu hai phía cùng bị sweep trong cửa sổ ngắn thì đứng ngoài.
3. FOMO extension: nến quá rộng/thân quá lớn so với ATR hoặc giá cách EMA21 quá xa; không đuổi theo nến đó.
4. Tin HIGH chặn Entry mới. FOMC được lấy tự động từ Fed; nếu chưa từng tải được lịch chính thức thì bot fail-closed. CPI/NFP/PCE/BoE/NBS hiện phải nhập thủ công bằng giờ UTC để tránh parser lịch bên ngoài bị thay đổi âm thầm.
5. Backtest dùng nến đã đóng, tín hiệu chỉ có hiệu lực từ nến kế tiếp, giả định stop xảy ra trước target nếu cùng chạm trong một nến, chốt 50% TP1 rồi dời phần còn lại về hòa vốn, và trừ phí/trượt giá/spread.

## Điều kiện trước khi cân nhắc chạy tiền thật

- Tối thiểu 100 lệnh out-of-sample hoặc forward paper trade, có đủ nhiều giai đoạn biến động và sự kiện vĩ mô.
- Profit factor tối thiểu `1.20`, average net R dương sau toàn bộ chi phí, drawdown phù hợp với mức chịu lỗ thực tế.
- Xem riêng thống kê từng tier; khoảng tin cậy 95% của win rate phải đủ hẹp. Không dùng tier S nếu mẫu tier S còn quá ít.
- Kiểm tra fee tier thực tế của tài khoản và chất lượng fill; bot chỉ mô phỏng chạm giá, không biết lệnh thật có khớp hay không.
