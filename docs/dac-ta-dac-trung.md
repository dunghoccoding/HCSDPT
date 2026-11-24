# Đặc tả trích rút đặc trưng và tìm kiếm chân dung

Phiên bản thiết kế: `geometry-hsv-lbp-v1`. Đây là đặc tả để triển khai phần 3–4 của bài tập lớn, giữ cấu trúc F1–F5 trong báo cáo. Các mốc đo và ngưỡng dưới đây là lựa chọn của project, cần kiểm tra trực quan trên tập thử trước khi cố định phiên bản. Chưa có kết quả thực nghiệm về chất lượng tìm kiếm.

## Dữ liệu và phạm vi

Dataset `data_best` đã kiểm kê có 1.249 ảnh JPG, tất cả RGB, kích thước 178 × 218 pixel; 563 tên file chứa `celeba`, 686 tên file chứa `male_faces_strict`. Các tiền tố chỉ dùng làm nhãn nhóm dữ liệu, không thay thế bằng chứng nguồn gốc hoặc nhãn danh tính.

Mục tiêu là truy hồi ảnh có ngoại hình tương đồng. Điểm xếp hạng không phải xác suất cùng người. Chỉ đánh giá truy hồi cùng danh tính khi có nhãn độc lập.

## Quy ước tọa độ và công cụ

Dùng MediaPipe Tasks Face Landmarker ở chế độ IMAGE. Công cụ trả các landmark chuẩn hóa; project chỉ dùng x, y của các điểm thuộc mesh cơ sở, không dùng z hoặc iris để đo. Chuyển về tọa độ pixel trước khi tính khoảng cách: p_i = (x_i × width, y_i × height). Không đo trực tiếp trên x, y chuẩn hóa vì ảnh không vuông.

Thiết lập tối đa hai mặt để phát hiện trường hợp nhiều mặt; chỉ chấp nhận đúng một mặt được phát hiện. Các ngưỡng detection/presence khởi tạo 0,5, lưu trong cấu hình. Đây là ngưỡng của mô hình, không phải độ tin cậy riêng cho từng landmark. Lưu hash model, phiên bản thư viện và cấu hình khi chạy.

Trái/phải bên dưới theo người trong ảnh, nhất quán với tên nhóm của MediaPipe. Không tự lật ảnh. Trên ảnh chính diện không phản chiếu, mắt phải của người ở bên trái ảnh.

Mốc nhóm chính:

| Vùng | Chỉ số landmark sử dụng |
| --- | --- |
| Mắt phải | Hai khóe 33, 133; mí trên 159; mí dưới 145 |
| Mắt trái | Hai khóe 362, 263; mí trên 386; mí dưới 374 |
| Lông mày phải | Đường giữa tính từ các cặp (70,46), (63,53), (105,52), (66,65), (107,55) |
| Lông mày trái | Đường giữa tính từ các cặp (336,285), (296,295), (334,282), (293,283), (300,276) |
| Đường viền | Má 234,454; hàm 172,397; vùng trên 127,356; cằm 152 |
| Mũi | Gốc 168; đáy 2; cánh 98,327 |
| Miệng | Khóe 61,291; môi ngoài 0,17; môi trong 13,14 |

Tâm mắt R/L là trung bình các landmark duy nhất của contour mắt tương ứng trong bảng kết nối chính thức. Tâm mỗi lông mày là trung bình 5 điểm giữa ở bảng trên. G là trung điểm hai tâm lông mày, E là trung điểm hai tâm mắt, M là trung điểm hai khóe miệng. Các lựa chọn vùng trên, má, hàm là đại diện hình học của mesh, không phải phép đo giải phẫu chính xác.

Đặt trục u từ tâm mắt phải tới tâm mắt trái; trục v vuông góc u và hướng về cằm. Với p bất kỳ, X(p)=(p−G)·u, Y(p)=(p−G)·v. G có tọa độ (0,0); các phép đo ngang/dọc dưới đây là phép chiếu trên hệ trục này.

## Căn chỉnh và chất lượng đầu vào

Dùng phép biến đổi similarity 2D, không kéo giãn độc lập ngang/dọc. Đưa hai tâm mắt về (57,82) và (121,82) trên canvas 178 × 218. Áp dụng cùng ma trận cho ảnh và landmark, không phát hiện landmark lại trên ảnh đã warp. Giữ ảnh gốc; dùng nội suy bilinear cho ảnh, nearest cho mask pixel hợp lệ. Pixel padding không được tham gia histogram.

F1–F3 đo trên tọa độ landmark với hệ trục cục bộ; F4–F5 lấy pixel trên ảnh căn chỉnh. Chưa dùng cân bằng histogram hoặc CLAHE trong cấu hình cơ sở để tránh thay đổi màu và texture ngoài ý muốn.

Khởi tạo các kiểm tra: tất cả điểm dùng để đo phải hữu hạn; Wf và HGC lớn hơn 20 pixel trên ảnh gốc; độ nghiêng đường mắt không quá 25 độ; các ROI phải có ít nhất 25 tâm pixel hợp lệ sau xử lý mask. Đây là ngưỡng thử nghiệm, chưa khẳng định đã phù hợp toàn dataset. Roll nhỏ không bảo đảm mặt chính diện: yaw/pitch và che khuất cần kiểm tra thêm trong tập thử.

Ảnh bị lỗi lưu trạng thái riêng: `unreadable`, `no_face`, `multiple_faces`, `invalid_geometry`, `invalid_roi`. Không thay đặc trưng lỗi bằng số 0. Không dùng việc phát hiện được landmark làm bằng chứng đủ về chất lượng mặt.

## F1 hình học tổng thể

Đặt Wf=|X(p454)−X(p234)|; HGC=Y(p152); Wj=|X(p397)−X(p172)|; Wu=|X(p356)−X(p127)|. Khoảng cách d là Euclid trong tọa độ pixel, góc tính bằng radian.

| Thứ tự | Tên | Công thức |
| --- | --- | --- |
| 1 | Cao trên rộng | HGC/Wf |
| 2 | Rộng hàm | Wj/Wf |
| 3 | Rộng vùng trên | Wu/Wf |
| 4 | Dài vùng cằm | [Y(p152)−Y(M)]/HGC |
| 5 | Hàm so với vùng trên | Wj/Wu |
| 6 | Góc cằm | angle(p172−p152, p397−p152) |
| 7 | Bất đối xứng | Trung bình khoảng cách giữa mỗi điểm và ảnh phản xạ của điểm đối xứng, chia Wf |

Các cặp đối xứng dùng cho thuộc tính 7: (234,454), (172,397), (127,356), (33,263), (133,362), (61,291). Phản xạ quanh X=0 của hệ tọa độ cục bộ.

Hai công thức số 2 và 5 trong báo cáo bị trùng nếu Wcheek=Wf. Thiết kế triển khai chọn thuộc tính 5 là Wj/Wu để so hàm với vùng trên và giữ đủ 7 chiều. Cần sửa công thức, tên và ý nghĩa tương ứng trong phần 2 trước khi nộp báo cáo. Kiểm tra tương quan các chiều trên tập thử; không tự thay đổi công thức giữa các lượt chạy.

Wu là độ rộng vùng mặt trên gần thái dương, không phải đường chân tóc. Sửa cách diễn đạt trong báo cáo để tránh khẳng định đo được toàn bộ trán bằng landmark.

## F2 mắt và lông mày

Đặt eye_width_R=|X(p133)−X(p33)|, eye_width_L=|X(p263)−X(p362)|; eye_height_R=|Y(p145)−Y(p159)|, eye_height_L=|Y(p374)−Y(p386)|. Các mẫu số phải dương và vượt epsilon.

| Thứ tự | Công thức |
| --- | --- |
| 1 | d(tâm mắt trái, tâm mắt phải)/Wf |
| 2 | eye_width_L/Wf |
| 3 | eye_width_R/Wf |
| 4 | eye_height_L/eye_width_L |
| 5 | eye_height_R/eye_width_R |
| 6 | Y(E)/HGC; tức Ytop=Y(G)=0 |
| 7 | [Y(tâm mắt trái)−Y(tâm mày trái)]/HGC |
| 8 | [Y(tâm mắt phải)−Y(tâm mày phải)]/HGC |
| 9 | Góc đường nối hai đầu đường giữa mày trái so với u |
| 10 | Góc đường nối hai đầu đường giữa mày phải so với u |

Với mỗi đường mày, sắp hai đầu theo X tăng rồi tính atan2(ΔY,ΔX), tránh hai phía lệch nhau π do thứ tự chỉ số. Đây là độ nghiêng trên ảnh đã bỏ roll, không suy ra độ dày hoặc mật độ lông mày.

## F3 mũi và miệng

Đặt Ln=Y(p2)−Y(p168), Wn=|X(p327)−X(p98)|, Wm=|X(p291)−X(p61)|, Hm=Y(p17)−Y(p0).

| Thứ tự | Công thức |
| --- | --- |
| 1 | Ln/HGC |
| 2 | Wn/Wf |
| 3 | Ln/Wn |
| 4 | Wm/Wf |
| 5 | Hm/HGC |
| 6 | [Y(p13)−Y(p0)]/HGC |
| 7 | [Y(p17)−Y(p14)]/HGC |
| 8 | [Y(M)−Y(p2)]/HGC |
| 9 | [Y(p152)−Y(M)]/HGC |
| 10 | [Y(p2)−Y(E)]/HGC |

Các kích thước âm không hợp lý được đánh lỗi hình học, không âm thầm lấy trị tuyệt đối để che lỗi thứ tự. F1 số 4 và F3 số 9 có cùng thông tin theo đặc tả gốc; cần ghi nhận khi đánh giá vì thông tin này nhận trọng số ở hai nhóm.

## ROI trên ảnh căn chỉnh

Các vùng dưới đây là công thức đề xuất của project, phải xuất overlay trên ảnh thử để kiểm tra. Đặt C=(p234+p454)/2, mặt nạ mặt là polygon contour chính thức. Mọi ROI giao với mặt nạ mặt và mask pixel thật của phép warp.

ROI da hai má là ellipse: tâm phía phải (p234+p2)/2, phía trái (p454+p2)/2; bán trục ngang 0,075Wf, dọc 0,065HGC, hướng theo u,v. Loại pixel thuộc mắt, mày, mũi, môi bằng các mask bộ phận nở thêm 2 pixel.

ROI trán là ellipse tâm G−0,22HGC·v, bán trục ngang 0,15Wf, dọc 0,09HGC. Đây là vùng ứng viên, không bảo đảm toàn da. Kiểm tra màu bằng tham chiếu hai má trước khi dùng cho F4.

ROI texture gồm trán ứng viên, má trái, má phải, vùng mắt và vùng miệng. Vùng mắt là bounding rectangle trong trục u,v của hai contour mắt và hai mày, mở rộng 0,03Wf ngang và 0,04HGC dọc. Vùng miệng là rectangle của contour môi ngoài, mở rộng 0,05Wf ngang và 0,06HGC dọc. F5 giữ tóc/râu hoặc môi nếu nằm trong ROI đã định nghĩa; không gọi toàn bộ F5 là texture da.

## F4 màu sắc với 42 chiều

Chuyển RGB sang HSV theo quy ước OpenCV uint8, sau đó chuẩn hóa H/180, S/255, V/255; histogram HS có 8×4 bin, histogram V có 4 bin. Các cạnh bin đều trong [0,1], bin cuối bao gồm 1.

Kiểm tra trán trong CIELAB thang chuẩn (L từ 0 đến 100): màu tham chiếu là trung vị từng kênh Lab của các pixel hai má. Pixel trán đạt nếu khoảng cách Euclid Lab đến tham chiếu ≤20; trán hợp lệ khi có ít nhất 25 pixel đạt và chiếm ít nhất 50% pixel ROI trán. Các ngưỡng là khởi tạo; hai má có râu hoặc bóng vẫn có thể làm lệch tham chiếu.

Để xử lý vòng Hue mà giữ 6 giá trị thống kê, dịch H cố định: Hc=(H+0,5) mod 1, dùng median(Hc), std(Hc), median(S), std(S), median(V), std(V). Đặt thứ tự vector thống kê là [median(Hc),median(S),median(V),std(Hc),std(S),std(V)]. Cách này chuyển điểm nối về vùng đối diện màu đỏ, giảm lỗi Hue quanh 0/1 cho các ROI da; không phải giải pháp vòng tổng quát cho mọi màu. Histogram HS vẫn dùng H gốc. Báo cáo phải ghi rõ phép dịch này.

Tính 6 thống kê, HS32 và V4 riêng cho từng ROI; gộp vector theo trọng số má trái 0,4, má phải 0,4, trán 0,2 nếu trán đạt; nếu không, hai má 0,5–0,5. Đây là trung bình có trọng số của các thống kê ROI, không phải trung vị của toàn bộ pixel gộp. Hai histogram riêng sau gộp đều có tổng 1. Không Z-score F4 trong cấu hình cơ sở.

Trước cosine, chia ba khối Fstat6, HS32, V4 cho norm L2 riêng rồi ghép với hệ số sqrt(1/3). Cosine trên vector F4 sau biến đổi tương đương trung bình cosine của ba khối; cách này tránh khối thống kê tự chi phối chỉ do thang đo. Nếu một khối có norm ≤1e−8, đánh F4 không hợp lệ trong bản đầu. Lưu cả F4 thô lẫn vector dùng để so khớp. Đây là bổ sung cần đưa vào mục chuẩn hóa phần 2.

## F5 texture với 50 chiều

Dùng `skimage.feature.local_binary_pattern(gray_uint8, P=8, R=1, method='uniform')`. Chốt biến thể uniform bất biến quay của thư viện, 10 nhãn từ 0 đến 9; không nhầm với biến thể nriu2 có số bin khác.

Tính LBP trên toàn ảnh xám căn chỉnh trước, sau đó lấy tâm pixel trong từng mask ROI đã co 1 pixel. Không tô vùng ngoài ROI bằng 0 trước khi tính LBP vì sẽ tạo biên giả. Đếm 10 bin, chuẩn hóa L1 riêng từng ROI; ghép theo thứ tự trán, má trái, má phải, mắt, miệng. Khi đủ cả năm ROI, vector có 50 chiều; có thể chia toàn vector cho 5 để tổng bằng 1 mà không đổi cosine.

Trán F5 là ROI texture, không dùng bộ lọc màu da của F4. Nếu ROI thiếu số pixel tối thiểu, đánh F5 lỗi thay vì đổi số chiều hoặc thay histogram bằng 0.

## Chuẩn hóa và so khớp

Fit mean và std (ddof=0) từng chiều F1–F3 trên gallery hợp lệ, không fit trên truy vấn hoặc tập test. Chiều có std<1e−8 được đặt z=0 cho mọi ảnh và truy vấn, đồng thời lưu cờ chiều hằng. Không chia epsilon để khuếch đại nhiễu ở chiều hằng. Lưu bộ tham số cùng snapshot gallery.

Với từng nhóm, cosine(A,B)=dot(A,B)/(norm(A)norm(B)); clip sai số số học về [−1,1]. Nếu norm một vector ≤1e−8, không định nghĩa cosine; trả lỗi rõ ràng. Không quy ước 0/0 thành điểm hoàn hảo. F1–F3 có thể có điểm âm sau Z-score; F4–F5 không âm trong cấu hình này.

S=0,25S1+0,20S2+0,25S3+0,10S4+0,20S5. Điểm tổng nằm trong [−0,7;1] khi đủ năm nhóm. Hiển thị điểm thập phân, không gắn nhãn phần trăm nhận diện. Bản đầu chỉ xếp hạng các ảnh gallery và truy vấn có đầy đủ năm nhóm; vẫn lưu toàn bộ ảnh lỗi trong CSDL để thống kê. Nếu số ảnh hợp lệ dưới 1.000, phải sửa pipeline hoặc bổ sung dữ liệu, không loại ảnh mà bỏ qua yêu cầu quy mô.

Sắp giảm dần theo S; hòa điểm dùng image_id tăng dần để kết quả lặp lại được. Trả min(5,N) ảnh, cảnh báo rõ nếu gallery có ít hơn 5 ảnh. Có chế độ loại ảnh trùng hash truy vấn khi đánh giá; demo mặc định giữ ảnh trùng để kiểm tra pipeline.

Cosine trên vector Z-score chủ yếu so hướng lệch so với trung bình, có thể cho điểm 1 với các vector cùng hướng nhưng khác độ dài. Sau khi có baseline, so với khoảng cách Euclid chuẩn hóa theo số chiều cho F1–F3 trên tập validation. Mọi thay đổi metric phải mang phiên bản mới và cập nhật báo cáo.

## Hợp đồng lưu trữ

SQLite quản lý các bảng `images`, `feature_sets`, `image_features`, `normalization_params`, `search_runs`, `search_results`. Khóa đặc trưng là (image_id, feature_set_id, normalization_id). Lưu vector float32 little-endian cùng số chiều; phép tính thống kê và tích vô hướng dùng float64. Metadata ghi rõ thứ tự chiều, model hash, cấu hình ROI, trạng thái lỗi, hash ảnh và snapshot gallery. Ảnh gốc lưu tại thư mục; đường dẫn trong CSDL tính tương đối với project.

Lưu landmark gốc, ma trận căn chỉnh và đường dẫn ảnh overlay để minh họa. Không tự ghi ảnh upload vào gallery. Khi thêm ảnh gallery, dùng normalization snapshot hiện tại hoặc xây snapshot mới cho toàn gallery; không trộn vector từ hai bộ chuẩn hóa trong cùng lượt tìm kiếm.

## Thứ tự triển khai và điều kiện nghiệm thu

1. Kiểm kê toàn dataset, hash file và kiểm tra trùng nội dung; lưu manifest.
2. Chọn 50–100 ảnh phân tầng theo nhóm tên file và kiểm tra thủ công các trường hợp nghiêng, cười, có kính/râu, ánh sáng khác nhau. Xuất overlay landmark, trục và ROI. Xác nhận mốc má, hàm, vùng trên và vùng môi trước khi chạy batch.
3. Trích F1–F5 trên tập thử: kiểm tra số chiều 7/10/10/42/50, không NaN/Inf, histogram đúng tổng, không đưa padding vào ROI. Kiểm tra các công thức và cập nhật mọi thay đổi trước khi cố định phiên bản.
4. Xây SQLite và trích toàn bộ 1.249 ảnh. Xuất số thành công/thất bại theo từng lý do, không báo 100% khi chưa đo.
5. Fit chuẩn hóa, viết hàm truy vấn và kiểm tra lưu/đọc vector giữ nguyên thứ tự và kết quả.
6. Với ảnh đã có và vector hợp lệ, điểm tự so phải gần 1 (sai số 1e−6); ảnh trùng vector có thể hòa điểm nên kiểm tra thuộc nhóm đồng hạng cao nhất. Kiểm tra ảnh không mặt và nhiều mặt trả đúng lỗi.
7. Xuất một truy vấn gồm ảnh gốc/căn chỉnh, overlay, vector thô/chuẩn hóa, S1–S5 và phép cộng điểm Top-5. Mọi con số minh họa lấy từ chương trình.
8. Làm demo Streamlit: upload, xem tiền xử lý, xem Top-5 và bảng điểm, tải kết quả JSON/CSV. Tách thời gian trích đặc trưng và thời gian so khớp.
9. Đánh giá trên 30–50 truy vấn độc lập: Hit@5/Recall@5 nếu có danh tính; Precision@5 hoặc nDCG@5 nếu có nhãn tương đồng được chấm độc lập. Loại ảnh tự trùng trong đánh giá chất lượng; điều chỉnh trọng số trên validation và giữ test riêng. So sánh đầy đủ với bỏ từng nhóm, chuẩn hóa lại tổng trọng số còn 1.

## Nguồn kỹ thuật

- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker): mô hình và đầu ra landmark. Các công thức đặc trưng, ROI và ngưỡng trong project là thiết kế riêng.
- [MediaPipe face mesh connections](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py): nhóm kết nối mắt, lông mày, môi và contour dùng để định nghĩa mask. Cần lưu bản/commit thực tế khi triển khai.
- [scikit-image local binary pattern](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local_binary_pattern): API và các biến thể LBP.
