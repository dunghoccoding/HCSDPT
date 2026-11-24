# Tìm kiếm chân dung theo ngoại hình

Thiết kế: `geometry-hsv-lbp-v1`, theo `docs/dac-ta-dac-trung.md`.

## Run the public source repository

This public version includes source code, tests, documentation and the official
MediaPipe model. It does NOT include portrait datasets, facial feature databases,
the illustrated PDF or generated image artifacts. Numbers below describe the
local development run; they are not bundled results in this public checkout.

Use Python 3.13 and prepare a dataset you are authorized to use in `data_best/`.

```powershell
git clone https://github.com/dunghoccoding/HCSDPT.git
cd HCSDPT
python -m pip install -r requirements.txt
# Add your images to data_best/ before the following commands.
python -m portrait_search.inventory
python -m portrait_search.build_gallery
python -m portrait_search.verify
python run_demo.py
```

Open http://127.0.0.1:8501. Use a sufficiently large, varied gallery; constant
feature groups cannot be compared by cosine after Z-score normalization.
Installed packages, temporary files and uploaded queries are excluded from Git.

## Trạng thái

Đã có kiểm kê, landmark/căn chỉnh, ROI, F1–F5, SQLite, snapshot chuẩn hóa và
tìm kiếm Top-5 bằng dòng lệnh. Đã trích đủ 119 chiều cho 1.146 ảnh đại diện;
103 bản trùng chỉ lưu liên kết trong DB, không trích lại hay dùng để fit.
Đã có giao diện Streamlit để trình diễn. Chưa có đánh giá chất lượng có nhãn độc lập.

## Mở giao diện demo

```powershell
python run_demo.py
```

Mở trình duyệt tại **http://127.0.0.1:8501**. Có thể nhấp đúp `start_demo.bat`
để khởi động; giữ cửa sổ terminal mở trong lúc demo, nhấn Ctrl+C để dừng.
`app.py` là giao diện; `run_demo.py` hỗ trợ thư viện cục bộ trong `.demo-deps`
và chỉ phục vụ trên máy hiện tại. Nếu chuyển sang máy khác, cài `requirements.txt`
và mang theo dataset, model, DB cùng các ảnh gallery minh họa trong `artifacts/`.

Các màn hình và chức năng:

- **Tìm kiếm chân dung:** ảnh mẫu hoặc upload JPG/PNG/WEBP; xem căn chỉnh, ROI,
  thời gian trích/so khớp, Top-5, điểm từng nhóm và biểu đồ đóng góp.
- **Vector đặc trưng:** xem từng thuộc tính thô và giá trị dùng so khớp.
- **Khám phá gallery:** tìm tên/ID ảnh, phân trang, chuyển ảnh gốc sang overlay ROI.
- **Phương pháp & kiểm tra:** số chiều, trọng số, cách trình diễn và kết quả verify.
- **Tải kết quả:** JSON chứa vector và thông tin truy vấn; CSV chứa Top-5 và S1–S5.
- Có bộ lọc loại ảnh trùng truy vấn, xử lý lỗi không mặt/nhiều mặt/ROI không hợp lệ.

Kịch bản demo khoảng 3 phút:

1. Chọn ảnh mẫu, giữ nguyên bộ lọc rồi bấm **Tìm 5 ảnh tương đồng**.
2. Giới thiệu ảnh căn chỉnh, vùng ROI và ảnh tự trùng có điểm gần 1.
3. Bật **Loại ảnh trùng với truy vấn**, bấm tìm lại để xem các chân dung khác.
4. Chọn một kết quả, giải thích bảng điểm và đóng góp năm nhóm; tải JSON/CSV.
5. Tải một ảnh khác để thử; mở gallery để minh họa ROI đã trích.

Upload tối đa 10 MB và 16 triệu pixel. File upload lưu trong
`artifacts/demo/uploads`; kết quả từng lần tìm lưu riêng trong `artifacts/demo/runs`.
Các file này không được thêm vào gallery. Truy vấn ghi lịch sử vào DB.
Đổi ảnh hoặc bộ lọc sẽ ẩn kết quả cũ cho tới khi bấm tìm lại.

## Chạy kiểm kê

Chạy từ thư mục gốc bằng Python có Pillow:

```powershell
python -m portrait_search.inventory
```

Kết quả trong `artifacts/inventory/`:

- `manifest.csv`: ID theo đường dẫn đã sắp xếp, nhóm tên file, SHA-256 file,
  SHA-256 pixel RGB đã giải mã, kích thước, mode và trạng thái đọc.
- `summary.json`: thống kê và danh sách nhóm ảnh trùng.
- `review.csv`: 80 ảnh phân tầng với seed 42 và cột để chấm thủ công.
- `review_01.jpg`, `review_02.jpg`: bảng ảnh đối chiếu ID trong manifest.

Chạy lại sẽ ghi đè đầu ra cùng tên, bao gồm các cột chấm trong `review.csv`;
dùng `--output artifacts/inventory-new` để giữ lần chấm trước.
Hash pixel kiểm tra trùng chính xác sau giải mã, không phát hiện ảnh gần giống.
Không xóa ảnh trùng khỏi dữ liệu gốc. ID này thuộc lần kiểm kê hiện tại;
khi thêm/bớt file cần quản lý ID bền vững trong CSDL ở giai đoạn sau.

## Kết quả kiểm kê ban đầu

- 1.249/1.249 file đọc được; JPEG, RGB, 178 × 218.
- 563 tên file thuộc nhóm `celeba`, 686 thuộc `male_faces_strict`.
- 103 nhóm trùng chính xác ở cả hash file và hash pixel; 1.146 nội dung riêng biệt.
- Tập rà soát: 36 ảnh `celeba`, 44 ảnh `male_faces_strict`.

Đây là kết quả đọc file, chưa phải số ảnh khuôn mặt hợp lệ để truy hồi.
Tập rà soát mới phân tầng theo tên file; các thuộc tính tư thế, kính, râu,
biểu cảm và ánh sáng cần chấm để xác nhận độ bao phủ.

## Chạy landmark và căn chỉnh

Model chính thức đã tải tại `models/face_landmarker.task` (không đưa vào Git).
Để tải lại:

```powershell
New-Item -ItemType Directory -Force models
curl.exe -L --fail https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -o models/face_landmarker.task
python -m portrait_search.preprocess
```

Đầu ra ở `artifacts/preprocess/`: landmark gốc `.npy`, ảnh căn chỉnh, mask pixel
hợp lệ, overlay contour, ma trận similarity và landmark căn chỉnh `.npz`.
`metadata.json` lưu hash model, phiên bản thư viện và ngưỡng detection/presence.
`summary.json` ghi trạng thái từng ảnh. Lần chạy đầu: 80/80 vượt kiểm tra số mặt,
roll và kích thước hình học. Chưa có kiểm tra ROI nên chưa kết luận ảnh hợp lệ
cho đủ F1–F5. Đã xem trực quan một bảng ghép 20 overlay đầu.

Môi trường hiện có chạy được hai lệnh trên. Việc tạo `.venv` ban đầu thất bại
ở `ensurepip`; chưa xác minh cài mới toàn bộ `requirements.txt` trong môi trường
sạch. Streamlit đã được cài riêng trong `.demo-deps`; `run_demo.py` nạp thư mục này
để không thay đổi các thư viện trích đặc trưng của gallery.

## Trích đặc trưng và xây DB

```powershell
python -m portrait_search.build_gallery
python -m portrait_search.verify
python -m unittest discover -s tests -v
```

DB: `artifacts/portraits.sqlite3`. Chạy lại cùng dữ liệu/cấu hình bỏ qua ảnh đã
trích, giữ nguyên ID và snapshot. Nếu file đã có trong DB thay đổi nội dung,
chương trình báo lỗi; dùng DB mới để bảo toàn dữ liệu thực nghiệm cũ.
Thêm file mới được cấp ID mới. Không dùng `inventory.image_id` để nối với DB
sau khi thay đổi dataset; nối bằng đường dẫn.

Các bảng:

| Bảng | Ý nghĩa |
| --- | --- |
| `images` | Toàn bộ 1.249 đường dẫn, hash, trạng thái, `duplicate_of` trỏ ảnh đại diện |
| `feature_sets` | Công thức, tên/thứ tự chiều, ROI, hash model/code, thư viện và bản kết nối mesh thực tế |
| `image_features` | F1–F5, F4 dùng so khớp, landmark gốc, ma trận, metadata và đường dẫn overlay |
| `normalization_params` | Snapshot raw hoặc Z-score, danh sách gallery, mean/std và cờ chiều hằng |
| `search_runs` | Truy vấn và thời gian trích/so khớp |
| `search_results` | Top-5 cùng điểm S1–S5 và điểm tổng |

Mỗi nhóm vector được lưu thành BLOB float32 little-endian; số chiều ở
`dimensions_json`. Bản `raw` giữ vector thô; bản `zscore` giữ F1–F3 chuẩn hóa,
F4/F5 thô cùng `f4_match` đã cân bằng ba khối. Landmark và ma trận lưu float64
little-endian với shape lần lượt `(468,2)` và `(2,3)`. Phép tính dùng float64.
Khóa đặc trưng là `(image_id, feature_set_id, normalization_id)`.
Ảnh lỗi có trạng thái và nguyên nhân, không tạo vector giả bằng số 0.

Ảnh đại diện là đường dẫn đầu tiên theo thứ tự tăng dần trong mỗi nhóm hash
pixel. Các file gốc vẫn giữ nguyên. Việc fit chuẩn hóa chỉ dùng ảnh đại diện
có đủ năm nhóm, loại mọi bản trùng. Snapshot ready được commit nguyên tử;
truy vấn không trộn các snapshot và không thêm ảnh truy vấn vào gallery.

Đầu ra dễ đọc nằm ở `artifacts/gallery/<16 ký tự đầu feature_set_id>/`:

- `features.csv`: mỗi hàng là ID, đường dẫn và 119 giá trị thô có tên cột.
- `image_status.csv`: toàn bộ ảnh cùng trạng thái, ảnh đại diện và nguyên nhân lỗi.
- `summary.json`: số lượng, phiên bản và snapshot.
- `<image_id>/features.npz`, `aligned.png`, `roi.png`: dữ liệu và ảnh minh họa.

## Tìm kiếm

```powershell
python -m portrait_search.search data_best/data_final_img_align_celeba_000055.jpg
python -m portrait_search.search data_best/data_final_male_faces_strict_000055.jpg --exclude-duplicates --output artifacts/query_exclude_duplicates
```

Mặc định dùng snapshot ready mới nhất và giữ ảnh tự trùng để kiểm tra pipeline.
Có thể chỉ định snapshot bằng `--snapshot <normalization_id>`. `--exclude-duplicates`
loại cả hash file và hash pixel của truy vấn. Điểm là cosine có trọng số,
không phải xác suất cùng người. Hòa điểm sắp theo ID tăng dần.
`artifacts/query_example/` chứa `result.json` (vector thô/so khớp, thời gian,
trọng số và điểm từng nhóm), `top5.csv`, ảnh gốc, căn chỉnh và overlay ROI.

## Kết quả kiểm tra baseline

- Trên 80 ảnh thử: đủ 7/10/10/42/50 chiều, đã xem bốn bảng overlay ROI.
- Trên gallery: 1.146/1.146 ảnh đại diện trích thành công, không có vector zero-norm
  sau Z-score, vượt yêu cầu ít nhất 1.000 ảnh hợp lệ theo kiểm tra tự động hiện tại.
- Toàn bộ histogram và vector đọc lại từ DB đạt kiểm tra; SQLite integrity/khóa ngoại đạt.
- Cả 1.146 ảnh tự so thuộc nhóm đồng hạng cao nhất, sai số điểm tự so tối đa
  khoảng `2.22e-16`. Truy vấn trích lại ảnh `000055` có điểm tự so gần 1.
- Truy vấn loại trùng bỏ được ảnh đại diện dù dùng bản sao tên khác.
- Ảnh xám không mặt và ảnh ghép hai mặt trả đúng `no_face`/`multiple_faces`.
- Chạy lại build giữ nguyên snapshot và không tăng số bản ghi đặc trưng.

Báo cáo kiểm tra: `artifacts/verification.json`, `artifacts/detection_checks.json`.
Chưa có nhãn tương đồng/danh tính độc lập nên chưa báo Precision/Recall/nDCG.
ROI trán có thể chạm tóc; F4 chỉ nhận pixel trán qua lọc Lab khi đạt điều kiện,
F5 giữ texture trong ROI theo đặc tả. Việc phát hiện thành công không tự xác nhận
chất lượng khuôn mặt hoặc chất lượng truy hồi. Việc chấm thuộc tính tư thế/kính/
râu/ánh sáng trong CSV vẫn chưa được hoàn tất.

## Bước tiếp theo

Tập đánh giá độc lập; hoàn thiện báo cáo từ số liệu đã chạy.

Tài liệu API: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker?tab=Python

LBP uniform: https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.local_binary_pattern
