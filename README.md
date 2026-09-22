# Phát hiện tin tuyển dụng có dấu hiệu lừa đảo

**Bài 2 - Dữ liệu văn bản | Môn Học máy | Học kỳ I, 2026-2027**  
Khoa Khoa học và Kỹ thuật Máy tính, Trường Đại học Bách Khoa, ĐHQG-HCM.

Dự án so sánh BoW/TF-IDF với embedding GloVe huấn luyện sẵn để phân loại tin tuyển dụng tiếng Anh trong EMSCAD. Có EDA, 26 cấu hình truyền thống, kiểm soát nhóm trùng nội dung, ngưỡng cảnh báo chọn trên validation và demo giải thích bằng mô hình tuyến tính. Phần mở rộng TextCNN học embedding end-to-end được dùng để so sánh pipeline deep learning với mô hình truyền thống.

## Bắt đầu

### Google Colab

1. Mở `notebooks/Job_Scam_Detection.ipynb` bằng Colab: **File → Upload notebook**.
2. Chọn **Runtime → Run all**. GPU không bắt buộc nhưng giúp TextCNN chạy nhanh hơn.
3. Xem bảng kết quả và nhập một tin tuyển dụng tiếng Anh vào demo cuối notebook.
4. ZIP của lần chạy nằm trong `job_scam_project/output/` ở thanh Files.

Notebook mang theo bản module nguồn có SHA-256 và tự tải dữ liệu/GloVe từ nguồn công khai. Không cần mount Drive, tài khoản Kaggle, khóa API hay upload module riêng. Mạng Internet cần hoạt động. Nếu nguồn công khai gián đoạn, code báo lỗi và không thay bằng dữ liệu giả.

Thời gian phụ thuộc CPU và tốc độ mạng; xem `results/run_summary.json` của lần chạy thực tế. Bộ nguồn tải gồm khoảng 69 MB vector GloVe nén và CSV tuyển dụng; RAM sử dụng chủ yếu cho từ điển GloVe và ma trận thưa. Không chuyển toàn bộ TF-IDF thành ma trận dense.

### Chạy local

Python 3.12+ được khuyến nghị. Các thư viện tương thích được khai báo trong `requirements.txt`; phiên bản chính xác của môi trường kiểm tra được lưu trong `requirements-lock.txt` và `results/environment.json`.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_experiments.py
python -m pip install -r requirements-deep-learning.txt
python run_deep_learning.py
python -m unittest discover -s tests -v
python tools/build_notebook.py
python tools/package.py
```

Mỗi lần `run_experiments.py` chạy lại huấn luyện và ghi đè kết quả trong `results/`, `features/`, `models/`. Tải dữ liệu và đọc GloVe có cache. Dùng `--root <thư_mục_khác>` để giữ kết quả cũ.

## Thiết kế thí nghiệm

| Thành phần | Cách làm |
|---|---|
| Dữ liệu | EMSCAD, 17.880 tin, nhãn 0 hợp lệ / 1 gian lận |
| Văn bản đầu vào | title, company_profile, description, requirements, benefits |
| Làm sạch | HTML decode, bỏ thẻ, chuẩn hóa khoảng trắng/chữ thường, thay URL/email bằng token; giữ từ phủ định |
| Trùng lặp | Loại bản sao văn bản chính xác và nhóm văn bản có nhãn mâu thuẫn |
| Chia nhóm | Description hoặc company_profile chuẩn hóa trùng nhau, dài tối thiểu 80 ký tự; nối nhóm bắc cầu |
| Chia tập | StratifiedGroupKFold cố định seed; mục tiêu xấp xỉ 60/20/20, xem tỷ lệ thực tế trong audit |
| BoW | Unigram + MultinomialNB, alpha 0.1/1.0 |
| TF-IDF | Unigram/bigram, có/không English stopwords; LR và Linear SVM, C 0.5/2.0 |
| GloVe | Frozen 50 chiều; mean pooling / train-IDF pooling; LR và Linear SVM, C 0.1/1/10 |
| Deep learning | TextCNN với embedding học được, convolution kernel 3/4/5, global max pooling và dropout |
| Lựa chọn | F1 gian lận trên validation, ngưỡng chọn trên validation; tie-break AP rồi thứ tự cấu hình |
| Test | Chỉ đánh giá các cấu hình đã khóa; không chọn lại mô hình sau khi thấy test |
| Demo | TF-IDF + LR tốt nhất trên validation; có thể khác mô hình tốt nhất tổng thể |

Không dùng `fraudulent`, `job_id`, kết quả dự đoán hoặc thông tin từ nhãn làm đặc trưng. Từ điển BoW/TF-IDF, IDF và scaler chỉ fit bằng train. GloVe pretrained được cố định; việc đọc embedding không huấn luyện trên test. Không SMOTE hay cân bằng lại tập test.

Trong lần chạy hiện tại, TextCNN chọn epoch 6 và ngưỡng 0,6019 trên validation. Trên test khóa, mô hình đạt F1 gian lận 0,4578 và AP 0,4890, thấp hơn TF-IDF + Linear SVM (F1 0,5507; AP 0,5348). Kết quả này được giữ nguyên để báo cáo so sánh trung thực, không chọn lại theo test.

Các cột có/không logo và câu hỏi không được sử dụng trong phiên bản text-only này. Bản trùng nội dung có thể gây kết quả quá lạc quan, vì vậy nhóm trùng được tách biệt giữa ba tập. Phương pháp này **không** bảo đảm tách mọi công ty thật hoặc phát hiện mọi tin viết lại.

## Kết quả được lưu ở đâu?

- `results/data_manifest.json`: nguồn, checksum và số lượng nhãn dữ liệu gốc.
- `results/data_audit.json`: số dòng loại bỏ, nhóm và tỷ lệ chia tập thực tế.
- `results/validation_scores.csv`: toàn bộ cấu hình và thời gian fit, cảnh báo hội tụ nếu có.
- `results/selection.json`: lựa chọn được khóa trước test.
- `results/test_scores.csv`: accuracy, precision/recall/F1 gian lận, average precision, ROC-AUC, TP/FP/FN/TN.
- `results/test_predictions.csv`: dự đoán từng tin để tính lại chỉ số độc lập.
- `results/deep_learning_scores.csv`: kết quả validation/test của TextCNN với ngưỡng chọn trên validation.
- `results/deep_learning_history.csv`: loss và chỉ số validation theo epoch của TextCNN.
- `results/errors/`: tin bị bỏ sót/cảnh báo nhầm, dùng để phân tích sau đánh giá.
- `results/figures/`: EDA, precision-recall, confusion matrix và so sánh mô hình.
- `features/`: embedding `.npy`, ma trận thưa `.npz`, nhãn/ID/split và checksum.
- `models/`: mô hình local do dự án tạo. Chỉ load các file joblib tin cậy của chính lần chạy.

Chỉ số chính là F1 của lớp gian lận và average precision. Accuracy của mô hình luôn đoán hợp lệ đã rất cao do mất cân bằng nhãn. AP được tính bằng `average_precision_score`, không đồng nhất với PR-AUC tích phân hình thang.

## Cấu trúc

```text
notebooks/    Notebook nguồn và bản đã thực thi khi có
modules/      Xử lý dữ liệu, EDA, embedding, huấn luyện, đánh giá, demo
features/     File đặc trưng và manifest thứ tự dòng
models/       Các mô hình đã fit
results/      Số liệu thật, biểu đồ, dự đoán và audit
reports/      Báo cáo PDF và ghi chú bàn giao
tests/        Kiểm tra leakage, ngưỡng, file đặc trưng và tính lại metrics
tools/        Tạo notebook, kiểm tra và đóng gói
data/         Dữ liệu tải và cache local; không bắt buộc đưa vào ZIP
```

## Phạm vi và giới hạn

1. Dữ liệu lịch sử tiếng Anh (2012-2014); chưa đo khả năng dùng cho tiếng Việt hoặc lừa đảo hiện nay.
2. Nhãn dữ liệu có thể có sai sót; dự đoán không xác minh danh tính hay kết luận một tổ chức phạm pháp.
3. GloVe mean pooling mất thứ tự từ/ngữ cảnh, tin dài bị cắt ở 1.500 token (có thống kê).
4. Điều chỉnh tham số và ngưỡng trên cùng validation; không phải nested cross-validation.
5. Điểm logistic dùng trong demo chưa được hiệu chuẩn; không diễn giải là xác suất gian lận ngoài thực tế.
6. Cụm từ giải thích là đóng góp toán học trong mô hình, không phải bằng chứng hoặc quan hệ nhân quả.
7. TextCNN là phần mở rộng deep learning; cần chạy trên Colab và dùng kết quả thực tế trước khi đưa chỉ số vào báo cáo.

## Thông tin học phần và cá nhân

- Tên môn: Học máy; mã môn: chưa được cung cấp.
- Học kỳ I, năm học 2026-2027.
- GVHD: TS. Trương Vĩnh Lân; lớp A01; nhóm 01.
- Người thực hiện: 1 người theo yêu cầu; họ tên, MSSV, email chưa được cung cấp.
- Phạm vi công việc dự kiến của người thực hiện: toàn bộ bài tập. Không giả lập thành viên hay minh chứng họp nhóm.
- Đề gốc quy định nhóm 2-3 thành viên; cần thống nhất việc nộp cá nhân với GVHD.
- GitHub/Colab công khai: chưa tạo liên kết. Notebook local có thể upload trực tiếp vào Colab.
- Báo cáo PDF: `reports/job_scam_report.pdf`; nguồn làm việc trên [Overleaf](https://www.overleaf.com/project/6aaa50614095f851ad080bca).
- Gói vẫn cần bổ sung họ tên, MSSV, email, mã môn và xác nhận nộp cá nhân trước khi nộp chính thức.

## Nguồn

1. Vidros et al. (2017), *Automatic Detection of Online Recruitment Frauds*: https://doi.org/10.3390/fi9010006.
2. Dataset mirror: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction.
3. Pennington et al. (2014), *GloVe*: https://nlp.stanford.edu/projects/glove/.
4. Vector distribution: https://github.com/piskvorky/gensim-data (tham khảo license riêng của vector trong metadata nguồn).
5. Scikit-learn evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html.

Code được hỗ trợ xây dựng bằng AI; người nộp cần đọc, chạy lại, hiểu và giải thích được lựa chọn phương pháp, kết quả và giới hạn, đồng thời tuân thủ quy định sử dụng AI của môn học nếu có.
