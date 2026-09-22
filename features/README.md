# File đặc trưng

Được tạo tự động sau bước `extract_features`:

- `glove_mean.npy`, `glove_idf.npy`: hai cách gộp vector GloVe 50 chiều, float32, có chuẩn hóa L2.
- `glove_coverage.npy`: tỷ lệ token có trong từ điển pretrained cho mỗi tin.
- `labels.npy`, `job_ids.npy`, `splits.npy`: nhãn, mã tin và phân vùng tương ứng từng dòng.
- `bow.npz`, `tfidf_*.npz`: ma trận thưa; không chuyển sang dense để tránh tốn RAM.
- `glove_pooling.json`: trọng số IDF học từ train, không học trên validation/test.
- `manifest.json`: SHA-256 của file và manifest chia dữ liệu, cấu hình, thời gian trích xuất.

Thứ tự dòng **phải** khớp `data/processed/split_manifest.csv`. Không dùng file embedding từ một lần chia dữ liệu khác với trọng số/manifest hiện tại. `np.load(..., allow_pickle=False)` dùng cho các file `.npy`.

GloVe là embedding từ huấn luyện sẵn được đề bài chấp nhận. Mean pooling không có khả năng biểu diễn ngữ cảnh như BERT; không gọi GloVe là Transformer hoặc tuyên bố đã fine-tune học sâu.
