"""Generate a standalone, source-bundled notebook with no personal cloud dependency."""
from pathlib import Path
import base64
import hashlib
import io
import textwrap
import zipfile
import nbformat as nbf

def build_notebook(root, destination=None):
    root=Path(root).resolve()
    archive_bytes=io.BytesIO()
    with zipfile.ZipFile(archive_bytes,'w',compression=zipfile.ZIP_DEFLATED) as z:
        paths=list((root/'modules').glob('*.py'))+list((root/'tests').glob('*.py'))
        paths += [root/p for p in ['requirements.txt','requirements-deep-learning.txt','README.md',
                                  'run_experiments.py','run_deep_learning.py','tools/build_notebook.py',
                                  'tools/package.py','features/README.md','reports/README.md',
                                  'reports/job_scam_report.pdf']]
        for path in sorted(paths):
            if path.exists():
                info=zipfile.ZipInfo(path.relative_to(root).as_posix(),date_time=(2026,1,1,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED
                z.writestr(info,path.read_bytes())
    payload=base64.b64encode(archive_bytes.getvalue()).decode()
    checksum=hashlib.sha256(archive_bytes.getvalue()).hexdigest()
    cells=[]
    def md(text): cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(text): cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))
    md('''
    # Phát hiện tin tuyển dụng có dấu hiệu lừa đảo
    **Bài 2 - Học máy với dữ liệu văn bản | Học kỳ I, 2026-2027**

    **Phạm vi:** tin tuyển dụng tiếng Anh trong EMSCAD (2012-2014). Phân loại gian lận/hợp lệ
    nhằm nghiên cứu khả năng sàng lọc, không xác minh danh tính hoặc kết luận pháp lý.

    ## Cách chạy
    1. Mở notebook trong Google Colab (File → Upload notebook).
    2. Chọn **Runtime → Run all**. CPU đủ cho cấu hình mặc định; không bắt buộc GPU.
    3. Đợi các thí nghiệm hoàn tất, đọc bảng kết quả và dùng ô demo ở cuối.

    Notebook tự chuẩn bị module, tải dữ liệu công khai và vector GloVe. **Không mount Google Drive,
    không cần khóa API hoặc đăng nhập Kaggle.** Các module nguồn đi kèm giúp notebook chạy độc lập.
    Mọi điểm số hiển thị đều được tính trong lần chạy này.

    **Thông tin nộp:** GVHD TS. Trương Vĩnh Lân, lớp A01, nhóm 01. Họ tên, MSSV, email và mã môn
    chưa được cung cấp. Người thực hiện dự kiến làm cá nhân; đề gốc quy định nhóm 2-3 người nên cần
    thống nhất việc nộp cá nhân với GVHD. Báo cáo nằm tại `reports/job_scam_report.pdf`.
    ''')
    md('''
    ## 1. Chuẩn bị môi trường
    Module nguồn trong ô sau là bản đóng gói từ thư mục `modules/`, có kiểm tra SHA-256.
    Kết quả ghi vào `/content/job_scam_project` trên Colab hoặc `job_scam_run` khi chạy local.
    Có thể đặt biến môi trường `EMSCAD_WORKDIR` để đổi thư mục chạy local.
    ''')
    bootstrap=f'''
import base64, hashlib, io, os, sys, subprocess, zipfile, importlib.metadata
from pathlib import Path
IS_COLAB = 'google.colab' in sys.modules or bool(os.environ.get('COLAB_RELEASE_TAG'))
WORK = Path(os.environ.get('EMSCAD_WORKDIR', '/content/job_scam_project' if IS_COLAB else str(Path.cwd() / 'job_scam_run'))).resolve()
WORK.mkdir(parents=True, exist_ok=True)
PAYLOAD = {payload!r}
SOURCE_SHA256 = {checksum!r}
blob = base64.b64decode(PAYLOAD)
assert hashlib.sha256(blob).hexdigest() == SOURCE_SHA256, 'Source bundle checksum mismatch'
with zipfile.ZipFile(io.BytesIO(blob)) as source:
    for info in source.infolist():
        target = (WORK / info.filename).resolve()
        if not target.is_relative_to(WORK):
            raise ValueError('Unsafe archive path')
        target.parent.mkdir(parents=True, exist_ok=True)
        content = source.read(info)
        if target.exists() and target.read_bytes() != content:
            raise RuntimeError('Existing source differs: ' + str(target) + '. Use a new EMSCAD_WORKDIR to preserve edits.')
        target.write_bytes(content)
requirements = {{'numpy':'numpy>=1.26,<3','pandas':'pandas>=2.2,<4','scipy':'scipy>=1.11,<2',
 'scikit-learn':'scikit-learn>=1.4,<2','matplotlib':'matplotlib>=3.8,<4','joblib':'joblib>=1.3,<2',
 'threadpoolctl':'threadpoolctl>=3.2,<4','nbformat':'nbformat>=5.9,<6','ipywidgets':'ipywidgets>=8.1,<9'}}
from packaging.requirements import Requirement
needed=[]
for package,spec in requirements.items():
    try:
        version=importlib.metadata.version(package)
        if version not in Requirement(spec).specifier:
            needed.append(spec)
    except importlib.metadata.PackageNotFoundError:
        needed.append(spec)
if needed:
    subprocess.check_call([sys.executable,'-m','pip','install','--quiet',*needed])
if IS_COLAB:
    try:
        importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
        subprocess.check_call([sys.executable,'-m','pip','install','--quiet','torch>=2.2,<3'])
sys.path.insert(0,str(WORK))
os.chdir(WORK)
print('Workspace:', WORK)
print('Python:', sys.version.split()[0], '| Bundled source:', SOURCE_SHA256[:16])
'''
    code(bootstrap)
    cells[-1].metadata.update({'cellView':'form'})
    md('''
    ## 2. Cấu hình và quy trình đánh giá
    - Nhãn `1`: tin gian lận; nhãn `0`: tin hợp lệ.
    - Gộp title, company_profile, description, requirements, benefits. Không dùng nhãn/ID làm đặc trưng.
    - Loại bản sao văn bản chính xác; loại các nhóm văn bản có nhãn mâu thuẫn.
    - Nhóm các tin có description hoặc company_profile chuẩn hóa trùng nhau (tối thiểu 80 ký tự).
    - Chia theo nhóm, mục tiêu xấp xỉ 60% train / 20% validation / 20% test; tỷ lệ thực tế được in ra.
    - Chỉ train được dùng để fit vectorizer, scaling và IDF pooling. Validation chọn mô hình và ngưỡng.
    - Khóa các lựa chọn trước khi xem test. Không cân bằng lại test và không chọn mô hình theo điểm test.

    **Giới hạn:** nhóm này không tương đương mã công ty thật và không phát hiện hết các tin viết lại.
    ''')
    code('''
    from modules.config import Config
    from modules.experiment import prepare, extract_features, train_candidates, evaluate_test
    from IPython.display import display, Image
    import pandas as pd
    config = Config(root=str(WORK), seed=42, max_features=40000, min_df=2, max_tokens=1500)
    ctx = prepare(config)
    display(pd.crosstab(ctx['df']['split'], ctx['df']['fraudulent']))
    ''')
    md('''
    ## 3. EDA
    Cấu trúc, tỷ lệ nhãn và số ô trống được kiểm tra trên dữ liệu gốc. Phân tích từ vựng và độ dài
    dùng train. Trường trống được thay bằng chuỗi rỗng; không tự coi thông tin không khai báo là gian lận.
    ''')
    code('''
    display(Image(filename=str(WORK/'results/figures/01_data_overview.png')))
    display(Image(filename=str(WORK/'results/figures/02_lengths_splits.png')))
    display(pd.read_csv(WORK/'results/top_words_train.csv').groupby('label').head(12))
    print('Accuracy nếu luôn đoán hợp lệ trên dữ liệu gốc:', round(ctx['eda']['raw_majority_accuracy'],4))
    ''')
    md('''
    ## 4. Trích xuất đặc trưng và lưu file
    **Truyền thống:** BoW; TF-IDF unigram; TF-IDF unigram + bigram; cấu hình có/không loại stopwords.

    **Embedding hiện đại theo yêu cầu đề:** GloVe pretrained 50 chiều, trung bình token và trung bình
    có trọng số IDF học từ train. Vector cuối được chuẩn hóa L2. Không huấn luyện lại GloVe.
    Giới hạn 1.500 token/tin; thống kê số tin bị cắt và độ phủ từ vựng được lưu để phân tích.

    File `.npy` chứa embedding; `.npz` chứa ma trận BoW/TF-IDF thưa. `job_ids.npy`, `labels.npy`,
    `splits.npy` và `manifest.json` bảo đảm truy vết thứ tự dòng. GloVe pooling mất thứ tự từ,
    không tương đương BERT và không tự được tính là phần mở rộng fine-tuning học sâu.
    ''')
    code('''
    timing = extract_features(ctx)
    display(pd.DataFrame(timing).T)
    import json
    display(json.loads((WORK/'results/embedding_audit.json').read_text(encoding='utf-8')))
    ''')
    md('''
    ## 5. Huấn luyện và lựa chọn trên validation
    26 cấu hình: BoW + MultinomialNB, TF-IDF + Logistic Regression/Linear SVM,
    GloVe + Logistic Regression/Linear SVM. SVM và Logistic Regression dùng class_weight=balanced.

    Ngưỡng được chọn theo F1 của lớp gian lận trên validation. Nếu bằng F1, chọn ngưỡng lớn hơn.
    Mỗi nhóm giữ một cấu hình tốt nhất; chọn mô hình tổng thể trên validation trước khi đánh giá test.
    Quy trình không phải nested cross-validation. Thời gian fit không bao gồm tải dữ liệu/trích xuất đặc trưng.
    ''')
    code('''
    validation = train_candidates(ctx)
    display(validation[['candidate','family','representation','params','f1_fraud','precision_fraud','recall_fraud','average_precision','threshold','fit_seconds','warnings']])
    print('Mô hình chọn trước khi xem test:', ctx['selected'])
    ''')
    md('''
    ## 6. Đánh giá test giữ riêng
    Accuracy phải đọc cùng precision, recall, F1 của lớp gian lận và average precision (AP).
    AP là tóm tắt đường precision-recall, không phải diện tích hình thang.
    - False positive: tin hợp lệ bị cảnh báo nhầm.
    - False negative: tin gian lận bị bỏ sót.
    Các dòng đánh giá dưới đây đều là cấu hình đã chốt bằng validation. Không đổi lựa chọn sau bảng này.
    ''')
    code('''
    test_results = evaluate_test(ctx)
    display(test_results)
    display(Image(filename=str(WORK/'results/figures/03_precision_recall.png')))
    display(Image(filename=str(WORK/'results/figures/04_model_comparison.png')))
    selected_family = ctx['candidates'][ctx['selected']]['record']['family']
    display(Image(filename=str(WORK/f'results/figures/cm_{selected_family}.png')))
    ''')
    md('''
    ## 7. Pipeline deep learning mở rộng: TextCNN

    TextCNN học embedding và các bộ lọc tích chập trực tiếp từ văn bản train. Mô hình dùng convolution
    với kernel 3/4/5, global max pooling và dropout. Epoch tốt nhất cùng ngưỡng cảnh báo chỉ được chọn
    trên validation; test được đọc đúng một lần sau khi khóa lựa chọn. Đây là phần mở rộng deep learning
    thật sự để so sánh với pipeline truyền thống, không phải GloVe pooling cố định.

    GPU giúp chạy nhanh hơn nhưng Colab CPU vẫn có thể chạy. Nếu chạy notebook local trên Python chưa
    hỗ trợ PyTorch, phần này được bỏ qua có thông báo; trên Google Colab, PyTorch được cài tự động nếu thiếu.
    ''')
    code('''
    import importlib.util, subprocess, sys
    if importlib.util.find_spec('torch') is None and IS_COLAB:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', 'torch>=2.2,<3'])
    if importlib.util.find_spec('torch') is None:
        deep_results = None
        print('Bỏ qua TextCNN ở môi trường local vì chưa có PyTorch. Hãy chạy notebook trên Colab.')
    else:
        from modules.deep_learning import run_textcnn
        deep_results = run_textcnn(ctx, epochs=6, batch_size=128)
        display(deep_results)
        display(Image(filename=str(WORK/'results/figures/05_textcnn_training.png')))
        display(Image(filename=str(WORK/'results/figures/cm_textcnn.png')))
        traditional = test_results[['family','f1_fraud','average_precision']].copy()
        deep_test = deep_results.loc[deep_results['split']=='test', ['family','f1_fraud','average_precision']]
        display(pd.concat([traditional, deep_test], ignore_index=True).sort_values('f1_fraud', ascending=False))
    ''')
    md('''
    ## 8. Phân tích lỗi
    Xem tin bị bỏ sót và cảnh báo nhầm để nhận xét giới hạn. Không dùng các lỗi test này để
    sửa mô hình rồi báo lại cùng test như một kết quả độc lập.
    ''')
    code('''
    errors = pd.read_csv(WORK/f'results/errors/{selected_family}.csv')
    display(errors[['job_id','error_type','title','score','text_excerpt']].groupby('error_type').head(3))
    ''')
    md('''
    ## 9. Demo giải thích dự đoán
    Demo dùng **TF-IDF + Logistic Regression tốt nhất trên validation**, có thể khác mô hình thắng
    tổng thể. Điểm mô hình chưa được hiệu chuẩn thành xác suất thực tế.
    Cụm từ hiển thị là đóng góp vào log-odds của mô hình tuyến tính, không phải nguyên nhân hay bằng chứng gian lận.
    Tin tiếng Việt và các cách lừa đảo hiện nay nằm ngoài phạm vi đã kiểm chứng.
    ''')
    code('''
    from modules.inference import predict_text, notebook_demo
    example = 'We are looking for a software engineer to develop Python applications, review code and collaborate with our engineering team. Applicants should have experience with databases and automated testing.'
    result, towards_fraud, towards_legitimate = predict_text(example, WORK)
    display(result)
    display(towards_fraud)
    print('Đây là ví dụ minh họa tự viết, không có nhãn thật và không tham gia đánh giá.')
    demo_widgets = notebook_demo(WORK)
    ''')
    md('''
    ## 10. Kiểm tra và đóng gói
    Kiểm tra tính tách biệt nhóm, thứ tự embedding, SHA-256, cách chọn mô hình và tính lại chỉ số từ dự đoán.
    ZIP gồm notebook nguồn, modules, features, models, results và báo cáo PDF trong `reports/`.
    Dữ liệu thô và GloVe tự tải lại, không đưa vào ZIP.
    ''')
    code('''
    import unittest
    suite = unittest.defaultTestLoader.discover(str(WORK/'tests'), pattern='test_*.py')
    test_run = unittest.TextTestRunner(verbosity=2).run(suite)
    assert test_run.wasSuccessful(), 'Kiểm tra pipeline thất bại'
    from tools.build_notebook import build_notebook
    from tools.package import package
    build_notebook(WORK)
    archive_path = package(WORK)
    print('Gói dự án:', archive_path)
    print('Báo cáo PDF:', WORK/'reports/job_scam_report.pdf')
    ''')
    md('''
    ### Tải file trên Colab (tùy chọn)
    Trong thanh Files bên trái, mở `job_scam_project/output`, chọn tải `job_scam_project.zip`.
    Notebook đang mở có thể lưu thêm bằng **File → Download → Download .ipynb** để giữ output lần chạy.

    ## Nguồn tham khảo
    1. Vidros et al. (2017). [EMSCAD và phát hiện gian lận tuyển dụng](https://doi.org/10.3390/fi9010006).
    2. [Bản dữ liệu công khai trên Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction).
    3. Pennington et al. (2014). [GloVe](https://nlp.stanford.edu/projects/glove/).
    4. [Gensim-data, nguồn phân phối vector GloVe](https://github.com/piskvorky/gensim-data).
    5. [Scikit-learn: đánh giá mô hình](https://scikit-learn.org/stable/modules/model_evaluation.html).

    ## Những việc còn lại trước khi nộp
    Điền họ tên/MSSV/email/mã môn; xác nhận được làm cá nhân và dataset chưa trùng;
    kiểm tra Run all trên Colab bằng tài khoản của người nộp; bổ sung liên kết
    GitHub/Colab khi đã tạo. Không có minh chứng họp nhóm hoặc lịch sử làm việc giả định.
    ''')
    notebook=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'},'colab':{'name':'Job_Scam_Detection.ipynb','provenance':[]}})
    destination=Path(destination or root/'notebooks'/'Job_Scam_Detection.ipynb')
    destination.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(notebook,destination)
    nbf.validate(notebook)
    return destination

if __name__=='__main__':
    print(build_notebook(Path(__file__).resolve().parents[1]))
