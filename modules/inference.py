"""Explain the fitted TF-IDF logistic model without presenting its score as certainty."""
from pathlib import Path
import html
import joblib
import numpy as np
import pandas as pd
from .data import clean_text

def predict_text(text, root='.'):
    if not isinstance(text,str) or len(text.strip())<20:
        raise ValueError('Nhập tin tuyển dụng tiếng Anh có ít nhất 20 ký tự.')
    if len(text)>100000:
        raise ValueError('Tin quá dài; giới hạn 100.000 ký tự cho demo.')
    bundle=joblib.load(Path(root)/'models'/'demo_tfidf.joblib')
    vectorizer,model=bundle['vectorizer'],bundle['model']
    X=vectorizer.transform([clean_text(text)])
    if X.nnz==0:
        raise ValueError('Không có từ nằm trong bộ từ vựng đã học. Hãy kiểm tra ngôn ngữ và nội dung.')
    score=float(model.predict_proba(X)[0,1])
    indices=X.indices
    contributions=X.data*model.coef_[0,indices]
    terms=vectorizer.get_feature_names_out()[indices]
    evidence=pd.DataFrame({'term':terms,'contribution_to_log_odds':contributions})
    top=evidence.sort_values('contribution_to_log_odds',ascending=False).head(8)
    bottom=evidence.sort_values('contribution_to_log_odds').head(8)
    result={'decision':'Cần kiểm tra thêm' if score>=bundle['threshold'] else 'Chưa vượt ngưỡng cảnh báo',
            'model_score':score,'threshold':float(bundle['threshold']),
            'model':'TF-IDF + Logistic Regression','candidate':int(bundle['candidate']),
            'matched_features':int(X.nnz),
            'note':'Điểm mô hình chưa được hiệu chuẩn. Cụm từ ảnh hưởng dự đoán không phải bằng chứng gian lận. Chỉ đánh giá trên tiếng Anh lịch sử.'}
    return result,top,bottom

def notebook_demo(root='.'):
    # Optional interactive layer. The plain function works without ipywidgets.
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    text=widgets.Textarea(placeholder='Paste an English job posting here...',
                          layout=widgets.Layout(width='100%',height='190px'))
    button=widgets.Button(description='Phân tích tin',button_style='primary')
    output=widgets.Output()
    def clicked(_):
        with output:
            clear_output()
            try:
                result,top,bottom=predict_text(text.value,root)
                display(HTML('<h3>'+html.escape(result['decision'])+'</h3>'))
                print(f"Điểm: {result['model_score']:.3f} | Ngưỡng validation: {result['threshold']:.3f}")
                print(result['note'])
                print('Cụm từ đẩy dự đoán về phía gian lận:')
                display(top[top.contribution_to_log_odds>0])
                print('Cụm từ đẩy dự đoán về phía hợp lệ:')
                display(bottom[bottom.contribution_to_log_odds<0])
            except ValueError as exc:
                print(str(exc))
    button.on_click(clicked)
    display(text,button,output)
    return text,button,output
