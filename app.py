"""Vietnamese Streamlit demo for the existing portrait retrieval pipeline."""

import csv
import hashlib
import io
import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'artifacts/portraits.sqlite3'
GROUPS = ['F1 · Khuôn mặt', 'F2 · Mắt & mày', 'F3 · Mũi & miệng', 'F4 · Màu sắc', 'F5 · Texture']
WEIGHTS = [.25, .20, .25, .10, .20]

st.set_page_config(page_title='Portrait Lab · Tìm kiếm chân dung', page_icon='🔎', layout='wide')
st.markdown('''<style>
.stApp {background:#f4f7fb;color:#15263b}
[data-testid="stSidebar"] {background:#10273a}
[data-testid="stSidebar"] * {color:#eff7fb}
[data-testid="stSidebar"] [data-baseweb="select"] * {color:#15263b}
.block-container {padding-top:2rem;max-width:1440px}
h1,h2,h3 {letter-spacing:-.025em}
.hero {background:linear-gradient(115deg,#10273a,#164d61);border-radius:20px;padding:32px 36px;color:white;margin-bottom:24px}
.hero h1 {color:white;font-size:2.5rem;margin:5px 0 8px;padding:0}
.hero p {color:#d3e5ec;font-size:1.05rem;margin:0;max-width:780px}
.eyebrow {text-transform:uppercase;letter-spacing:.16em;font-size:.8rem;font-weight:600;color:#7de3d2}
[data-testid="stMetric"] {background:white;border:1px solid #dce5ed;border-radius:14px;padding:16px}
[data-testid="stMetricLabel"] {color:#566a7e}
.stButton button[kind="primary"] {background:#117c80;border:0;border-radius:9px;min-height:44px}
.stTabs [data-baseweb="tab-list"] {gap:24px;margin-bottom:14px}
.stTabs [aria-selected="true"] {color:#117c80}
.stTabs [data-baseweb="tab-highlight"] {background:#117c80}
.stDownloadButton button {border-radius:9px}
.stCaption {color:#516579}
</style>''', unsafe_allow_html=True)


def read_db(sql, params=()):
    with closing(sqlite3.connect(DB.resolve().as_uri()+'?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(sql, params)]


@st.cache_data(ttl=10)
def catalog():
    snapshots = read_db("""SELECT n.normalization_id,n.feature_set_id,n.created_at,f.version,
        (SELECT COUNT(*) FROM image_features x WHERE x.normalization_id=n.normalization_id
        AND x.status='ok') AS searchable FROM normalization_params n
        JOIN feature_sets f USING(feature_set_id) WHERE n.kind='zscore' AND n.status='ready'
        ORDER BY n.created_at DESC,n.rowid DESC""")
    counts = read_db('SELECT status,COUNT(*) AS count FROM images GROUP BY status')
    return snapshots, counts


def show_image(path, **kwargs):
    if Path(path).is_file():
        st.image(str(path), **kwargs)
    else:
        st.warning('Không tìm thấy ảnh minh họa.')


def result_table(results):
    return pd.DataFrame([{'Hạng': index, 'Ảnh': Path(r['path']).name, 'Điểm tổng': r['score'],
                          **{GROUPS[i]: r[f's{i+1}'] for i in range(5)}}
                         for index, r in enumerate(results, 1)])


if not DB.is_file():
    st.error('Chưa có cơ sở dữ liệu gallery. Chạy `python -m portrait_search.build_gallery` trước khi mở demo.')
    st.stop()
try:
    snapshots, counts = catalog()
except sqlite3.Error:
    st.error('Không đọc được cơ sở dữ liệu. Kiểm tra file artifacts/portraits.sqlite3.')
    st.stop()
if not snapshots:
    st.warning('Chưa có gallery đã chuẩn hóa. Hãy chạy bước xây dựng gallery trước.')
    st.stop()

with st.sidebar:
    st.markdown('## PORTRAIT LAB')
    st.caption('Hệ thống truy hồi ảnh chân dung')
    st.divider()
    snapshot_index = st.selectbox('Bộ dữ liệu tìm kiếm', range(len(snapshots)),
        format_func=lambda i: f"{snapshots[i]['searchable']:,} ảnh · {snapshots[i]['created_at']}")
    snapshot = snapshots[snapshot_index]
    st.markdown('### Đặc trưng sử dụng')
    for label, weight in zip(GROUPS, WEIGHTS):
        st.write(f'{label} — trọng số {weight:.2f}')
    st.divider()
    st.caption('Ảnh tải lên chỉ dùng để truy vấn, không được thêm vào gallery.')
    st.caption('Điểm thể hiện độ tương đồng ngoại hình, không phải xác suất cùng người.')

st.markdown('''<div class="hero"><div class="eyebrow">Tra cứu ảnh theo nội dung</div>
<h1>Tìm kiếm chân dung tương đồng</h1>
<p>Từ hình học khuôn mặt, màu sắc đến texture. Chọn một bức ảnh và khám phá
5 chân dung gần nhất trong bộ dữ liệu.</p></div>''', unsafe_allow_html=True)

total = sum(row['count'] for row in counts)
duplicates = sum(row['count'] for row in counts if row['status'] == 'duplicate')
for col, label, value in zip(st.columns(4), ['Ảnh trong dataset','Ảnh trong gallery','Bản trùng đã bỏ qua','Đặc trưng mỗi ảnh'],
                             [f'{total:,}',f"{snapshot['searchable']:,}",str(duplicates),'119 chiều']):
    col.metric(label,value)

search_tab, gallery_tab, about_tab = st.tabs(['Tìm kiếm chân dung','Khám phá gallery','Phương pháp & kiểm tra'])

with search_tab:
    input_col, preview_col = st.columns([1,1.65],gap='large')
    source_path, signature = None, None
    with input_col:
        st.subheader('Ảnh truy vấn')
        mode = st.radio('Nguồn ảnh', ['Ảnh mẫu','Tải ảnh lên'], horizontal=True)
        if mode == 'Ảnh mẫu':
            samples = read_db('''SELECT i.image_id,i.path FROM images i JOIN image_features f USING(image_id)
                WHERE f.normalization_id=? AND f.status='ok' AND i.duplicate_of IS NULL
                ORDER BY i.image_id LIMIT 16''',(snapshot['normalization_id'],))
            if samples:
                sample_index = st.selectbox('Chọn ảnh có sẵn',range(len(samples)),
                    format_func=lambda i:f"Ảnh {samples[i]['image_id']} · {Path(samples[i]['path']).name}")
                source_path = ROOT / samples[sample_index]['path']
        else:
            upload = st.file_uploader('Chọn ảnh JPG, PNG hoặc WEBP',type=['jpg','jpeg','png','webp'],
                                      help='Tối đa 10 MB. Ảnh rõ mặt, chỉ có một người, gần chính diện.')
            if upload is not None:
                content = upload.getvalue()
                try:
                    if len(content)>10*1024*1024:
                        raise ValueError('Ảnh vượt giới hạn 10 MB.')
                    with Image.open(io.BytesIO(content)) as img:
                        if img.width*img.height>16_000_000:
                            raise ValueError('Ảnh quá lớn. Hãy chọn ảnh dưới 16 triệu pixel.')
                        img.verify()
                    folder = ROOT/'artifacts/demo/uploads'
                    folder.mkdir(parents=True,exist_ok=True)
                    source_path = folder/(hashlib.sha256(content).hexdigest()+'.image')
                    if not source_path.exists():
                        source_path.write_bytes(content)
                except (ValueError,OSError,UnidentifiedImageError,Image.DecompressionBombError) as exc:
                    st.error(f'Không thể dùng ảnh này: {exc}')
        exclude = st.checkbox('Loại ảnh trùng với truy vấn',value=False,
            help='Bật khi đánh giá chất lượng. Tắt để kiểm tra khả năng tìm lại chính ảnh mẫu.')
        if source_path and source_path.is_file():
            signature = hashlib.sha256(source_path.read_bytes()).hexdigest()+snapshot['normalization_id']+str(exclude)
            show_image(source_path,caption='Ảnh đầu vào',width=220)
        submitted = st.button('Tìm 5 ảnh tương đồng',type='primary',disabled=signature is None,width='stretch')
        st.caption('Nên chọn ảnh rõ mắt, mũi, miệng; tránh che mặt hoặc nghiêng mạnh.')
        if submitted:
            st.session_state.pop('demo_result',None)
            try:
                with st.spinner('Đang căn chỉnh, trích đặc trưng và so khớp gallery…'):
                    from portrait_search.search import search
                    output = ROOT/'artifacts/demo/runs'/uuid.uuid4().hex
                    report = search(source_path,DB,snapshot['normalization_id'],exclude,output)
                    st.session_state['demo_result'] = dict(signature=signature,report=report,output=str(output))
            except Exception as exc:
                from portrait_search.features import FeatureError
                errors = {'no_face':'Không phát hiện khuôn mặt. Hãy chọn ảnh rõ mặt hơn.',
                    'multiple_faces':'Ảnh có nhiều khuôn mặt. Hãy chọn ảnh chỉ có một người.',
                    'invalid_geometry':'Khuôn mặt chưa đạt điều kiện hình học. Hãy chọn ảnh gần chính diện hơn.',
                    'invalid_roi':'Vùng lấy đặc trưng không đủ dữ liệu. Hãy chọn ảnh mặt rõ và ít che khuất.',
                    'zero_norm':'Không thể tính cosine cho ảnh này. Hãy thử một ảnh khác.'}
                if isinstance(exc,FeatureError):
                    st.error(errors.get(exc.status,str(exc)))
                    with st.expander('Chi tiết kiểm tra'):
                        st.code(str(exc))
                else:
                    st.error('Không thể hoàn tất tìm kiếm. Kiểm tra model, DB và phiên bản thư viện.')
                    with st.expander('Chi tiết lỗi'):
                        st.code(str(exc))
    current = st.session_state.get('demo_result')
    active = current if current and current['signature']==signature else None
    with preview_col:
        st.subheader('Tiền xử lý khuôn mặt')
        if active:
            report, output = active['report'], Path(active['output'])
            col1,col2 = st.columns(2)
            with col1:
                show_image(output/'aligned.png',caption='Căn chỉnh · 178 × 218',width=230)
            with col2:
                show_image(output/'roi.png',caption='Vùng trích đặc trưng · ROI',width=230)
            st.caption('Cam: trán · Hồng/xanh lam: hai má · Xanh lá: mắt và mày · Tím: miệng')
            a,b,c = st.columns(3)
            a.metric('Trích đặc trưng',f"{report['extraction_seconds']*1000:.0f} ms")
            b.metric('So khớp',f"{report['matching_seconds']*1000:.0f} ms")
            c.metric('Ảnh được so sánh',f"{report['gallery_candidates']:,}")
            if report['warning']:
                st.warning('Gallery còn ít hơn 5 ảnh phù hợp sau khi lọc.')
        else:
            st.info('Chọn ảnh rồi bấm “Tìm 5 ảnh tương đồng” để xem ảnh căn chỉnh, ROI và kết quả.')
            st.markdown('''**Quy trình xử lý**

Phát hiện một khuôn mặt → căn chỉnh theo tâm mắt → trích 5 nhóm đặc trưng →
chuẩn hóa theo gallery → tính cosine có trọng số và xếp hạng.
''')
    if active:
        report = active['report']
        st.divider()
        st.subheader('Top-5 chân dung tương đồng')
        if not report['results']:
            st.warning('Không có ảnh còn lại sau khi áp dụng bộ lọc.')
        else:
            for index,(col,result) in enumerate(zip(st.columns(len(report['results'])),report['results']),1):
                with col:
                    st.markdown(f"**#{index} · Điểm {result['score']:.4f}**")
                    show_image(ROOT/result['path'],width='stretch')
                    st.caption(f"Ảnh {result['image_id']} · {Path(result['path']).name}")
            score_tab, vector_tab, detail_tab = st.tabs(['Bảng điểm & đóng góp','Vector đặc trưng','Chi tiết lần tìm kiếm'])
            with score_tab:
                table = result_table(report['results'])
                st.dataframe(table,hide_index=True,width='stretch',column_config={
                    name:st.column_config.NumberColumn(format='%.4f') for name in ['Điểm tổng',*GROUPS]})
                selected = st.selectbox('Giải thích điểm của kết quả',range(len(report['results'])),
                    format_func=lambda i:f"Hạng {i+1} · Ảnh {report['results'][i]['image_id']}")
                result = report['results'][selected]
                contribution = pd.DataFrame({'Nhóm':GROUPS,'Cosine':[result[f's{i}'] for i in range(1,6)],
                                             'Trọng số':WEIGHTS,
                                             'Đóng góp':[result[f's{i+1}']*WEIGHTS[i] for i in range(5)]})
                a,b = st.columns([1,1])
                a.dataframe(contribution,hide_index=True,width='stretch',column_config={
                    name:st.column_config.NumberColumn(format='%.4f') for name in ['Cosine','Trọng số','Đóng góp']})
                b.bar_chart(contribution.set_index('Nhóm')['Đóng góp'],horizontal=True,color='#117c80')
                st.caption(f"Tổng điểm = tổng đóng góp của năm nhóm = {result['score']:.6f}. Cosine F1–F3 có thể âm sau chuẩn hóa.")
            with vector_tab:
                from portrait_search.features import NAMES
                vector_group = st.selectbox('Nhóm đặc trưng',range(5),format_func=lambda i:GROUPS[i])
                key = f'f{vector_group+1}'
                st.dataframe(pd.DataFrame({'Thuộc tính':NAMES[vector_group],
                    'Giá trị thô':report['raw_vectors'][key],'Giá trị so khớp':report['matching_vectors'][key]}),
                    hide_index=True,width='stretch')
                st.caption('F1–F3: Z-score theo gallery. F4: chuẩn hóa L2 riêng ba khối. F5: histogram LBP của năm ROI.')
            with detail_tab:
                st.write(f"Mã lần tìm kiếm: {report['search_run_id']}")
                st.write('Đã loại trùng truy vấn' if report['exclude_duplicates'] else 'Giữ ảnh tự trùng để kiểm tra pipeline')
                st.json({'snapshot':report['normalization_id'],'feature_set':report['feature_set_id'],
                         'kiểm_tra_ROI':report['metadata']},expanded=False)
        json_data = json.dumps(report,ensure_ascii=False,indent=2).encode('utf-8')
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer,fieldnames=['image_id','path','score','s1','s2','s3','s4','s5'])
        writer.writeheader()
        writer.writerows(report['results'])
        a,b,_ = st.columns([1,1,2])
        a.download_button('Tải kết quả JSON',json_data,'portrait_results.json','application/json',width='stretch')
        b.download_button('Tải bảng điểm CSV',csv_buffer.getvalue().encode('utf-8-sig'),
                          'portrait_top5.csv','text/csv',width='stretch')

with gallery_tab:
    st.subheader('Gallery đã trích đặc trưng')
    st.caption('Mỗi nội dung ảnh có một đại diện. Các bản trùng được giữ nguyên trong dataset gốc.')
    keyword = st.text_input('Tìm theo tên file hoặc ID ảnh',placeholder='Ví dụ: 000055 hoặc 1241')
    gallery = read_db('''SELECT i.image_id,i.path,f.overlay_path FROM images i JOIN image_features f USING(image_id)
        WHERE f.normalization_id=? AND f.status='ok' AND i.duplicate_of IS NULL ORDER BY i.image_id''',
        (snapshot['normalization_id'],))
    filtered = [r for r in gallery if not keyword or keyword.lower() in Path(r['path']).name.lower() or keyword==str(r['image_id'])]
    st.caption(f'{len(filtered):,} ảnh phù hợp')
    if filtered:
        pages = (len(filtered)+19)//20
        page = st.selectbox('Trang',range(pages),format_func=lambda i:f'Trang {i+1}/{pages}')
        roi_view = st.toggle('Xem vùng trích đặc trưng',value=False)
        for offset in range(page*20,min((page+1)*20,len(filtered)),5):
            for col,item in zip(st.columns(5),filtered[offset:offset+5]):
                with col:
                    show_image(ROOT/(item['overlay_path'] if roi_view else item['path']),width='stretch')
                    st.caption(f"ID {item['image_id']} · {Path(item['path']).name}")
    else:
        st.info('Không tìm thấy ảnh theo từ khóa này.')

with about_tab:
    st.subheader('Hình học + HSV + LBP')
    st.write('Hệ thống tìm ảnh có ngoại hình tương đồng bằng các đặc trưng có thể giải thích. '
             'MediaPipe dùng để định vị các mốc khuôn mặt; điểm xếp hạng được tính từ năm nhóm đặc trưng.')
    st.dataframe(pd.DataFrame({'Nhóm':GROUPS,'Số chiều':[7,10,10,42,50],'Trọng số':WEIGHTS,
        'Nội dung':['Tỷ lệ khuôn mặt, hàm, cằm, bất đối xứng','Khoảng cách, tỷ lệ mắt và độ nghiêng mày',
                   'Tỷ lệ mũi, môi và khoảng cách dọc','Thống kê HSV và histogram HS/V','LBP uniform trên năm ROI']}),
        hide_index=True,width='stretch')
    st.code('S = 0.25×S1 + 0.20×S2 + 0.25×S3 + 0.10×S4 + 0.20×S5',language=None)
    st.markdown('**Cách trình diễn:** chọn ảnh mẫu, tìm khi giữ ảnh trùng để kiểm tra; '
                'sau đó bật loại trùng và tìm lại để xem các ảnh khác có điểm cao nhất. '
                'Cuối cùng tải ảnh bên ngoài, xem ROI và giải thích đóng góp từng nhóm.')
    st.info('Điểm tự so gần 1 kiểm tra tính nhất quán của pipeline. Chất lượng truy hồi chưa được '
            'đánh giá bằng tập nhãn độc lập; không diễn giải điểm thành phần trăm nhận diện.')
    checks = ROOT/'artifacts/verification.json'
    if checks.is_file():
        with st.expander('Kết quả kiểm tra dữ liệu đã chạy'):
            st.json(json.loads(checks.read_text(encoding='utf-8')))

st.divider()
st.caption('Portrait Lab · Bài tập lớn HCSDLĐPT · MediaPipe / OpenCV / NumPy / scikit-image / SQLite')
