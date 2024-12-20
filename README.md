﻿# Phân Tích Dữ Liệu COVID-19
## Thông tin nhóm
Nhóm: KNK
1. Trần Quốc Khánh 23020387
2. Nguyễn Hữu Hoàng Nam 23020405
3. Nguyễn Gia Khánh 23020385




Dự án này sử dụng Python, Jupyternotebook và Streamlit để phân tích dữ liệu về đại dịch COVID-19, bao gồm số ca nhiễm, tử vong, tiêm chủng và tác động kinh tế - xã hội.  Ứng dụng web tương tác cho phép người dùng khám phá dữ liệu theo nhiều cách khác nhau, từ biểu đồ theo thời gian đến bản đồ thế giới và phân tích so sánh giữa các quốc gia và châu lục.

**Lưu ý:**

*  Nhớ thay thế `app.py` bằng tên file Python chính của dự án.
*  Tạo file `requirements.txt` chứa danh sách các thư viện Python mà dự án sử dụng.  Bạn có thể tạo file này bằng lệnh `pip freeze > requirements.txt`.
*  link demo dashboard `https://drive.google.com/file/d/1i0TpVSC1IpjlZpxTWLVt0wr6eVd95PI3/view?usp=sharing`.

## Phân tích chính

**Tổng quan:**

* Cung cấp cái nhìn tổng quan về tình hình dịch bệnh toàn cầu, bao gồm tổng số ca nhiễm, tử vong, và tiêm chủng.
* Số ca nhiễm, tử vong, và tiêm chủng theo thời gian, cho phép người dùng lựa chọn khoảng thời gian cụ thể.
* Bản đồ thế giới hiển thị số ca nhiễm, tử vong, hỗ trợ giãn cách xã hội.


**Phân tích theo khu vực (Châu lục, Quốc gia):**

* Biểu đồ tròn thể hiện tỉ lệ ca nhiễm/tử vong theo châu lục.
* Xu hướng số ca nhiễm/tử vong/tiêm chủng theo thời gian tại các châu lục.
* Top 10 quốc gia có số ca nhiễm/tử vong cao nhất thế giới.


**Phân tích tương quan và kiểm định:**

* **Tương quan:**
    * Tương quan giữa ca nhiễm và tử vong.
    * Tương quan giữa tỉ lệ hồi phục và ca nhiễm mới.
    * Tương quan giữa các biến thể chính (ví dụ: Delta và Omicron).
    * Tương quan giữa tiêm chủng và ca nhiễm/tử vong.

* **Kiểm định giả thuyết và dự đoán:**
    * Kiểm định tác động của biện pháp phong tỏa đến số ca nhiễm/tử vong.
    * Kiểm định tác động của tiêm chủng đến số ca nhiễm/tử vong.
    * So sánh hai biến thể Omicron và Delta.
    * Dự đoán số ca nhiễm/tử vong trong tương lai gần.

**Ảnh hưởng của COVID-19:**

* Ảnh hưởng đến kinh tế của các nước phát triển trong Quý 2 năm 2020.
* Ảnh hưởng đến du lịch (số lượng các chuyến đi).
* Ảnh hưởng đến GDP và tỉ lệ thất nghiệp trong giai đoạn 2020-2021.
* Ảnh hưởng đến môi trường (lượng khí thải CO2).

## Công nghệ sử dụng

* Python
* Pandas
* Streamlit
* Plotly
* Scikit-learn
* Matplotlib
* Seaborn
* Geopandas

## Cài đặt và chạy dự án

1. Cài đặt các thư viện cần thiết:
pip freeze > requirements.txt
```bash
pip install -r requirements.txt 
2. chạy dự án 
streamlit run app.py
