# Phân tích Bệnh Tim: Luật Kết hợp & Phân cụm

Dự án này thực hiện phân tích tập dữ liệu bệnh tim (`HeartDiseaseTrain-Test.csv`) nhằm tìm ra các mẫu tiềm ẩn và nhóm các bệnh nhân có đặc điểm tương đồng. Dự án áp dụng hai kỹ thuật khai phá dữ liệu chính: **Luật kết hợp (Association Rules)** và **Phân cụm (Clustering)**.

## 📋 Mục lục
- [Giới thiệu](#giới-thiệu)
- [Dữ liệu](#dữ-liệu)
- [Công nghệ sử dụng](#công-nghệ-sử-dụng)
- [Phương pháp phân tích](#phương-pháp-phân-tích)
    - [1. Khai phá Luật kết hợp (Apriori)](#1-khai-phá-luật-kết-hợp-apriori)
    - [2. Phân cụm (K-Means)](#2-phân-cụm-k-means)
- [Kết quả](#kết-quả)
- [Cài đặt & Hướng dẫn](#cài-đặt--hướng-dẫn)
- [Tác giả](#tác-giả)

## 📖 Giới thiệu
Mục tiêu của bài tập là áp dụng các thuật toán học máy không giám sát để hiểu rõ hơn về dữ liệu bệnh tim:
1.  **Apriori:** Tìm mối liên hệ giữa các triệu chứng, chỉ số sức khỏe và khả năng mắc bệnh tim.
2.  **K-Means:** Phân chia bệnh nhân thành các nhóm (cluster) để xây dựng hồ sơ rủi ro.

## 📊 Dữ liệu
Tập dữ liệu: `HeartDiseaseTrain-Test.csv`
Kích thước: 1025 dòng, 14 cột.

Các thuộc tính chính bao gồm:
- `age`: Tuổi
- `sex`: Giới tính
- `chest_pain_type`: Loại đau ngực
- `resting_blood_pressure`: Huyết áp khi nghỉ
- `cholestoral`: Cholesterol
- `target`: Biến mục tiêu (0 hoặc 1)
- Và các chỉ số khác (ECG, nhịp tim tối đa, thalassemia, v.v.)

## 🛠 Công nghệ sử dụng
Dự án được thực hiện bằng ngôn ngữ **Python** trên môi trường **Jupyter Notebook**.
Các thư viện chính:
* **Xử lý dữ liệu:** `pandas`, `numpy`
* **Trực quan hóa:** `matplotlib`, `seaborn`
* **Khai phá luật:** `mlxtend` (apriori, association_rules)
* **Học máy:** `scikit-learn` (StandardScaler, OneHotEncoder, KMeans, silhouette_score)

## 📈 Phương pháp phân tích

### 1. Khai phá Luật kết hợp (Apriori)
Để áp dụng thuật toán Apriori, dữ liệu số được chuyển đổi sang dạng phân loại (Discretization/Binning):
* **Tiền xử lý:**
    * Phân nhóm độ tuổi (Thanh niên, Trung niên, Cao niên, Người già).
    * Phân nhóm huyết áp (Bình thường, Tiền cao huyết áp, Cao huyết áp).
    * Phân nhóm Cholesterol và Nhịp tim.
    * Chuyển đổi toàn bộ dữ liệu sang định dạng `Thuộc_tính=Giá_trị`.
* **Mô hình:** Sử dụng thuật toán Apriori để tìm các tập phổ biến (frequent itemsets) với `min_support=0.2`.
* **Luật:** Sinh luật kết hợp dựa trên độ đo `lift`.

### 2. Phân cụm (K-Means)
Phân nhóm bệnh nhân dựa trên sự tương đồng về đặc điểm.
* **Tiền xử lý:**
    * Mã hóa One-Hot cho các biến phân loại (Sex, Chest Pain Type...).
    * Chuẩn hóa (Scaling) cho các biến số (Age, BP, Cholesterol...).
* **Tìm K tối ưu:** Sử dụng **Phương pháp Elbow** (Khuỷu tay) để xác định số lượng cụm hợp lý.
* **Đánh giá:** Sử dụng **Silhouette Score** để đánh giá chất lượng phân cụm.

## 📝 Kết quả
* **Luật kết hợp:** Đã tìm ra các luật có độ tin cậy (confidence) và lift cao, chỉ ra mối quan hệ mạnh mẽ giữa các yếu tố như *đau ngực kiểu không điển hình* hoặc *nhịp tim cao* với khả năng mắc bệnh.
* **Phân cụm:** Dữ liệu được chia thành các cụm (ví dụ: K=3) đại diện cho các nhóm bệnh nhân có hồ sơ rủi ro khác nhau.

## ⚙️ Cài đặt & Hướng dẫn
Để chạy dự án này, bạn cần cài đặt các thư viện phụ thuộc:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend

Sau đó mở file notebook:
jupyter notebook "Lò_Văn_Long_BT_buổi4.ipynb"
