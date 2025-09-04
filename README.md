# RAG-Legal-Document-Assistant-UMKM
UMKM (Usaha Kecil Menengah) di Indonesia menghadapi kesulitan dalam mengakses informasi hukum yang relevan dan terpercaya untuk operasional bisnis mereka. Ini dibuat agar membantu UMKM dalam masalah legalitas hukum.

# 1. Identifikasi Masalah Bisnis
**Masalah Utama**<br>
UMKM di Indonesia sering menghadapi tantangan hukum dalam menjalankan bisnis mereka seperti kesulitan memahami regulasi bisnis yang kompleks, biaya konsultasi hukum yang mahal, atau kurangnya pengetahuan tentang dokumen legal yang diperlukan.

**Pertanyaan Bisnis**<br>
Bagaimana mengurangi biaya konsultasi hukum UMKM dengan tetap memberikan panduan yang akurat?<br>
Bagaimana memberikan akses 24/7 untuk konsultasi hukum dasar?

**Dampak yang diharapkan**<br>
Penghematan UMKM dari biaya konsultasi<br>
Mengurangi risiko sanksi hukum<br>
Mempercepat proses pengurusan izin dan dokumen legal

# 2. Datasets
Data yang digunakan berfokus pada scope utama UMKM antara lain:
Perizinan
Ketenagakerjaan
Perpajakan Dasar

**Kualitas dan relevansi data**<br>
Sumber resmi dari pemerintah<br>
Bahasa Indonesia formal dan legal

# 3. Data Preprocessing
**Data Cleaning**<br>
Format Dokumen<br>
* Dokumen UU/PP sering memiliki header, footer, watermark, dan nomor halaman yang harus dihapus karena tidak relevan untuk content understanding.
* Format dua kolom, indentasi yang tidak konsisten, dan spacing yang irregular memerlukan normalisasi layout.
* Dokumen lama sering menggunakan encoding yang berbeda yang harus di standarisasi ke UTF-8.

Text Normalization
* Mengubah karakter khusus seperti bullet points menjadi format standar
* Multiple spaces, tabs, dan line breaks yang tidak konsisten
* Berbagai jenis tanda kutip diseragamkan

Identifikasi dan Penanganan Noise
* Document Headers seperti informasi kop surat yang berulang
* Bagian pengesahan, tanda tangan, dan cap yang tidak relevan
* Nomor dokumen internal yang tidak memberikan value untuk semantic understanding

Penanganan Missing Values
* Identifikasi dokumen tidak lengkap
* Manual review

