OCRTrans {VERSION}
================================

Dich phu de / van ban tieng Anh sang tieng Viet ngay tren man hinh.

CACH DUNG
---------
1. Giai nen thu muc nay ra mot cho bat ky (vi du D:\OCRTrans).
2. Chay OCRTrans.exe
3. Bam "Add region" roi keo chon vung co chu can dich (thuong la vung phu de).
4. Chon engine:
     - argos  : offline, chay CPU, nhe. Lan dau se tai goi ngon ngu ~100 MB.
     - google / mymemory : online, mien phi, khong can key (phai tick dong y
       cho phep gui van ban len nha cung cap).
     - gemini / deepl : online, can API key.
5. Bam Start. Phim tat Ctrl+Alt+Z de bat/tat nhanh.

KHONG CAN CAI DAT
-----------------
Khong can Python, khong can NVIDIA. Ban nay la ban CPU: OCR chay bang
onnxruntime CPU, dich bang Argos. May co GPU NVIDIA nen dung ban cai dat tu
ma nguon (install.ps1) de co NLLB va OCR chay CUDA nhanh hon nhieu.

NEU CHAY CHAM
-------------
Ban CPU xu ly khoang 1-2 lan OCR/giay. De nhanh hon:
  - Chon vung nho hon (chi vung phu de, khong chon ca man hinh).
  - Giam FPS trong cua so dieu khien.
  - Doi engine sang "google" neu duoc phep dung mang.

DU LIEU LUU O DAU
-----------------
  %LOCALAPPDATA%\OCRTrans\config.json   (cai dat, vung chon)
  %LOCALAPPDATA%\OCRTrans\app.log       (nhat ky, gui kem khi bao loi)

GO LOI / BAO LOI
----------------
Mo issue tai: https://github.com/tuanvu17112003/ocr-trans-overlay/issues
Kem theo file app.log va mo ta man hinh luc loi.

QUYEN RIENG TU
--------------
Khong co telemetry. Engine offline (argos) khong gui gi ra ngoai.
Engine online chi gui van ban da nhan dang, va chi khi ban da tick dong y.

GIAY PHEP
---------
MIT License. Xem LICENSE.txt
