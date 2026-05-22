# Vision Recognition Toolkit
### OCR + Object Detection · DecodeLabs AI Industrial Training Kit · Batch 2026

A dual-mode computer vision pipeline that extracts text from images and detects objects in real time — powered by **Google's Tesseract OCR engine** and **MobileNet-SSD** deep learning architecture, all running locally with no cloud dependency.

---

## ✨ What It Does

| Mode | Engine | Task |
|------|--------|------|
| `ocr` | pytesseract + Tesseract LSTM | Extract machine-readable text from any image |
| `detect` | cv2.dnn + MobileNet-SSD | Identify & locate 20 object classes with bounding boxes |
| `both` | Both pipelines | Run OCR and detection sequentially on one image |

Both modes enforce an **80% minimum confidence gate** — low-confidence results are silently dropped, keeping output clean and reliable.

---

## 🚀 Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR engine

> Required for OCR mode only. Object detection has no external engine dependency.

| OS | Instructions |
|----|-------------|
| **Windows** | Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and run the installer (keep the default path) |
| **macOS** | `brew install tesseract` |
| **Linux** | `sudo apt install tesseract-ocr` |

The script **auto-detects** Tesseract from all common install locations — no manual path configuration needed.

### 3. Run

```bash
# Extract text from a document or sign
python project4_recognition.py ocr invoice.jpg

# Detect objects in a photo
python project4_recognition.py detect street.jpg

# Run both pipelines on the same image
python project4_recognition.py both sample.jpg
```

---

## 🔬 How It Works

### Path 1 — OCR Pipeline

```
Input Image
    │
    ▼
① Grayscale Conversion   — collapses RGB → intensity matrix
    │
    ▼
② Gaussian Blur (5×5)    — smooths JPEG noise and artifacts
    │
    ▼
③ Otsu's Binarisation    — adaptive black/white threshold
    │
    ▼
Tesseract LSTM Engine    — convolutional + bidirectional LSTM
    │
    ▼
Confidence Gate (≥ 80%)  — low-confidence words discarded
    │
    ▼
output_ocr_result.png    — original image + green word boxes
```

**Tesseract PSM modes available** (edit `custom_config` in source):

| PSM | Best For |
|-----|----------|
| `--psm 3` | Mixed / unknown layouts |
| `--psm 6` | Uniform blocks of text (default) |
| `--psm 7` | Single lines (number plates, headers) |
| `--psm 11` | Sparse scattered text (invoices) |

---

### Path 2 — Object Detection Pipeline

```
Input Image
    │
    ▼
Blob Construction        — resize 300×300, mean subtract 127.5, scale ÷127.5
    │
    ▼
MobileNet-SSD Forward Pass — single-shot detection across 20 VOC classes
    │
    ▼
Confidence Gate (≥ 80%)  — weak detections dropped
    │
    ▼
Coordinate Scaling       — normalised [0,1] → pixel coordinates
    │
    ▼
output_detection_result.png — colour-coded bounding boxes + labels
```

**Detectable object classes (VOC 2007):**
`aeroplane · bicycle · bird · boat · bottle · bus · car · cat · chair · cow · diningtable · dog · horse · motorbike · person · pottedplant · sheep · sofa · train · tvmonitor`

> Model files (`deploy.prototxt` + `mobilenet_iter_73000.caffemodel`) are downloaded automatically on first run and cached in `./mobilenet_ssd/`.

---

## 📦 Project Structure

```
├── project4_recognition.py   # Main script — both pipelines
├── requirements.txt           # Python dependencies
├── mobilenet_ssd/             # Auto-created: cached model files
│   ├── deploy.prototxt
│   └── mobilenet_iter_73000.caffemodel
├── output_preprocessed_ocr.png    # Generated: grayscale+threshold image
├── output_ocr_result.png          # Generated: OCR word boxes overlay
└── output_detection_result.png    # Generated: detection bounding boxes
```

---

## 🛠️ Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥ 4.8.0 | Image processing + DNN inference |
| `pytesseract` | ≥ 0.3.10 | Tesseract Python wrapper |
| `numpy` | ≥ 1.24.0 | Array operations |
| Tesseract binary | 5.x recommended | OCR engine (system install) |

---

## 🧪 Tips for Best Results

**For OCR:**
- Use high-resolution images (≥ 300 DPI for documents)
- Typed/printed text works far better than handwriting
- Avoid heavy shadows or uneven lighting
- If confidence is low, try a cropped or straightened image

**For Object Detection:**
- Works best on clear, well-lit photos
- Objects should occupy a reasonable portion of the frame
- The 80% threshold is strict — try lower (e.g. `0.60`) for more detections at the cost of precision

---
# contact
  
   👤 Name: Arslan Nafees<br>
   📱 Phone: +92 334 111 3047<br>
   📧 Email: arslannafees807@gmail.com<br>
[![GitHub](https://img.shields.io/badge/GitHub-arslannafees-181717?style=flat&logo=github)](https://github.com/arslannafees)
