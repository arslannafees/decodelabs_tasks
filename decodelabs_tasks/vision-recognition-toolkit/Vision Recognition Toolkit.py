import os
import sys
import shutil
import urllib.request
import cv2
import numpy as np


# =============================================================================
#  TESSERACT AUTO-DETECTION  (Windows / Linux / macOS)
# =============================================================================

def find_tesseract() -> str:
    """
    Auto-locate the Tesseract executable.
    Checks PATH first (Linux/macOS), then all common Windows install folders.
    Returns the full path if found, or empty string if not found.
    """
    # Check PATH first — works automatically on Linux and macOS
    path_result = shutil.which("tesseract")
    if path_result:
        return path_result

    # Windows common install locations
    username = os.environ.get("USERNAME", "") or os.environ.get("USER", "")
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        rf"C:\Users\{username}\AppData\Local\Tesseract-OCR\tesseract.exe",
        r"C:\Tesseract-OCR\tesseract.exe",
        r"C:\tesseract\tesseract.exe",
        r"C:\tools\tesseract\tesseract.exe",
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    return ""


def setup_tesseract(pytesseract_module) -> None:
    """
    Find Tesseract and configure pytesseract to use it.
    Exits with a helpful install guide if not found.
    """
    tess_path = find_tesseract()

    if tess_path:
        pytesseract_module.pytesseract.tesseract_cmd = tess_path
        print(f"  [TESSERACT] Found -> {tess_path}")
    else:
        print("\n[ERROR] Tesseract OCR engine is not installed or could not be found.")
        print()
        print("  HOW TO FIX:")
        print("  1. Go to: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  2. Download: tesseract-ocr-w64-setup-5.x.x.exe")
        print("  3. Run the installer (keep the default install path)")
        print("  4. Run this script again -- it will be detected automatically")
        print()
        print("  If you installed it to a custom folder, open project4_recognition.py")
        print("  and add this line inside run_ocr(), after 'import pytesseract':")
        print(r"  pytesseract.pytesseract.tesseract_cmd = r'C:\your\path\tesseract.exe'")
        sys.exit(1)


# =============================================================================
#  MODEL DOWNLOAD HELPER
# =============================================================================

def download_file(url: str, dest_path: str) -> bool:
    """
    Download a file from url to dest_path.
    Shows a simple progress indicator.
    Returns True on success, False on failure.
    """
    try:
        def show_progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(block_num * block_size * 100 / total_size, 100)
                print(f"\r    Downloading... {pct:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest_path, reporthook=show_progress)
        print()  # newline after progress
        return True
    except Exception as e:
        print(f"\n  [DOWNLOAD ERROR] {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)   # remove incomplete file
        return False


def download_model_files() -> tuple:
    """
    Download MobileNet-SSD v1 Caffe model files if not already present.
    Files are cached in ./mobilenet_ssd/ folder.
    Returns (prototxt_path, caffemodel_path).
    """
    model_dir = "mobilenet_ssd"
    os.makedirs(model_dir, exist_ok=True)

    prototxt_path    = os.path.join(model_dir, "deploy.prototxt")
    caffemodel_path  = os.path.join(model_dir, "mobilenet_iter_73000.caffemodel")

    prototxt_url    = (
        "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
        "master/deploy.prototxt"
    )
    caffemodel_url  = (
        "https://github.com/chuanqi305/MobileNet-SSD/raw/master/"
        "mobilenet_iter_73000.caffemodel"
    )

    for local_path, url in [(prototxt_path, prototxt_url),
                             (caffemodel_path, caffemodel_url)]:
        fname = os.path.basename(local_path)
        if os.path.exists(local_path):
            print(f"  [CACHE]    {fname} already present -- skipping download")
        else:
            print(f"  [DOWNLOAD] Fetching {fname} ...")
            ok = download_file(url, local_path)
            if not ok:
                print(f"\n  [ERROR] Could not download {fname}.")
                print("  Manual download links:")
                print(f"    prototxt   : {prototxt_url}")
                print(f"    caffemodel : {caffemodel_url}")
                print("  Place both files inside the 'mobilenet_ssd' folder.")
                sys.exit(1)
            print(f"  [OK]       Saved -> {local_path}")

    return prototxt_path, caffemodel_path


# =============================================================================
#  VOC CLASS LABELS  (MobileNet-SSD v1 — 21 classes)
# =============================================================================

VOC_LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(VOC_LABELS), 3), dtype="uint8")


# =============================================================================
#  SHARED PRE-PROCESSING PIPELINE
# =============================================================================

def load_image(path: str) -> np.ndarray:
    """Load an image from disk. Exits with a clear error if not found."""
    image = cv2.imread(path)
    if image is None:
        print(f"\n[ERROR] Could not load image: '{path}'")
        print("  Make sure:")
        print("  1. The image file is in the same folder as this script")
        print("  2. The filename and extension are correct (e.g. invoice.jpg)")
        sys.exit(1)
    print(f"[INFO]  Image loaded -> {image.shape[1]}x{image.shape[0]} px  |  {path}")
    return image


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Systematic 3-step pre-processing pipeline for OCR (Path 1).

    Step 1 -- Grayscale Conversion
        Collapses the 3D RGB matrix into a 1D intensity matrix.
        Removes distracting colour data so the engine focuses on
        character shape (edges and contours).

    Step 2 -- Gaussian Blur
        Smooths the image to eliminate micro-imperfections and
        artifact noise (JPEG rings, salt-and-pepper noise).
        Kernel size (5,5), sigma=0 lets OpenCV auto-calculate spread.

    Step 3 -- Adaptive Thresholding (Otsu's method)
        Forces every pixel to choose a side: 0 (Black) or 255 (White).
        Threshold computed per-region so it handles uneven lighting.
        IF pixel_intensity >= T  THEN pixel = 255 (White)
        IF pixel_intensity <  T  THEN pixel = 0   (Black)
    """
    print("\n[PRE-PROCESSING] OCR Pipeline started ...")

    # Step 1 -- Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("  Step 1 OK  Grayscale conversion -- RGB collapsed to intensity matrix")

    # Step 2 -- Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("  Step 2 OK  Gaussian Blur (5x5)  -- noise and artifacts smoothed")

    # Step 3 -- Otsu's Binarisation
    _, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print("  Step 3 OK  Adaptive Threshold   -- binary decision forced (Otsu's method)")

    out_path = "output_preprocessed_ocr.png"
    cv2.imwrite(out_path, thresh)
    print(f"  [SAVED] Pre-processed image -> {out_path}")

    return thresh


# =============================================================================
#  PATH 1 -- OPTICAL CHARACTER RECOGNITION
# =============================================================================

def run_ocr(image_path: str) -> None:
    """
    Path 1: OCR using pytesseract (wrapper for Google's Tesseract engine).

    Engine internals: convolutional + bi-directional LSTM pipeline.
    PSM modes:
        --psm 3  : Fully automatic (varied layouts)
        --psm 6  : Single uniform block of text (book pages)
        --psm 7  : Single text line (number plates, headers)
        --psm 11 : Sparse, scattered text (invoices)
    """
    # -- Import and configure pytesseract ------------------------------------
    try:
        import pytesseract
    except ImportError:
        print("\n[ERROR] pytesseract Python package is not installed.")
        print("  Fix:  pip install pytesseract")
        sys.exit(1)

    setup_tesseract(pytesseract)   # auto-find and configure Tesseract engine

    print("\n" + "=" * 60)
    print("  PATH 1 -- OPTICAL CHARACTER RECOGNITION (OCR)")
    print("=" * 60)

    original  = load_image(image_path)
    processed = preprocess_for_ocr(original)

    print("\n[OCR]  Running Tesseract engine ...")

    custom_config = r"--oem 3 --psm 6"

    try:
        raw_text = pytesseract.image_to_string(processed, config=custom_config)
        data     = pytesseract.image_to_data(
            processed,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )
    except Exception as e:
        print(f"\n[ERROR] Tesseract engine failed: {e}")
        print("  Try reinstalling Tesseract from:")
        print("  https://github.com/UB-Mannheim/tesseract/wiki")
        sys.exit(1)

    # -- Confidence score ----------------------------------------------------
    confidences    = [int(c) for c in data["conf"] if str(c).isdigit() and int(c) >= 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # -- 80% Confidence Gate -------------------------------------------------
    CONFIDENCE_THRESHOLD = 80.0
    passed = avg_confidence >= CONFIDENCE_THRESHOLD

    print("\n" + "-" * 60)
    print("  EXTRACTED TEXT:")
    print("-" * 60)
    cleaned = raw_text.strip()
    print(cleaned if cleaned else "[No text detected -- try a clearer image]")
    print("-" * 60)
    print(f"\n  Average Confidence Score : {avg_confidence:.1f}%")
    print(f"  Minimum Required         : {CONFIDENCE_THRESHOLD:.0f}%")

    if passed:
        print(f"  Gate Status              : PASSED  ({avg_confidence:.1f}% >= {CONFIDENCE_THRESHOLD:.0f}%)")
    else:
        print(f"  Gate Status              : FAILED  ({avg_confidence:.1f}% < {CONFIDENCE_THRESHOLD:.0f}%)")
        print("  Tip: use a clear, high-resolution image of typed text.")

    # -- Visual output: draw boxes around confident words --------------------
    output_img = original.copy()
    words  = data["text"]
    lefts  = data["left"]
    tops   = data["top"]
    widths = data["width"]
    heights= data["height"]
    confs  = data["conf"]

    for i, word in enumerate(words):
        word = word.strip()
        if not word:
            continue
        conf_val = int(confs[i]) if str(confs[i]).isdigit() else 0
        if conf_val < CONFIDENCE_THRESHOLD:
            continue   # 80% gate
        x, y, w, h = lefts[i], tops[i], widths[i], heights[i]
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(output_img, f"{conf_val}%",
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

    out_path = "output_ocr_result.png"
    cv2.imwrite(out_path, output_img)
    print(f"\n  [SAVED] Visual output -> {out_path}")
    print("=" * 60 + "\n")


# =============================================================================
#  PATH 2 -- OBJECT DETECTION  (cv2.dnn + MobileNet-SSD)
# =============================================================================

def run_object_detection(image_path: str) -> None:
    """
    Path 2: Object Detection using cv2.dnn + MobileNet-SSD.

    Step 1 -- Blob Construction (blobFromImage)
        4D tensor the network expects:
        - Resize to 300x300 (MobileNet-SSD input size)
        - Mean subtraction: 127.5 per channel
        - Scale factor: 1/127.5  (pixel range -> [-1, 1])

    Step 2 -- Forward Pass (Single Shot Detector)
        One pass detects ALL objects simultaneously.
        Network outputs normalised coordinates (0.0 - 1.0).

    Step 3 -- Confidence Gate (80% minimum)
        if confidence >= 0.80  ->  draw box and label
        else                   ->  drop detection

    Step 4 -- Coordinate Scaling
        pixel_x = norm_x * image_width
        pixel_y = norm_y * image_height
        Box described as  (X, Y, W, H)
    """
    print("\n" + "=" * 60)
    print("  PATH 2 -- OBJECT DETECTION (MobileNet-SSD)")
    print("=" * 60)

    # -- Load model ----------------------------------------------------------
    print("\n[MODEL] Loading MobileNet-SSD ...")
    prototxt, caffemodel = download_model_files()

    try:
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    except Exception as e:
        print(f"\n[ERROR] Failed to load model files: {e}")
        print("  Delete the 'mobilenet_ssd' folder and run again to re-download.")
        sys.exit(1)

    print("  [OK]  Model loaded into cv2.dnn")

    # -- Load image ----------------------------------------------------------
    original        = load_image(image_path)
    (img_h, img_w)  = original.shape[:2]

    # -- Step 1: Blob Construction -------------------------------------------
    print("\n[PRE-PROCESSING] Blob Construction ...")
    blob = cv2.dnn.blobFromImage(
        cv2.resize(original, (300, 300)),
        scalefactor = 1 / 127.5,
        size        = (300, 300),
        mean        = (127.5, 127.5, 127.5),
        swapRB      = True
    )
    print("  Step 1 OK  Blob constructed -- 4D tensor (1,3,300,300) ready")
    print("             Mean subtracted  | scaled to 300x300 | BGR->RGB")

    # -- Step 2: Forward Pass ------------------------------------------------
    print("  Step 2 OK  Running SSD single-shot forward pass ...")
    net.setInput(blob)
    detections = net.forward()
    print("  Step 2 OK  Output tensor received")

    # -- Step 3 & 4: Gate + Draw Boxes ---------------------------------------
    CONFIDENCE_THRESHOLD = 0.80
    output_img           = original.copy()
    detection_count      = 0

    print(f"\n[DETECTIONS] Applying {CONFIDENCE_THRESHOLD*100:.0f}% confidence gate ...")
    print("-" * 60)
    print(f"  {'#':<4} {'Label':<16} {'Confidence':>10}   Bounding Box (X,Y,W,H)")
    print("-" * 60)

    # detections shape: (1, 1, N, 7)
    # columns: [_, class_id, confidence, x1_norm, y1_norm, x2_norm, y2_norm]
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence < CONFIDENCE_THRESHOLD:
            continue   # 80% gate

        class_id = int(detections[0, 0, i, 1])
        label    = VOC_LABELS[class_id] if class_id < len(VOC_LABELS) else f"class_{class_id}"

        # Coordinate Scaling: normalised -> pixel
        x1 = int(detections[0, 0, i, 3] * img_w)
        y1 = int(detections[0, 0, i, 4] * img_h)
        x2 = int(detections[0, 0, i, 5] * img_w)
        y2 = int(detections[0, 0, i, 6] * img_h)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        box_x, box_y = x1, y1
        box_w, box_h = x2 - x1, y2 - y1

        color = [int(c) for c in COLORS[class_id % len(COLORS)]]

        # Draw bounding box
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label_text = f"{label}: {confidence*100:.1f}%"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = max(y1 - 10, label_size[1] + 10)

        cv2.rectangle(
            output_img,
            (x1, label_y - label_size[1] - 4),
            (x1 + label_size[0], label_y + 4),
            color, cv2.FILLED
        )
        cv2.putText(
            output_img, label_text,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )

        detection_count += 1
        print(f"  {detection_count:<4} {label:<16} {confidence*100:>9.1f}%   "
              f"({box_x}, {box_y}, {box_w}, {box_h})")

    print("-" * 60)
    if detection_count == 0:
        print("  No objects detected above the 80% threshold.")
        print("  Tip: use a photo with people, cars, animals, or common objects.")
    else:
        print(f"  Total detections (>=80%) : {detection_count}")

    out_path = "output_detection_result.png"
    cv2.imwrite(out_path, output_img)
    print(f"\n  [SAVED] Visual output -> {out_path}")
    print("=" * 60 + "\n")


# =============================================================================
#  ENTRY POINT
# =============================================================================

def print_banner():
    print("""
+--------------------------------------------------------------+
|   PROJECT 4  --  IMAGE & TEXT RECOGNITION                    |
|   DecodeLabs AI Industrial Training Kit  |  Batch 2026       |
+--------------------------------------------------------------+
|   Path 1 :  OCR             (pytesseract + OpenCV)           |
|   Path 2 :  Object Detection (cv2.dnn + MobileNet-SSD)       |
+--------------------------------------------------------------+
""")


def main():
    print_banner()

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python project4_recognition.py ocr     <image_path>")
        print("  python project4_recognition.py detect  <image_path>")
        print("  python project4_recognition.py both    <image_path>")
        print()
        print("Examples:")
        print("  python project4_recognition.py ocr    invoice.jpg")
        print("  python project4_recognition.py detect street.jpg")
        print("  python project4_recognition.py both   sample.jpg")
        sys.exit(0)

    mode       = sys.argv[1].lower()
    image_path = sys.argv[2]

    if mode == "ocr":
        run_ocr(image_path)
    elif mode in ("detect", "detection"):
        run_object_detection(image_path)
    elif mode == "both":
        run_ocr(image_path)
        run_object_detection(image_path)
    else:
        print(f"[ERROR] Unknown mode '{mode}'. Choose: ocr | detect | both")
        sys.exit(1)


if __name__ == "__main__":
    main()
