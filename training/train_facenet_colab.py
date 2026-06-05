# =============================================================================
# Smart Face Attendance System — FaceNet Training Pipeline
# Run this in Google Colab (GPU runtime recommended)
# =============================================================================

# Cell 1: Setup & Params
# ----------------------
import os, glob, math, random, shutil, sys, time
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Install and import FaceNet
# !pip install -q keras-facenet
from keras_facenet import FaceNet
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

print("TF version:", tf.__version__)

# ================= CONFIG =================
IMG_SIZE        = (160, 160)   # FaceNet standard input size
BATCH_SIZE      = 32
SEED            = 123
MAX_PER_CLASS   = 200
EMBEDDING_DIM   = 512          # FaceNet default embedding dimension

DATA_ROOT       = "/content/dataset"
DATA_RAW        = None         # auto-detected below
DRIVE_DEST      = "/content/drive/MyDrive"

USE_CENTROID        = True     # one embedding per person (recommended)
SAVE_MULTIPLE_EMBS  = False    # True => store one embedding per image
CONVERT_INT8        = True     # int8 quantization for mobile/PYNQ
VERBOSE             = True

def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


# Cell 2: Mount Drive & Unzip Dataset
# ------------------------------------
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
else:
    log("Not running in Colab — ensure dataset is at DATA_ROOT manually.")

# Place your zipped dataset in Google Drive as "dataset.zip"
ZIP_PATH = os.path.join(DRIVE_DEST, "dataset.zip")

if os.path.exists(ZIP_PATH) and not os.path.exists(DATA_ROOT):
    log("Unzipping", ZIP_PATH, "->", DATA_ROOT)
    os.system(f'unzip -q "{ZIP_PATH}" -d /content/')
else:
    log("No dataset.zip found at", ZIP_PATH, "or dataset already exists.")

# Auto-detect dataset folder layout:
#   /content/dataset/raw/<class_folders>   OR
#   /content/dataset/<class_folders>
DATA_RAW = None
raw_path = os.path.join(DATA_ROOT, "raw")

if os.path.isdir(raw_path):
    DATA_RAW = raw_path
else:
    entries = [
        d for d in glob.glob(os.path.join(DATA_ROOT, "*"))
        if os.path.isdir(d) and not d.endswith("raw")
    ]
    for d in entries:
        imgs = glob.glob(os.path.join(d, "*"))
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in imgs):
            DATA_RAW = DATA_ROOT
            break

if DATA_RAW is None:
    raise RuntimeError(
        "Could not auto-detect dataset folder.\n"
        "Expected: DATA_ROOT/raw/<class> OR DATA_ROOT/<class>"
    )
else:
    log("Using DATA_RAW =", DATA_RAW)


# Cell 3: Inspect & Validate Dataset
# ------------------------------------
assert DATA_RAW is not None, "DATA_RAW is None — dataset detection failed."

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')

classes = sorted([
    d for d in os.listdir(DATA_RAW)
    if os.path.isdir(os.path.join(DATA_RAW, d))
])

if len(classes) < 2:
    raise RuntimeError("Need at least 2 classes for face recognition.")

log("Classes found:", classes)

for c in classes:
    class_dir = os.path.join(DATA_RAW, c)
    images = [
        f for f in glob.glob(os.path.join(class_dir, "*"))
        if f.lower().endswith(IMAGE_EXTS)
    ]
    if len(images) < 20:
        log(f"⚠️  WARNING: Class '{c}' has only {len(images)} images. Recognition may be unstable.")
    log(f"  {c}: {len(images)} images")

CLASSES = classes.copy()  # lock class order for reproducibility


# Cell A: Load Pre-trained FaceNet Model
# ----------------------------------------
facenet = FaceNet()
embedding_model = facenet.model

log("✅ Loaded pre-trained FaceNet model")
log(f"   Input shape:  {embedding_model.input_shape}")
log(f"   Output shape: {embedding_model.output_shape}")
log(f"   Embedding dim: {embedding_model.output_shape[-1]}")

# Optional: load custom weights from Drive if available
h5_path = os.path.join(DRIVE_DEST, "facenet_custom.h5")
if os.path.exists(h5_path):
    try:
        embedding_model.load_weights(h5_path, by_name=True)
        log(f"✅ Loaded custom weights from {h5_path}")
    except Exception as e:
        log(f"⚠️  Could not load custom weights: {e}")


# Cell B: Compute Embeddings & Build Gallery
# -------------------------------------------
import cv2

gallery_list = []
names        = []
per_image_embeddings = {}

for cls in sorted(os.listdir(DATA_RAW)):
    cls_dir = os.path.join(DATA_RAW, cls)
    if not os.path.isdir(cls_dir):
        continue

    files = sorted([
        f for f in glob.glob(os.path.join(cls_dir, "*"))
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])[:MAX_PER_CLASS]

    embs = []

    for p in files:
        im = cv2.imread(p)
        if im is None:
            log("Warning: couldn't read", p)
            continue

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, IMG_SIZE).astype(np.float32)

        # FaceNet preprocessing: standardise to [-1, 1]
        im = (im - 127.5) / 128.0

        e = embedding_model.predict(np.expand_dims(im, 0), verbose=0)[0]

        # L2 normalise
        e = e / (np.linalg.norm(e) + 1e-10)
        embs.append(e.astype(np.float32))

        if SAVE_MULTIPLE_EMBS:
            per_image_embeddings.setdefault(cls, []).append((p, e.astype(np.float32)))

    if len(embs) == 0:
        log(f"Warning: no images found for class {cls}, skipping.")
        continue

    if USE_CENTROID:
        centroid = np.mean(embs, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        gallery_list.append(centroid.astype(np.float32))
        names.append(cls)
    else:
        for e in embs:
            gallery_list.append(e)
            names.append(cls)

if len(gallery_list) == 0:
    raise SystemExit("No gallery embeddings created — check DATA_RAW and images.")

gallery = np.vstack(gallery_list).astype(np.float32)
gallery = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-10)

# Save artifacts
OUT_DIR = "/content/artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

np.save(os.path.join(OUT_DIR, "gallery.npy"), gallery)
with open(os.path.join(OUT_DIR, "names.txt"), "w") as f:
    f.write("\n".join(names))

log(f"Saved gallery.npy {gallery.shape} and names.txt to {OUT_DIR}")


# Cell C: Convert FaceNet → TFLite (int8, with float fallback)
# --------------------------------------------------------------
rep_paths = []
for cls in sorted(os.listdir(DATA_RAW)):
    p = glob.glob(os.path.join(DATA_RAW, cls, "*"))
    rep_paths += p[:50]
rep_paths = rep_paths[:200]

log("Representative images for quantisation:", len(rep_paths))

def representative_gen():
    for p in rep_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE).astype(np.float32)
        img = (img - 127.5) / 128.0    # FaceNet preprocessing — must match inference!
        yield [np.expand_dims(img, 0)]

# Fix concrete input shape before conversion
run_model    = tf.function(lambda x: embedding_model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec(shape=[1, 160, 160, 3], dtype=tf.float32)
)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

tflite_paths = {}

if CONVERT_INT8:
    try:
        log("Attempting FULL_INT8 conversion...")
        converter.optimizations               = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset      = representative_gen
        converter.target_spec.supported_ops   = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type        = tf.int8
        converter.inference_output_type       = tf.int8

        tflite_model = converter.convert()
        int8_path    = os.path.join(OUT_DIR, "model_int8.tflite")
        open(int8_path, "wb").write(tflite_model)
        tflite_paths["int8"] = int8_path
        log("✅ Saved int8 model:", int8_path)

    except Exception as e:
        log("⚠️  INT8 conversion failed:", e)

# Always produce a float32 model (for PC testing)
run_model2     = tf.function(lambda x: embedding_model(x))
concrete_func2 = run_model2.get_concrete_function(
    tf.TensorSpec(shape=[1, 160, 160, 3], dtype=tf.float32)
)
converter2 = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func2])
converter2.optimizations = []

dyn_path = os.path.join(OUT_DIR, "model_dynamic.tflite")
open(dyn_path, "wb").write(converter2.convert())
tflite_paths["float"] = dyn_path
log("✅ Saved float32 model:", dyn_path)

# Verify
interp_test  = tf.lite.Interpreter(model_path=dyn_path)
interp_test.allocate_tensors()
log(f"✅ Verified TFLite input shape: {interp_test.get_input_details()[0]['shape']}")


# Cell D: Save Artifacts to Drive & Zip for Download
# ----------------------------------------------------
facenet_h5_path = os.path.join(OUT_DIR, "facenet_model.h5")
embedding_model.save(facenet_h5_path)
log(f"✅ Saved FaceNet model: {facenet_h5_path}")

os.makedirs(DRIVE_DEST, exist_ok=True)

artifacts = [
    os.path.join(OUT_DIR, "gallery.npy"),
    os.path.join(OUT_DIR, "names.txt"),
    os.path.join(OUT_DIR, "model_dynamic.tflite"),
    os.path.join(OUT_DIR, "model_int8.tflite") if os.path.exists(os.path.join(OUT_DIR, "model_int8.tflite")) else None,
    facenet_h5_path,
]
artifacts = [p for p in artifacts if p and os.path.exists(p)]

for p in artifacts:
    shutil.copy(p, os.path.join(DRIVE_DEST, os.path.basename(p)))
    log("Copied", os.path.basename(p), "->", DRIVE_DEST)

ZIP_BASE = os.path.join(OUT_DIR, "face_attendance_artifacts")
ZIP_OUT  = shutil.make_archive(base_name=ZIP_BASE, format='zip', root_dir=OUT_DIR)
shutil.copy(ZIP_OUT, os.path.join(DRIVE_DEST, os.path.basename(ZIP_OUT)))

log("\n" + "="*60)
log("🎉 PIPELINE COMPLETE!")
log("="*60)
log(f"📁 Artifacts saved to: {DRIVE_DEST}")
log(f"📦 Download zip:       {os.path.basename(ZIP_OUT)}")
log("="*60)


# Optional Helper: Add New Person Without Retraining
# ----------------------------------------------------
def add_new_person_from_folder(
    person_name,
    folder_path,
    model       = embedding_model,
    out_gallery = os.path.join(OUT_DIR, "gallery.npy"),
    out_names   = os.path.join(OUT_DIR, "names.txt")
):
    """
    Add a new person to the gallery without retraining.
    Simply compute their centroid embedding and append to gallery.npy + names.txt.

    Args:
        person_name : label for the new person
        folder_path : directory containing their face images
        model       : FaceNet embedding model
        out_gallery : path to gallery.npy
        out_names   : path to names.txt
    """
    imgs = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])[:MAX_PER_CLASS]

    if len(imgs) == 0:
        raise ValueError(f"No images found in {folder_path}")

    log(f"Processing {len(imgs)} images for {person_name}...")

    embs = []
    for p in imgs:
        im = cv2.imread(p)
        if im is None:
            log(f"⚠️  Skipping unreadable image: {p}")
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, IMG_SIZE).astype(np.float32)
        im = (im - 127.5) / 128.0   # must match training preprocessing!
        e  = model.predict(np.expand_dims(im, 0), verbose=0)[0]
        e  = e / (np.linalg.norm(e) + 1e-10)
        embs.append(e.astype(np.float32))

    if len(embs) == 0:
        raise ValueError(f"No valid embeddings generated for {person_name}")

    rep = np.mean(embs, axis=0)
    rep = rep / (np.linalg.norm(rep) + 1e-10)

    gallery_arr = np.load(out_gallery)
    name_list   = [l.strip() for l in open(out_names).read().splitlines()]

    if person_name in name_list:
        log(f"⚠️  {person_name} already in gallery — overwriting.")
        idx = name_list.index(person_name)
        gallery_arr[idx] = rep
    else:
        gallery_arr = np.vstack([gallery_arr, rep])
        name_list.append(person_name)

    np.save(out_gallery, gallery_arr)
    with open(out_names, "w") as f:
        f.write("\n".join(name_list))

    shutil.copy(out_gallery, os.path.join(DRIVE_DEST, "gallery.npy"))
    shutil.copy(out_names,   os.path.join(DRIVE_DEST, "names.txt"))

    log(f"✅ Added {person_name} → gallery shape {gallery_arr.shape}")
    log(f"   Used {len(embs)} images to compute centroid embedding")

# Example:
# add_new_person_from_folder("NewPerson", "/content/dataset/raw/NewPerson")

log("\n✅ All cells complete!")
