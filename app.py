from pathlib import Path
from threading import Lock

from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Traffic Sign Detection", layout="wide")


@st.cache_resource
def load_model():
    return YOLO(str(ROOT / "freeze_model.pt")), Lock()


st.sidebar.title("User Configuration")
source = st.sidebar.selectbox(
    "Image source", ["Sample images", "Upload an image", "Video / webcam"]
)

# Keep the existing Ultralytics video and webcam tools available.
if source == "Video / webcam":
    from ultralytics import solutions
    inf = solutions.Inference(model=str(ROOT / "freeze_model.pt"), imgsz=1280)
    inf.inference()
    st.stop()

st.title("Traffic Sign Detection")
st.write("Choose a sample image or upload your own to detect traffic signs.")
st.caption("YOLO26n · Powered by Ultralytics and Streamlit")

image_source = None
image_name = ""
if source == "Sample images":
    samples = sorted(
        path for path in (ROOT / "samples").glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if samples:
        image_source = st.sidebar.selectbox(
            "Sample image", samples, format_func=lambda path: path.name
        )
        image_name = image_source.name
    else:
        st.sidebar.info("No sample images are available. Choose Upload an image.")
else:
    image_source = st.sidebar.file_uploader(
        "Upload image", type=["jpg", "jpeg", "png"],
        help="Choose a road photo containing traffic signs.",
    )
    if image_source is not None:
        image_name = image_source.name

confidence = st.sidebar.slider("Confidence Threshold", 0.01, 1.0, 0.25, 0.01)
iou = st.sidebar.slider(
    "IoU Threshold", 0.0, 1.0, 0.70, 0.01,
    help="Overlap threshold when NMS filtering is used. May have no effect in YOLO26 end-to-end mode.",
)
st.sidebar.selectbox("Model", ["freeze_model.pt"])
model, lock = load_model()
class_ids = list(model.names)
selected_classes = st.sidebar.multiselect(
    "Classes", class_ids, default=class_ids,
    format_func=lambda class_id: model.names[class_id],
    help="All classes are selected initially. Remove classes to limit the results.",
)
start = st.sidebar.button("Start", type="primary", width="stretch")

if image_source is None:
    st.info("Upload an image or choose Sample images from the sidebar.")
    st.stop()

try:
    with Image.open(image_source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
except (UnidentifiedImageError, OSError):
    st.error("This image could not be opened. Please choose another JPG or PNG file.")
    st.stop()

st.subheader(image_name)
original, predicted = st.columns(2)
original.image(image, caption="Original image", width="stretch")

if start:
    if not selected_classes:
        st.warning("Select at least one sign class before clicking Start.")
    else:
        with st.spinner("Detecting traffic signs..."):
            with lock:
                result = model.predict(
                    source=image, imgsz=1280, conf=confidence, iou=iou,
                    classes=selected_classes, device="cpu", verbose=False,
                )[0]
                annotated = result.plot()
        predicted.image(annotated, channels="BGR", caption="Predicted image", width="stretch")
        count = len(result.boxes)
        if count:
            st.success(f"Detected {count} sign(s).")
        else:
            st.info("No signs detected with the selected classes and confidence threshold.")
else:
    predicted.info("Click Start to see the detections.")
