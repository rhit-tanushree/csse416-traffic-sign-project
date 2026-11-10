from pathlib import Path
from threading import Lock

import streamlit as st
from ultralytics import YOLO, solutions

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_sample_model():
    return YOLO(str(ROOT / "freeze_model.pt")), Lock()


try_sample = st.sidebar.checkbox("Try a sample image", value=False)

if try_sample:
    st.title("Traffic Sign Detection")
    st.write("Choose a sample road image and click Detect signs.")

    samples = sorted(
        path for path in (ROOT / "samples").glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not samples:
        st.info("Sample images are not available yet. Uncheck Try a sample image to upload your own.")
        st.stop()

    selected = st.selectbox("Sample image", samples, format_func=lambda path: path.name)
    confidence = st.sidebar.slider("Confidence threshold", 0.01, 1.0, 0.25, 0.01)
    st.caption("All model classes are included. Lower confidence can show more detections and more false positives.")

    original, predicted = st.columns(2)
    original.image(str(selected), caption="Original image", width="stretch")

    if st.button("Detect signs", type="primary"):
        with st.spinner("Detecting traffic signs..."):
            model, lock = load_sample_model()
            with lock:
                result = model.predict(
                    source=str(selected), imgsz=1280, conf=confidence,
                    device="cpu", verbose=False,
                )[0]
                annotated = result.plot()
        predicted.image(annotated, channels="BGR", caption="Predicted image", width="stretch")
        count = len(result.boxes)
        if count:
            st.success(f"Detected {count} sign(s).")
        else:
            st.info("No signs detected at this confidence threshold.")
else:
    inf = solutions.Inference(
        model=str(ROOT / "freeze_model.pt"),
        imgsz=1280,
    )
    inf.inference()
