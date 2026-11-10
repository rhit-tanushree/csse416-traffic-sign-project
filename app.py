from ultralytics import solutions

inf = solutions.Inference(
    model="freeze_model.pt",
    imgsz=1280,
)

inf.inference()