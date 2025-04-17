from ultralytics import FastSAM


if __name__ == "__main__":
    # Load the YOLO11 model
    model = FastSAM("FastSAM-s.pt")

    # Export the model to TensorRT format
    model.export(
        format="engine",
        imgsz=512,
        half=True,
        dynamic=True,
        nms=True
    )