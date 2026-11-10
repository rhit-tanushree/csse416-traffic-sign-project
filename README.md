# Traffic Sign Detection Demo

This app uses our trained YOLO26n model to detect and label traffic
signs in uploaded images. The interface uses Ultralytics and Streamlit.

## Files

- `app.py`: Starts the app and loads our model.
- `freeze_model.pt`: Our saved model after fine-tuning with 11 layers frozen.
- `requirements.txt`: Python packages needed to run the app.
- `packages.txt`: Linux libraries needed for cloud hosting.

Keep these files together in the same folder.

## Run locally

Open a terminal in this folder and install the Python packages:

```bash
python -m pip install -r requirements.txt
```

Start the app:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal.

## Use the app

1. Select **image** as the source.
2. Select `freeze_model.pt` as the model.
3. Select all classes.
4. Upload an image.
5. Start with the confidence threshold at **0.25**.
6. Click **Start** to see the predictions.

Predictions appear as boxes and labels on the image.
No boxes means no detections passed the current settings.
The confidence threshold is not the model's overall accuracy.

## Deploy online

1. Upload the app files and model to a GitHub repository.
2. Do not upload the `.venv` folder or training dataset.
3. Open [Streamlit Community Cloud](https://share.streamlit.io) and connect GitHub.
4. Create an app using this repository, the `main` branch, and `app.py` as the entrypoint.
5. Select **Python 3.11** in **Advanced settings**.
6. Deploy and test the generated public link.

## References

- [Ultralytics Streamlit example](https://docs.ultralytics.com/guides/streamlit-live-inference/)
- [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Streamlit dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)