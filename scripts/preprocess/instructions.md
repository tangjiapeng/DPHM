# Installation
MAYBE NEED __RECURSE SUBMODULE
incase you get an error bc. of np.int, check which numpy version is actually used. I had a case where two numpy version were installed and it was using 1.26 instead of 1.23.


## Facial Landmark Detection
### PiPNet Landmark Detection

Download the implementation of [PIPNet](https://github.com/jhb86253817/PIPNet), and the move our code/script under their folder.
cd ./scripts/preprocess/
git clone git@github.com:jhb86253817/PIPNet.git
```
cp pipnet_helpers/demo_seq.py  PIPNet/lib
cp pipnet_helpers/run_demo_seq.sh PIPNet/
```

```
cd ./PIPNet
mkdir snapshots
```
Then, Download the `WFLW` from [here](https://drive.google.com/drive/folders/1fz6UQR2TjGvQr4birwqVXusPp6tMAXxq), and put it under snapshots.


Run the script to detect landmarks.
```
bash run_demo_seq.sh
```

### MediaPipe Landmark Detection
cd ./scripts/preprocess/mediapipe

We can also use landmarks detected by MediaPipe. Please refer to the [setup guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python) for Python.
The original code is from the [example](https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/face_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Face_Landmarker.ipynb).
```
pip install -q mediapipe
wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Run the script to detect landmarks.
```
bash run_demo_seq.sh
```