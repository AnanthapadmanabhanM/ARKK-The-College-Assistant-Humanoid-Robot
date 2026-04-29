# Models

Place trained model files here:

| File | Description | Size |
|---|---|---|
| `emotion_model_1.h5` | VGG-16 face emotion detection model | ~60 MB |

## Note on Large Files

`.h5` model files are excluded from Git tracking by default (see `.gitignore`).

**Option 1 — Git LFS (recommended for GitHub)**
```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add models/emotion_model_1.h5
git commit -m "Add trained emotion model via LFS"
```

**Option 2 — Google Drive / cloud link**  
Upload the model to Google Drive and add the shareable link here.

**Option 3 — Release asset**  
Upload the `.h5` file as a GitHub Release asset and link it in the README.

## Re-training

Training notebooks for both models are available in `src/FaceEmotionDetection/`.
- Face emotion model: trained on [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- Speech emotion model: trained on CREMA-D, RAVDESS, SAVEE, TESS
