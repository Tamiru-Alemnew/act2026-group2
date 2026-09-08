# Does 99% Mean Anything?

An audit of a MedSigLIP chest X-ray classifier.
ACT-Africa 2026 Hackathon — Group 2, Health.

We froze `google/medsiglip-448`, embedded 2,600 chest X-rays, and trained a small
head to sort them into four classes. It scores 99%. This app is about whether that
number means anything.

**Live app:** _add your Streamlit URL here_
**Notebook:** `notebook/ACT_2026_Foundation_MedSigLIP_Workshop.ipynb`

## What's in it

| Section | What it shows |
|---|---|
| The result | Accuracy, confusion matrix, t-SNE, ROC, precision–recall |
| Is 99% real? | Near-duplicate leakage audit, and whether a 16×16 blur can do the same job |
| Zero-shot | Classification with no training at all, using MedSigLIP's unused text tower |
| Calibration | Reliability diagram, ECE, and how many errors were made confidently |
| Failure explorer | Every test image, with the real quantized model run live on its embedding |
| Edge deployment | 3.6 MB → 312 KB, and what that costs |

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

`assets/` must contain the bundle produced by the notebook's export cell:

```
assets/
  meta.json
  results.npz                  # embeddings, probabilities, zero-shot scores
  cxr_classifier_quant.tflite  # 312 KB — the app loads and runs this for real
  thumbs/<index>.jpg           # one per test image
  plots/*.png                  # figures saved by the notebook
```

Run Cell 48 of the notebook, download `cxr_app_assets.zip`, and unzip it into
`assets/`.

## Why the embeddings are precomputed

MedSigLIP is ~880M parameters. Streamlit Community Cloud gives about 1 GB of RAM,
so the vision encoder cannot be loaded there. Image embeddings are therefore
computed once in Colab and shipped as a 1.4 MB float16 array. Everything downstream
— the quantized classifier, the zero-shot scoring, the latency benchmark — runs
live in the app.

This is also the honest architecture for the deployment claim we make: the edge
artifact is the *classifier*, not the whole pipeline.

## Deploying

Push to GitHub, then at share.streamlit.io point a new app at `app.py` on the main
branch. Do **not** add `tensorflow` to `requirements.txt` — it is ~600 MB and will
exhaust the free tier before the app starts. `ai-edge-litert` is the runtime that
loads `.tflite` files.

## Data

COVID-19 Radiography Database, four classes (COVID, NORMAL, PNEUMONIA, TB).
Check the dataset licence before redistributing any images; this repo ships only
low-resolution thumbnails for the failure explorer.

## Reference

DeGrave, Janizek & Lee. *AI for radiographic COVID-19 detection selects shortcuts
over signal.* Nature Machine Intelligence, 2021.
