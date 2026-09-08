"""
Does 99% Mean Anything?  —  an audit of a MedSigLIP chest X-ray classifier.
ACT-Africa 2026 Hackathon · Group 2, Health.

Ships the real 312 KB quantized classifier and runs it live in-session.
MedSigLIP itself (~3.5 GB) does not fit in Streamlit Cloud's memory, so image
embeddings are precomputed; everything downstream of them is real.
"""

import json
import os
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

st.set_page_config(page_title="Does 99% Mean Anything?", page_icon="🫁",
                   layout="wide", initial_sidebar_state="collapsed")

INK, GOOD, WARN, BAD, MUTE = "#eaeef4", "#33a094", "#e8a33d", "#d9614a", "#7d8794"
LINE, PANEL = "#212936", "#141a23"

st.markdown(f"""
<style>
  .block-container {{padding-top: 3rem; padding-bottom: 4rem; max-width: 1500px;}}
  #MainMenu, footer {{visibility: hidden;}}

  .hero {{padding: .5rem 0 1rem;}}
  .hero h1 {{font-size: 2.75rem; font-weight: 800; letter-spacing: -.035em;
             line-height: 1.2; margin: 0 !important;
             padding: .06em 0 !important;}}
  .hero h1.grad {{
      background-image: linear-gradient(96deg,#e8a33d 0%,#f4d49b 55%,#e8a33d 100%);
      -webkit-background-clip: text; background-clip: text;
      -webkit-text-fill-color: transparent; color: transparent !important;}}
  .sub {{font-size: 1.05rem; color: #aeb8c6; max-width: 82ch; line-height: 1.55;
         margin-top: .6rem;}}
  .chips {{margin-top: .8rem; display: flex; flex-wrap: wrap; gap: .4rem;}}
  .chip {{font-size: .74rem; letter-spacing: .05em; text-transform: uppercase;
          color: #9aa4b2; border: 1px solid {LINE}; border-radius: 999px;
          padding: .3rem .72rem; background: rgba(255,255,255,.02);}}
  .chip.live {{color: {GOOD}; border-color: rgba(51,160,148,.45);}}

  .eyebrow {{font-size: .71rem; letter-spacing: .15em; text-transform: uppercase;
             color: {MUTE}; font-weight: 700; margin: .2rem 0 .45rem;}}
  .lede {{font-size: 1.02rem; line-height: 1.62; color: #aeb8c6; max-width: 72ch;}}
  .cap {{font-size: .84rem; color: {MUTE}; line-height: 1.5; margin-top: .5rem;}}

  .stat {{font-size: 2.75rem; font-weight: 800; line-height: 1; letter-spacing: -.035em;}}
  .statlab {{font-size: .72rem; text-transform: uppercase; letter-spacing: .1em;
             color: {MUTE}; margin-top: .34rem; line-height: 1.4;}}

  .verdict {{border-left: 3px solid #e8a33d; padding: .95rem 1.15rem;
             margin: 1.1rem 0 .3rem; background: rgba(232,163,61,.075);
             border-radius: 0 10px 10px 0; font-size: 1.02rem; line-height: 1.6;}}
  .verdict.good {{border-left-color: {GOOD}; background: rgba(51,160,148,.075);}}
  .big {{font-size: 1.3rem; font-weight: 600; line-height: 1.5;}}
  .quiet {{color: {MUTE}; font-size: .85rem;}}

  div[data-testid="stImage"] img {{border-radius: 12px; border: 1px solid {LINE};
                                   background: #fff;}}
  .stTabs [data-baseweb="tab-list"] {{gap: .3rem; border-bottom: 1px solid {LINE};}}
  .stTabs [data-baseweb="tab"] {{height: 46px; padding: 0 1rem; font-weight: 600;
                                 font-size: .95rem;}}
  .stButton button {{border-radius: 10px; font-weight: 600;}}
  hr {{border-color: {LINE};}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- loading
@st.cache_data(show_spinner=False)
def load_meta():
    with open(os.path.join(ASSETS, "meta.json")) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_arrays():
    z = np.load(os.path.join(ASSETS, "results.npz"))
    d = {k: z[k] for k in z.files}
    d["X_test"] = d["X_test"].astype(np.float32)
    d["y_test"] = d["y_test"].astype(np.int64)
    return d


@st.cache_resource(show_spinner=False)
def load_interpreter():
    p = os.path.join(ASSETS, "cxr_classifier_quant.tflite")
    if not os.path.exists(p):
        return None, "no file"
    blob = open(p, "rb").read()
    for mod, name in (("ai_edge_litert.interpreter", "ai-edge-litert"),
                      ("tflite_runtime.interpreter", "tflite-runtime"),
                      ("tensorflow.lite", "tensorflow")):
        try:
            Interpreter = __import__(mod, fromlist=["Interpreter"]).Interpreter
        except ImportError:
            continue
        itp = Interpreter(model_content=blob)
        itp.allocate_tensors()
        return itp, name
    return None, "no runtime"


def infer(itp, emb):
    di, do = itp.get_input_details()[0], itp.get_output_details()[0]
    itp.resize_tensor_input(di["index"], [1, emb.shape[0]])
    itp.allocate_tensors()
    itp.set_tensor(di["index"], emb[None].astype(np.float32))
    t0 = time.perf_counter()
    itp.invoke()
    return itp.get_tensor(do["index"])[0], (time.perf_counter() - t0) * 1000


if not os.path.exists(os.path.join(ASSETS, "meta.json")):
    st.title("Assets not found")
    st.write("Unzip `cxr_app_assets.zip` into `assets/` so `assets/meta.json` exists.")
    st.stop()

M, A = load_meta(), load_arrays()
C = M["class_names"]
y_test, y_prob = A["y_test"], A["y_prob"]
pred, conf = y_prob.argmax(1), y_prob.max(1)
errors = np.where(pred != y_test)[0]
itp, backend = load_interpreter()
SC, LK = M["shortcut"], M["leakage"]
BROKEN = SC["lowres"] > 0.6


# ---------------------------------------------------------------- ui atoms
def stat(col, value, label, color=INK):
    col.markdown(f"<div class='stat' style='color:{color}'>{value}</div>"
                 f"<div class='statlab'>{label}</div>", unsafe_allow_html=True)


def eyebrow(txt):
    st.markdown(f"<div class='eyebrow'>{txt}</div>", unsafe_allow_html=True)


def lede(txt):
    st.markdown(f"<p class='lede'>{txt}</p>", unsafe_allow_html=True)


def cap(txt):
    st.markdown(f"<div class='cap'>{txt}</div>", unsafe_allow_html=True)


def verdict(txt, good=False):
    st.markdown(f"<div class='verdict{' good' if good else ''}'>{txt}</div>",
                unsafe_allow_html=True)


def _img(obj, width=None):
    """st.image, tolerant of Streamlit versions with different width kwargs."""
    if width is not None:
        return st.image(obj, width=width)
    for kw in ({"use_container_width": True}, {"use_column_width": True}, {}):
        try:
            return st.image(obj, **kw)
        except TypeError:
            continue


def figure(name, title=None, caption=None, cell=None):
    p = os.path.join(ASSETS, "plots", name)
    if title:
        eyebrow(title)
    if os.path.exists(p):
        _img(p)
        if caption:
            cap(caption)
        return True
    st.markdown(
        f"<div style='border:1px dashed {LINE};border-radius:12px;padding:2.6rem 1rem;"
        f"text-align:center;color:{MUTE};font-size:.86rem'>"
        f"<code>plots/{name}</code> not in the bundle"
        + (f"<br>run {cell} in the notebook, then re-run the export cell" if cell else "")
        + "</div>", unsafe_allow_html=True)
    return False


def hbars(labels, values, kinds=None, domain=None, rng=None, height=220,
          fmt=".3f", xtitle=None, pct_axis=False, xmax=None):
    df = pd.DataFrame({"label": labels, "value": [float(v) for v in values],
                       "kind": kinds or ["a"] * len(labels)})
    ax = {"title": xtitle, "gridColor": "#1e2530", "tickCount": 5}
    if pct_axis:
        ax["format"] = "%"
    x = alt.X("value:Q", axis=alt.Axis(**ax))
    if xmax is not None:
        x = x.scale(domain=[0, xmax])
    color = (alt.Color("kind:N", legend=None,
                       scale=alt.Scale(domain=domain, range=rng))
             if kinds else alt.value(GOOD))
    bar = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=22).encode(
        y=alt.Y("label:N", sort=None, axis=alt.Axis(title=None, labelLimit=220,
                                                    labelFontSize=13)),
        x=x, color=color,
        tooltip=["label", alt.Tooltip("value:Q", format=".4f")])
    txt = alt.Chart(df).mark_text(align="left", dx=7, color=INK, fontWeight="bold",
                                  fontSize=13).encode(
        y=alt.Y("label:N", sort=None), x="value:Q",
        text=alt.Text("value:Q", format=fmt))
    return (bar + txt).properties(height=height).configure_view(strokeWidth=0)


def prob_chart(values, labels, true_idx=None, height=180, fmt=".3f"):
    kinds = ["true" if (true_idx is not None and k == true_idx) else "other"
             for k in range(len(labels))]
    return hbars(labels, values, kinds, ["true", "other"], [GOOD, "#39424f"],
                 height=height, fmt=fmt)


def thumb(i, size=None):
    p = os.path.join(ASSETS, "thumbs", f"{i}.jpg")
    if not os.path.exists(p):
        return None
    im = Image.open(p).convert("L")
    return im.resize((size, size), Image.BILINEAR) if size else im


# ---------------------------------------------------------------- hero
st.markdown(f"""
<div class="hero">
  <h1>Our model scores {M['probe_acc']:.1%}.</h1>
  <h1 class="grad">{'So does a 16×16 blur.' if BROKEN
                    else f'A 16×16 blur gets {SC["lowres"]:.1%}.'}</h1>
  <div class="sub">We froze Google's MedSigLIP, embedded every chest X-ray once, and
  trained a small head on {M['n_train']:,} of them. It scores almost perfectly.
  This is our audit of whether that number means anything.</div>
  <div class="chips">
    <span class="chip">{M['model']}</span>
    <span class="chip">{M['n_test']:,} held-out images</span>
    <span class="chip">{M['embedding_dim']} dimensions</span>
    <span class="chip">0 foundation weights trained</span>
    <span class="chip {'live' if itp else ''}">
      {'classifier running live · ' + backend if itp else 'precomputed'}</span>
  </div>
</div>
""", unsafe_allow_html=True)

T = st.tabs(["  Overview  ", "  The audit  ", "  Zero-shot  ", "  Calibration  ",
             "  Try it yourself  ", "  Explore  ", "  Deployment  "])


# ================================================================== overview
with T[0]:
    c = st.columns(4)
    stat(c[0], f"{M['probe_acc']:.1%}", "test accuracy", GOOD)
    stat(c[1], f"{SC['lowres']:.1%}", "accuracy from a 16×16 blur",
         WARN if BROKEN else GOOD)
    stat(c[2], f"{M['zeroshot_acc']:.1%}", "zero-shot, no training at all", WARN)
    stat(c[3], f"{M['n_wrong']}", f"mistakes out of {M['n_test']:,}")
    st.write("")

    a, b = st.columns([1.25, 1])
    with a:
        eyebrow("What survives when we destroy the image")
        st.altair_chart(
            hbars(["MedSigLIP + trained head", "16×16 blur", "Border only, no lungs",
                   "5 metadata numbers", "Chance"],
                  [SC["full"], SC["lowres"], SC["border"], SC["meta"], SC["chance"]],
                  ["model", "degraded", "degraded", "degraded", "chance"],
                  ["model", "degraded", "chance"], [GOOD, WARN, "#39424f"],
                  height=250, xtitle="test accuracy", pct_axis=True, xmax=1.08),
            use_container_width=True)
        if BROKEN:
            verdict(f"<b>A 16×16 thumbnail reaches {SC['lowres']:.1%}.</b> At that "
                    "size no lung structure is visible at all — so the classes are "
                    "separable from acquisition artifacts alone. The model is "
                    "reading which source archive an image came from, not whether "
                    "the patient is sick.")
        else:
            verdict(f"A 16×16 blur reaches only {SC['lowres']:.1%} against "
                    f"{SC['full']:.1%} for the full model. The signal is in the "
                    "anatomy, not in acquisition artifacts. This result survives "
                    "the audit.", good=True)
    with b:
        figure("class_samples.png", "The four classes",
               "One held-out example from each class.", cell="Cell 49")

    st.write("")
    st.markdown("---")
    eyebrow("The standard results, for completeness")
    g1, g2 = st.columns(2)
    with g1:
        figure("confusion_matrix.png", "Confusion matrix")
        st.write("")
        figure("roc_curve.png", "ROC, one-vs-rest")
    with g2:
        figure("tsne_embeddings.png", "Embedding space before any training",
               "t-SNE of the frozen MedSigLIP embeddings. The classes separate "
               "before our classifier exists — which is the first hint that this "
               "task is easier than it should be.")
        st.write("")
        figure("precision_recall_curve.png", "Precision–recall")

    with st.expander("Method, and what we are not claiming"):
        st.markdown(f"""
**Data** — {M['dataset']}. {M['n_train']:,} training, {M['n_test']:,} held out,
{len(C)} balanced classes. An 80/20 stratified split of the training archive gave a
validation set.

**Model** — `{M['model']}`, frozen. Every image embedded once into
{M['embedding_dim']} dimensions and cached; a small Keras head trained on top. No
foundation-model weight is ever updated.

**Zero-shot** — the text tower embeds one sentence per class; predictions are the
argmax of cosine similarity between image and text embeddings. SigLIP requires
`padding="max_length"` in the processor.

**Shortcut audit** — logistic regression on the same split using degraded inputs: a
16×16 bilinear downsample, the outer border of a 32×32 downsample with the lung
fields masked out, and five scalar image statistics.

**Limitations** — one public dataset, one split, no external validation, no
prospective clinical data, no radiologist adjudication of the labels. Nothing here
is evidence of clinical utility, and the audit is the reason we say so.

*cf. DeGrave, Janizek & Lee, "AI for radiographic COVID-19 detection selects
shortcuts over signal", Nature Machine Intelligence, 2021.*
""")


# ================================================================== audit
with T[1]:
    lede("Two ways a medical benchmark result can be hollow: the test set overlaps "
         "the training set, or the classes are separable from something that isn't "
         "the disease. We checked both.")
    st.write("")

    eyebrow("Audit 1 — did the test set leak into training?")
    c = st.columns(3)
    stat(c[0], f"{LK['median_nn_sim']:.3f}", "median similarity to nearest training image")
    stat(c[1], f"{LK['frac_above_098']:.1%}", "test images above 0.98 similarity",
         BAD if LK["frac_above_098"] > 0.02 else GOOD)
    stat(c[2], f"{LK['nn_same_class']:.1%}", "nearest neighbour shares the class")
    st.write("")
    la, lb = st.columns([1.5, 1])
    with la:
        figure("leakage_pairs.png", None,
               "The four test images most similar to anything in training, with "
               "their nearest training neighbour below.")
    with lb:
        if LK["frac_above_098"] > 0.02:
            verdict("Near-duplicates are present across the split. Part of this "
                    "accuracy is memorisation.")
        else:
            verdict("Clean — the test images are genuinely unseen. That rules out "
                    "the easy explanation, and makes the next test the important "
                    "one.", good=True)

    st.markdown("---")
    eyebrow("Audit 2 — do you even need the lungs?")
    lede("Drag the slider. Watch the anatomy disappear. A classifier trained on "
         f"images at 16×16 still scores <b>{SC['lowres']:.1%}</b>.")
    st.write("")

    ca, cb = st.columns([1, 2.1])
    with ca:
        cls = st.selectbox("Class", C, key="bl_c")
        pool = np.where(y_test == C.index(cls))[0]
        k = st.slider("Image", 0, max(0, len(pool) - 1), 0, key="bl_i")
        i = int(pool[k])
        res = st.select_slider("Resolution", options=[224, 128, 64, 32, 16, 8],
                               value=16, key="bl_r")
        st.markdown(f"<div class='quiet'>{res}×{res} = {res*res:,} pixels, "
                    f"{100*res*res/(224*224):.1f}% of the original</div>",
                    unsafe_allow_html=True)
    with cb:
        im = thumb(i)
        if im is None:
            st.info("No thumbnails in this bundle.")
        else:
            x, y = st.columns(2)
            with x:
                _img(im.resize((420, 420), Image.BILINEAR), width=400)
                cap("original")
            with y:
                _img(im.resize((res, res), Image.BILINEAR)
                       .resize((420, 420), Image.NEAREST), width=400)
                cap(f"{res}×{res}")

    if res <= 16 and BROKEN:
        verdict(f"This is what a classifier scoring <b>{SC['lowres']:.1%}</b> sees. "
                "Nothing in this image is anatomy. It is separating the classes on "
                "brightness, framing and contrast — properties of the archive each "
                "class was collected from.")
    st.write("")
    figure("shortcut_audit.png", "The full comparison")

    st.markdown("---")
    eyebrow("Audit 3 — where does the model look?")
    ga, gb = st.columns([1.6, 1])
    with ga:
        figure("gradcam_example.png", None)
    with gb:
        lede("Grad-CAM back through the frozen vision encoder to the patch tokens. "
             "Heat on markers, borders or text rather than lung fields is the same "
             "finding, seen a second way.<br><br>Saliency is suggestive, not proof — "
             "which is exactly why we ran the quantitative test as well. Two "
             "independent methods, one conclusion.")


# ================================================================== zero-shot
with T[2]:
    lede("MedSigLIP has two towers — one reads images, one reads text. The workshop "
         "pipeline only ever calls the image one. Give it four sentences instead of "
         f"{M['n_train']:,} labelled examples and it classifies X-rays with no "
         "training at all.")
    st.write("")
    c = st.columns(3)
    stat(c[0], f"{M['zeroshot_acc']:.1%}", "zero-shot · 0 training images", WARN)
    stat(c[1], f"{M['probe_acc']:.1%}", f"trained head · {M['n_train']:,} images", GOOD)
    stat(c[2], f"{M['probe_acc'] - M['zeroshot_acc']:+.1%}",
         "what the labels bought us")
    st.write("")

    za, zb = st.columns([1, 1])
    with za:
        eyebrow("The four sentences")
        st.dataframe(pd.DataFrame([{"class": k, "prompt": v}
                                   for k, v in M["prompts"].items()]),
                     hide_index=True, use_container_width=True)
        ps = M.get("prompt_sensitivity") or {}
        if ps:
            st.write("")
            eyebrow("Same weights, same images — only the sentence changed")
            st.altair_chart(hbars(list(ps.keys()), list(ps.values()), height=170,
                                  xtitle="zero-shot accuracy", pct_axis=True,
                                  xmax=max(ps.values()) * 1.25),
                            use_container_width=True)
            lo, hi = min(ps.values()), max(ps.values())
            verdict(f"Wording alone moves accuracy by {hi - lo:.1%}. Any zero-shot "
                    "medical number quoted without its prompt is not a reproducible "
                    "result.")
    with zb:
        figure("alignment_heatmap.png", "Image classes against text prompts",
               M.get("align_note", ""))


# ================================================================== calibration
with T[3]:
    lede("In a clinic with no radiologist to overrule it, confidence is the whole "
         "safety mechanism. Accuracy says how often the model is right. Calibration "
         "says whether it can be trusted to ask for help.")
    st.write("")
    c = st.columns(4)
    stat(c[0], f"{M['ece']:.3f}", "expected calibration error",
         GOOD if M["ece"] < 0.05 else WARN)
    stat(c[1], f"{M['n_overconfident']}/{M['n_wrong']}",
         "errors made above 90% confidence", BAD if M["n_overconfident"] else GOOD)
    stat(c[2], f"{M['mean_conf_right']:.3f}", "mean confidence when right")
    stat(c[3], "—" if M["n_wrong"] == 0 else f"{M['mean_conf_wrong']:.3f}",
         "mean confidence when wrong", WARN)
    st.write("")
    figure("calibration.png", None)

    if M["n_overconfident"]:
        verdict(f"{M['n_overconfident']} of {M['n_wrong']} mistakes were made above "
                "90% confidence. A confidence threshold would not have caught them — "
                "they reach the patient unflagged. That is the number a deployment "
                "review should ask for, and it is not the accuracy.")

    st.markdown("---")
    eyebrow("What a confidence threshold actually buys you")
    t = st.slider("Refuse to answer below this confidence", 0.25, 1.0, 0.90, 0.01)
    keep = conf >= t
    nk = int(keep.sum())
    acc_k = float((pred[keep] == y_test[keep]).mean()) if nk else float("nan")
    slipped = int(((pred != y_test) & keep).sum())
    s = st.columns(4)
    stat(s[0], f"{keep.mean():.1%}", "answered automatically")
    stat(s[1], f"{acc_k:.2%}" if nk else "—", "accuracy on those", GOOD)
    stat(s[2], f"{slipped}", "wrong answers still getting through",
         BAD if slipped else GOOD)
    stat(s[3], f"{len(y_test) - nk}", "escalated to a human", WARN)
    st.write("")

    rows = []
    for th in np.linspace(0.25, 1.0, 70):
        m = conf >= th
        if m.sum() == 0:
            continue
        rows.append({"threshold": float(th), "coverage": float(m.mean()),
                     "accuracy on answered": float((pred[m] == y_test[m]).mean())})
    cv = pd.DataFrame(rows).melt("threshold", var_name="series", value_name="value")
    line = alt.Chart(cv).mark_line(strokeWidth=2.6).encode(
        x=alt.X("threshold:Q", axis=alt.Axis(title="confidence threshold",
                                             gridColor="#1e2530")),
        y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1.02]),
                axis=alt.Axis(title=None, format="%", gridColor="#1e2530")),
        color=alt.Color("series:N", scale=alt.Scale(range=[GOOD, WARN]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["series", alt.Tooltip("threshold:Q", format=".2f"),
                 alt.Tooltip("value:Q", format=".3f")])
    rule = alt.Chart(pd.DataFrame({"t": [t]})).mark_rule(
        color=INK, strokeDash=[4, 4]).encode(x="t:Q")
    st.altair_chart((line + rule).properties(height=300)
                    .configure_view(strokeWidth=0), use_container_width=True)

    st.markdown("---")
    figure("failure_gallery.png", "Every mistake the model made",
           "Sorted by confidence — the most confidently wrong first.", cell="Cell 46")


# ================================================================== game
with T[4]:
    lede("Four classes, one X-ray, no medical training required. See how you do — "
         "then see what the model said, with the real quantized classifier running "
         "live on that image's embedding.")
    st.write("")

    if "g_i" not in st.session_state:
        st.session_state.update(g_i=int(np.random.default_rng(0).integers(len(y_test))),
                                g_you=0, g_mod=0, g_n=0, g_ans=None)
    i = st.session_state.g_i
    gl, gr = st.columns([1, 1.15])

    with gl:
        im = thumb(i, 460)
        if im is not None:
            _img(im, width=440)
        else:
            st.info("No thumbnails in this bundle.")

    with gr:
        if st.session_state.g_ans is None:
            eyebrow("Your call")
            cols = st.columns(2)
            for k, name in enumerate(C):
                if cols[k % 2].button(name, key=f"g{k}", use_container_width=True):
                    st.session_state.g_ans = k
                    st.session_state.g_n += 1
                    st.session_state.g_you += int(k == y_test[i])
                    st.session_state.g_mod += int(pred[i] == y_test[i])
                    st.rerun()
            st.markdown("<div class='quiet' style='margin-top:1rem'>Most people find "
                        "this near-impossible. That is the point — and the model "
                        f"does it at {M['probe_acc']:.0%}.</div>",
                        unsafe_allow_html=True)
        else:
            k, truth, said = st.session_state.g_ans, int(y_test[i]), int(pred[i])
            st.markdown(
                f"<div class='big'>"
                f"You said <b>{C[k]}</b> — "
                f"<span style='color:{GOOD if k == truth else BAD}'>"
                f"{'correct' if k == truth else 'wrong'}</span><br>"
                f"Model said <b>{C[said]}</b> at {conf[i]:.0%} — "
                f"<span style='color:{GOOD if said == truth else BAD}'>"
                f"{'correct' if said == truth else 'wrong'}</span><br>"
                f"Truth: <b style='color:{GOOD}'>{C[truth]}</b></div>",
                unsafe_allow_html=True)
            st.write("")
            if itp is not None:
                probs, ms = infer(itp, A["X_test"][i])
                st.markdown(f"<div class='quiet'>live inference · quantized TFLite · "
                            f"{ms:.2f} ms</div>", unsafe_allow_html=True)
            else:
                probs = A["y_prob_q"][i]
            st.altair_chart(prob_chart(probs, C, truth, height=160),
                            use_container_width=True)
            if st.button("Next image →", type="primary"):
                st.session_state.g_i = int(np.random.default_rng(
                    st.session_state.g_n * 7 + 13).integers(len(y_test)))
                st.session_state.g_ans = None
                st.rerun()

    if st.session_state.g_n:
        st.write("")
        s = st.columns(4)
        stat(s[0], f"{st.session_state.g_you}/{st.session_state.g_n}", "you", WARN)
        stat(s[1], f"{st.session_state.g_mod}/{st.session_state.g_n}",
             "the model on the same images", GOOD)
        stat(s[2], f"{st.session_state.g_you/st.session_state.g_n:.0%}", "your accuracy")
        stat(s[3], f"{M['probe_acc']:.0%}", "its accuracy on all 600")


# ================================================================== explore
with T[5]:
    view = st.radio("View", ["Browse every image", "Embedding map"],
                    horizontal=True, label_visibility="collapsed")
    st.write("")

    if view == "Browse every image":
        f1, f2, f3 = st.columns([1, 1, 2])
        only = f1.selectbox("Show", ["Mistakes only", "All images", "Correct only"])
        cls = f2.selectbox("True class", ["Any"] + C, key="ex_c")
        idxs = (errors if only == "Mistakes only" else
                np.where(pred == y_test)[0] if only == "Correct only" else
                np.arange(len(y_test)))
        if cls != "Any":
            idxs = idxs[y_test[idxs] == C.index(cls)]
        if len(idxs) == 0:
            st.warning("Nothing matches that filter.")
        else:
            idxs = idxs[np.argsort(-conf[idxs])]
            labels = [f"#{j} · {C[y_test[j]]} → {C[pred[j]]} ({conf[j]:.0%})"
                      for j in idxs]
            pickd = f3.selectbox(f"Image  ({len(idxs)} matching)", labels)
            i = int(idxs[labels.index(pickd)])
            st.write("")
            a, b = st.columns([1, 1.35])
            with a:
                im = thumb(i, 400)
                if im is not None:
                    _img(im, width=380)
                ok = pred[i] == y_test[i]
                st.markdown(
                    f"**True** {C[y_test[i]]} &nbsp;·&nbsp; **said** {C[pred[i]]} "
                    f"{'✓' if ok else '✗'}<br>"
                    f"<span class='quiet'>confidence {conf[i]:.1%} · nearest "
                    f"training image {A['nn_sim'][i]:.3f}</span>",
                    unsafe_allow_html=True)
            with b:
                if itp is not None:
                    probs, ms = infer(itp, A["X_test"][i])
                    note = (" · quantized head disagrees with float32"
                            if int(probs.argmax()) != int(pred[i]) else "")
                    eyebrow(f"Trained head · live · {ms:.2f} ms{note}")
                else:
                    probs = A["y_prob_q"][i]
                    eyebrow("Trained head")
                st.altair_chart(prob_chart(probs, C, int(y_test[i]), height=155),
                                use_container_width=True)
                eyebrow("Zero-shot on the same image — cosine similarity, nothing trained")
                st.altair_chart(prob_chart(A["zs_scores"][i], C, int(y_test[i]),
                                           height=145),
                                use_container_width=True)
            if M.get("confusions"):
                st.markdown("---")
                eyebrow("Which confusions happen")
                st.dataframe(pd.DataFrame(M["confusions"],
                                          columns=["true", "predicted", "count"]),
                             hide_index=True, use_container_width=True)
    else:
        lede("Every held-out image, projected from 1,152 dimensions to two. These "
             "are the frozen embeddings — no training involved. Hover a point; the "
             "crosses are the model's mistakes.")
        e = A["emb2d"]
        d = pd.DataFrame({"x": e[:, 0], "y": e[:, 1],
                          "class": [C[k] for k in y_test],
                          "predicted": [C[k] for k in pred],
                          "confidence": conf, "index": np.arange(len(y_test)),
                          "outcome": np.where(pred == y_test, "correct", "wrong")})
        pts = alt.Chart(d).mark_circle(size=58, opacity=.75).transform_filter(
            alt.datum.outcome == "correct").encode(
            x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
            color=alt.Color("class:N", scale=alt.Scale(scheme="tableau10"),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["index", "class", "predicted",
                     alt.Tooltip("confidence:Q", format=".3f")])
        bad = alt.Chart(d).mark_point(shape="cross", size=200, strokeWidth=3,
                                      color=BAD).transform_filter(
            alt.datum.outcome == "wrong").encode(
            x="x:Q", y="y:Q",
            tooltip=["index", "class", "predicted",
                     alt.Tooltip("confidence:Q", format=".3f")])
        st.altair_chart((pts + bad).properties(height=520)
                        .configure_view(strokeWidth=0), use_container_width=True)


# ================================================================== deploy
with T[6]:
    sz, lat = M["sizes_kb"], M.get("latency_ms", {})
    lede("The foundation model stays frozen on a server. What has to ship to the "
         "edge is the classifier head — and after dynamic-range quantization it is a "
         "third of a megabyte.")
    st.write("")
    c = st.columns(4)
    stat(c[0], f"{sz['keras']/1024:.1f} MB", "Keras checkpoint")
    stat(c[1], f"{sz['tflite_int8']:.0f} KB", "quantized TFLite", GOOD)
    stat(c[2], f"{sz['keras']/sz['tflite_int8']:.0f}×", "smaller")
    stat(c[3], f"{M['quant_acc'] - M['probe_acc']:+.2%}", "accuracy change",
         GOOD if abs(M["quant_acc"] - M["probe_acc"]) < 0.005 else WARN)
    st.write("")

    da, db = st.columns([1.15, 1])
    with da:
        eyebrow("Size of each artifact")
        st.altair_chart(
            hbars(["Keras float32", "TFLite float32", "TFLite int8"],
                  [sz["keras"], sz["tflite_f32"], sz["tflite_int8"]],
                  ["a", "a", "b"], ["a", "b"], ["#39424f", GOOD],
                  height=170, fmt=",.0f", xtitle="kilobytes"),
            use_container_width=True)
    with db:
        eyebrow("The numbers")
        tbl = pd.DataFrame([
            {"artifact": "Keras float32", "size (KB)": f"{sz['keras']:,.1f}",
             "accuracy": f"{M['probe_acc']:.4f}",
             "ms / image": f"{lat.get('f32_ms', float('nan')):.3f}"},
            {"artifact": "TFLite float32", "size (KB)": f"{sz['tflite_f32']:,.1f}",
             "accuracy": f"{M['probe_acc']:.4f}",
             "ms / image": f"{lat.get('f32_ms', float('nan')):.3f}"},
            {"artifact": "TFLite int8", "size (KB)": f"{sz['tflite_int8']:,.1f}",
             "accuracy": f"{M['quant_acc']:.4f}",
             "ms / image": f"{lat.get('int8_ms', float('nan')):.3f}"}])
        st.dataframe(tbl, hide_index=True, use_container_width=True)

    verdict("The honest caveat: the embedding still has to come from somewhere. This "
            "is an edge <i>classifier</i>, not an edge <i>pipeline</i> — the "
            "880M-parameter vision encoder is the part that would actually need "
            "solving.")

    if itp is not None:
        st.markdown("---")
        eyebrow("Benchmark it on this machine, right now")
        n = st.slider("Images", 50, len(y_test), min(300, len(y_test)), step=50)
        if st.button("Run the quantized model", type="primary"):
            sub = A["X_test"][:n]
            t0 = time.perf_counter()
            out = np.stack([infer(itp, r)[0] for r in sub])
            dt = (time.perf_counter() - t0) / n * 1000
            r = st.columns(3)
            stat(r[0], f"{dt:.2f} ms", "per image, this CPU", GOOD)
            stat(r[1], f"{1000/dt:,.0f}", "images per second")
            stat(r[2], f"{(out.argmax(1) == y_test[:n]).mean():.1%}",
                 f"accuracy on the first {n}")