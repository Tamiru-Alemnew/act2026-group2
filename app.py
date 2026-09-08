"""
Does 99% mean anything? — an audit of a MedSigLIP chest X-ray classifier.
ACT-Africa 2026 Hackathon, Group 2 (Health).

Ships the real quantized classifier and runs it live in-session. MedSigLIP itself
does not fit in Streamlit Cloud's memory, so image embeddings are precomputed;
everything downstream of them is real.
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

st.set_page_config(page_title="Does 99% mean anything?", page_icon="🫁",
                   layout="wide", initial_sidebar_state="collapsed")

# Palette: a radiology reading room. Cool near-black ground, film-grey mid-tones,
# one amber that only ever marks the audit thread, one teal for the model itself.
BG, PANEL, LINE = "#0d1116", "#141920", "#232b35"
INK, DIM, FAINT = "#e8ecf2", "#8d97a5", "#5f6875"
TEAL, AMBER, RED = "#3aa396", "#e0a03f", "#d4614c"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

.stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label,
.stApp button, .stApp li, .stApp td, .stApp th, .stMarkdown,
.stTabs [data-baseweb="tab"] {{
  font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}}
html {{ scroll-behavior: smooth; }}
.block-container {{ padding-top: 3rem; padding-bottom: 5rem; max-width: 1480px; }}
#MainMenu, footer {{ visibility: hidden; }}

.hero h1 {{ font-size: 2.6rem; font-weight: 700; letter-spacing: -.03em;
            line-height: 1.22; margin: 0 !important; padding: .04em 0 !important; }}
.hero h1.b {{ color: {AMBER}; }}
.sub {{ font-size: 1.02rem; color: {DIM}; max-width: 60ch; line-height: 1.6;
        margin: .7rem 0 0; }}
.chips {{ margin-top: .95rem; display: flex; flex-wrap: wrap; gap: .35rem; }}
.chip {{ font-size: .76rem; color: {DIM}; border: 1px solid {LINE};
         border-radius: 6px; padding: .25rem .6rem; }}
.chip.live {{ color: {TEAL}; border-color: rgba(58,163,150,.4); }}

.lab {{ font-size: .88rem; font-weight: 600; color: {DIM}; margin: .1rem 0 .5rem; }}
.note {{ font-size: .86rem; color: {FAINT}; line-height: 1.55; margin-top: .5rem; }}
.lede {{ font-size: 1rem; line-height: 1.62; color: {DIM}; max-width: 62ch;
         margin: 0 0 1.2rem; }}

.stat {{ font-size: 2.5rem; font-weight: 600; line-height: 1;
         letter-spacing: -.03em; font-feature-settings: "tnum" 1; }}
.statlab {{ font-size: .82rem; color: {DIM}; margin-top: .35rem; line-height: 1.4; }}

.say {{ border-left: 2px solid {AMBER}; padding: .1rem 0 .1rem 1rem;
        margin: 1.2rem 0 .4rem; font-size: 1.06rem; line-height: 1.55;
        color: {INK}; max-width: 68ch; }}
.say.ok {{ border-left-color: {TEAL}; }}
.big {{ font-size: 1.22rem; line-height: 1.6; }}

div[data-testid="stImage"] img {{
  border-radius: 10px; background: #fff; padding: 10px;
  border: 1px solid {LINE}; transition: border-color .18s ease; }}
div[data-testid="stImage"]:hover img {{ border-color: #33404f; }}

.stTabs [data-baseweb="tab-list"] {{ gap: .1rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ height: 44px; padding: 0 .95rem;
                                font-weight: 500; font-size: .95rem; }}
.stButton button {{ border-radius: 8px; font-weight: 500;
                    transition: border-color .15s ease, background .15s ease; }}
hr {{ border-color: {LINE}; margin: 2rem 0 1.6rem; }}
@media (prefers-reduced-motion: reduce) {{
  html {{ scroll-behavior: auto; }}
  * {{ transition: none !important; }} }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
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


# ---------------------------------------------------------------- atoms
def stat(col, value, label, color=INK):
    col.markdown(f"<div class='stat' style='color:{color}'>{value}</div>"
                 f"<div class='statlab'>{label}</div>", unsafe_allow_html=True)


def lab(t):
    st.markdown(f"<div class='lab'>{t}</div>", unsafe_allow_html=True)


def note(t):
    st.markdown(f"<div class='note'>{t}</div>", unsafe_allow_html=True)


def lede(t):
    st.markdown(f"<p class='lede'>{t}</p>", unsafe_allow_html=True)


def say(t, ok=False):
    st.markdown(f"<div class='say{' ok' if ok else ''}'>{t}</div>",
                unsafe_allow_html=True)


def _img(obj, width=None):
    if width is not None:
        return st.image(obj, width=width)
    for kw in ({"use_container_width": True}, {"use_column_width": True}, {}):
        try:
            return st.image(obj, **kw)
        except TypeError:
            continue


def figure(name, label=None, caption=None, cell=None):
    p = os.path.join(ASSETS, "plots", name)
    if label:
        lab(label)
    if os.path.exists(p):
        _img(p)
        if caption:
            note(caption)
        return True
    st.markdown(
        f"<div style='border:1px dashed {LINE};border-radius:10px;padding:2.4rem 1rem;"
        f"text-align:center;color:{FAINT};font-size:.85rem'>plots/{name} missing"
        + (f" — run {cell}, then re-run the export cell" if cell else "") + "</div>",
        unsafe_allow_html=True)
    return False


def hbars(labels, values, kinds=None, domain=None, rng=None, height=210,
          fmt=".3f", xtitle=None, pct=False, xmax=None):
    df = pd.DataFrame({"label": labels, "value": [float(v) for v in values],
                       "kind": kinds or ["a"] * len(labels)})
    ax = {"title": xtitle, "gridColor": "#1a212b", "tickCount": 5,
          "labelColor": FAINT, "titleColor": FAINT, "domainColor": LINE}
    if pct:
        ax["format"] = "%"
    x = alt.X("value:Q", axis=alt.Axis(**ax))
    if xmax is not None:
        x = x.scale(domain=[0, xmax])
    color = (alt.Color("kind:N", legend=None,
                       scale=alt.Scale(domain=domain, range=rng))
             if kinds else alt.value(TEAL))
    bar = alt.Chart(df).mark_bar(cornerRadiusEnd=3, height=20).encode(
        y=alt.Y("label:N", sort=None,
                axis=alt.Axis(title=None, labelLimit=230, labelFontSize=13,
                              labelColor=INK, domainColor=LINE, tickColor=LINE)),
        x=x, color=color,
        tooltip=["label", alt.Tooltip("value:Q", format=".4f")])
    txt = alt.Chart(df).mark_text(align="left", dx=7, color=INK, fontWeight=600,
                                  fontSize=13).encode(
        y=alt.Y("label:N", sort=None), x="value:Q",
        text=alt.Text("value:Q", format=fmt))
    return (bar + txt).properties(height=height).configure_view(strokeWidth=0)


def prob_chart(values, labels, true_idx=None, height=170, fmt=".3f"):
    kinds = ["true" if (true_idx is not None and k == true_idx) else "other"
             for k in range(len(labels))]
    return hbars(labels, values, kinds, ["true", "other"], [TEAL, "#333c48"],
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
  <h1>Our model reads chest X-rays at {M['probe_acc']:.1%}.</h1>
  <h1 class="b">{'So does a 16×16 blur.' if BROKEN
                 else f'A 16×16 blur manages {SC["lowres"]:.1%}.'}</h1>
  <p class="sub">MedSigLIP, frozen. A small head trained on {M['n_train']:,} images.
  Then an audit of whether the number means anything.</p>
  <div class="chips">
    <span class="chip">{M['n_test']:,} held out</span>
    <span class="chip">{M['embedding_dim']}-d embeddings</span>
    <span class="chip">0 foundation weights trained</span>
    <span class="chip {'live' if itp else ''}">
      {'classifier live, ' + backend if itp else 'precomputed'}</span>
  </div>
</div>
""", unsafe_allow_html=True)
st.write("")

T = st.tabs(["  Result  ", "  The audit  ", "  Zero-shot  ", "  Calibration  ",
             "  Try it  ", "  Explore  ", "  Deployment  "])


# ================================================================== result
with T[0]:
    c = st.columns(4)
    stat(c[0], f"{M['probe_acc']:.1%}", "test accuracy", TEAL)
    stat(c[1], f"{SC['lowres']:.1%}", "from a 16×16 blur", AMBER if BROKEN else TEAL)
    stat(c[2], f"{M['zeroshot_acc']:.1%}", "zero-shot, untrained", AMBER)
    stat(c[3], f"{M['n_wrong']}", f"mistakes in {M['n_test']:,}")
    st.write("")

    a, b = st.columns([1.3, 1])
    with a:
        lab("What survives when we destroy the image")
        st.altair_chart(
            hbars(["MedSigLIP + head", "16×16 blur", "Border only, no lungs",
                   "5 metadata numbers", "Chance"],
                  [SC["full"], SC["lowres"], SC["border"], SC["meta"], SC["chance"]],
                  ["model", "degraded", "degraded", "degraded", "chance"],
                  ["model", "degraded", "chance"], [TEAL, AMBER, "#333c48"],
                  height=240, xtitle="test accuracy", pct=True, xmax=1.08),
            use_container_width=True)
        if BROKEN:
            say(f"A 16×16 thumbnail reaches {SC['lowres']:.1%} — and shows no lung "
                "structure at all. The model is reading the source archive, not "
                "the patient.")
        else:
            say(f"A 16×16 blur manages only {SC['lowres']:.1%}. The signal is in "
                "the anatomy. This result survives the audit.", ok=True)
    with b:
        figure("class_samples.png", "The four classes", cell="Cell 49")

    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        figure("confusion_matrix.png", "Confusion matrix")
        st.write("")
        figure("roc_curve.png", "ROC, one-vs-rest")
    with g2:
        figure("tsne_embeddings.png", "Frozen embeddings, before any training",
               "They already separate — the first hint this task is too easy.")
        st.write("")
        figure("precision_recall_curve.png", "Precision–recall")

    with st.expander("Method"):
        st.markdown(f"""
{M['dataset']}. {M['n_train']:,} training, {M['n_test']:,} held out, {len(C)}
balanced classes. `{M['model']}` frozen; every image embedded once into
{M['embedding_dim']} dimensions and cached; a small Keras head trained on top.

Zero-shot predictions are the argmax of cosine similarity between image and text
embeddings. The shortcut audit trains logistic regression on the same split using
degraded inputs: a 16×16 downsample, the outer border of a 32×32 downsample with
the lung fields masked, and five scalar image statistics.

One dataset, one split, no external validation, no prospective data, no radiologist
adjudication of the labels. Nothing here is evidence of clinical utility.

DeGrave, Janizek & Lee, *AI for radiographic COVID-19 detection selects shortcuts
over signal*, Nature Machine Intelligence, 2021.
""")


# ================================================================== audit
with T[1]:
    lede("Two ways a result like ours can be hollow: the test set overlaps training, "
         "or the classes are separable from something that isn't the disease.")

    lab("1 — Did the test set leak into training?")
    c = st.columns(3)
    stat(c[0], f"{LK['median_nn_sim']:.3f}", "median similarity to nearest "
                                             "training image")
    stat(c[1], f"{LK['frac_above_098']:.1%}", "test images above 0.98",
         RED if LK["frac_above_098"] > 0.02 else TEAL)
    stat(c[2], f"{LK['nn_same_class']:.1%}", "neighbour shares the class")
    st.write("")
    la, lb = st.columns([1.6, 1])
    with la:
        figure("leakage_pairs.png")
    with lb:
        if LK["frac_above_098"] > 0.02:
            say("Near-duplicates cross the split. Part of this accuracy is memory.")
        else:
            say("Clean. The test images are genuinely unseen — which makes the next "
                "test the important one.", ok=True)

    st.markdown("---")
    lab("2 — Do you even need the lungs?")
    ca, cb = st.columns([1, 2.2])
    with ca:
        cls = st.selectbox("Class", C, key="bl_c")
        pool = np.where(y_test == C.index(cls))[0]
        k = st.slider("Image", 0, max(0, len(pool) - 1), 0, key="bl_i")
        i = int(pool[k])
        res = st.select_slider("Resolution", options=[224, 128, 64, 32, 16, 8],
                               value=16, key="bl_r")
        note(f"{res*res:,} pixels, {100*res*res/(224*224):.1f}% of the original")
    with cb:
        im = thumb(i)
        if im is None:
            st.info("No thumbnails in this bundle.")
        else:
            x, y = st.columns(2)
            with x:
                _img(im.resize((420, 420), Image.BILINEAR), width=390)
            with y:
                _img(im.resize((res, res), Image.BILINEAR)
                       .resize((420, 420), Image.NEAREST), width=390)
    if res <= 16 and BROKEN:
        say(f"A classifier on images this size scores {SC['lowres']:.1%}.")
    st.write("")
    figure("shortcut_audit.png")

    st.markdown("---")
    lab("3 — Where does it look?")
    ga, gb = st.columns([1.7, 1])
    with ga:
        figure("gradcam_example.png")
    with gb:
        st.write("")
        note("Grad-CAM through the frozen encoder to the patch tokens. Heat on "
             "markers and borders rather than lung fields is the same finding, "
             "seen a second way. Saliency is suggestive, not proof — which is why "
             "we ran the quantitative test too.")


# ================================================================== zero-shot
with T[2]:
    lede("MedSigLIP has two towers. The workshop pipeline only ever calls the one "
         "that reads images.")
    c = st.columns(3)
    stat(c[0], f"{M['zeroshot_acc']:.1%}", "four sentences, no training", AMBER)
    stat(c[1], f"{M['probe_acc']:.1%}", f"{M['n_train']:,} labels", TEAL)
    stat(c[2], f"{M['probe_acc'] - M['zeroshot_acc']:+.1%}", "what the labels bought")
    st.write("")

    za, zb = st.columns([1, 1])
    with za:
        st.dataframe(pd.DataFrame([{"class": k, "prompt": v}
                                   for k, v in M["prompts"].items()]),
                     hide_index=True, use_container_width=True)
        ps = M.get("prompt_sensitivity") or {}
        if ps:
            st.write("")
            lab("Same weights, same images, different sentence")
            st.altair_chart(hbars(list(ps.keys()), list(ps.values()), height=160,
                                  pct=True, xmax=max(ps.values()) * 1.28),
                            use_container_width=True)
            say(f"Wording alone moves accuracy {max(ps.values()) - min(ps.values()):.1%}.")
    with zb:
        figure("alignment_heatmap.png", "Image classes against text prompts",
               M.get("align_note", ""))


# ================================================================== calibration
with T[3]:
    lede("With no radiologist to overrule it, confidence is the safety mechanism.")
    c = st.columns(4)
    stat(c[0], f"{M['ece']:.3f}", "calibration error",
         TEAL if M["ece"] < 0.05 else AMBER)
    stat(c[1], f"{M['n_overconfident']}/{M['n_wrong']}", "errors made above 90%",
         RED if M["n_overconfident"] else TEAL)
    stat(c[2], f"{M['mean_conf_right']:.3f}", "confidence when right")
    stat(c[3], "—" if M["n_wrong"] == 0 else f"{M['mean_conf_wrong']:.3f}",
         "confidence when wrong", AMBER)
    st.write("")
    figure("calibration.png")
    if M["n_overconfident"]:
        say(f"{M['n_overconfident']} of {M['n_wrong']} mistakes were made above 90% "
            "confidence. A threshold does not catch them.")

    st.markdown("---")
    lab("What a threshold actually buys")
    t = st.slider("Refuse to answer below", 0.25, 1.0, 0.90, 0.01)
    keep = conf >= t
    nk = int(keep.sum())
    s = st.columns(4)
    stat(s[0], f"{keep.mean():.1%}", "answered")
    stat(s[1], f"{(pred[keep] == y_test[keep]).mean():.2%}" if nk else "—",
         "accuracy on those", TEAL)
    stat(s[2], f"{int(((pred != y_test) & keep).sum())}", "wrong, still answered",
         RED if int(((pred != y_test) & keep).sum()) else TEAL)
    stat(s[3], f"{len(y_test) - nk}", "sent to a human", AMBER)
    st.write("")
    rows = []
    for th in np.linspace(0.25, 1.0, 70):
        m = conf >= th
        if m.sum():
            rows.append({"threshold": float(th), "coverage": float(m.mean()),
                         "accuracy on answered": float((pred[m] == y_test[m]).mean())})
    cv = pd.DataFrame(rows).melt("threshold", var_name="series", value_name="value")
    line = alt.Chart(cv).mark_line(strokeWidth=2.4).encode(
        x=alt.X("threshold:Q", axis=alt.Axis(title=None, gridColor="#1a212b",
                                             labelColor=FAINT, domainColor=LINE)),
        y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1.02]),
                axis=alt.Axis(title=None, format="%", gridColor="#1a212b",
                              labelColor=FAINT, domainColor=LINE)),
        color=alt.Color("series:N", scale=alt.Scale(range=[TEAL, AMBER]),
                        legend=alt.Legend(title=None, orient="bottom",
                                          labelColor=DIM)),
        tooltip=["series", alt.Tooltip("threshold:Q", format=".2f"),
                 alt.Tooltip("value:Q", format=".3f")])
    rule = alt.Chart(pd.DataFrame({"t": [t]})).mark_rule(
        color=INK, strokeDash=[4, 4]).encode(x="t:Q")
    st.altair_chart((line + rule).properties(height=290)
                    .configure_view(strokeWidth=0), use_container_width=True)

    st.markdown("---")
    figure("failure_gallery.png", "Every mistake, most confident first", cell="Cell 46")


# ================================================================== try it
with T[4]:
    lede("Four classes, one X-ray. See how you do, then see what the model said.")
    if "g_i" not in st.session_state:
        st.session_state.update(g_i=int(np.random.default_rng(0).integers(len(y_test))),
                                g_you=0, g_mod=0, g_n=0, g_ans=None)
    i = st.session_state.g_i
    gl, gr = st.columns([1, 1.15])

    with gl:
        im = thumb(i, 440)
        if im is not None:
            _img(im, width=420)
        else:
            st.info("No thumbnails in this bundle.")

    with gr:
        if st.session_state.g_ans is None:
            lab("Your call")
            cols = st.columns(2)
            for k, name in enumerate(C):
                if cols[k % 2].button(name, key=f"g{k}", use_container_width=True):
                    st.session_state.g_ans = k
                    st.session_state.g_n += 1
                    st.session_state.g_you += int(k == y_test[i])
                    st.session_state.g_mod += int(pred[i] == y_test[i])
                    st.rerun()
            note(f"Most people find this near-impossible. The model does it at "
                 f"{M['probe_acc']:.0%}.")
        else:
            k, truth, said = st.session_state.g_ans, int(y_test[i]), int(pred[i])
            st.markdown(
                f"<div class='big'>You said <b>{C[k]}</b>, "
                f"<span style='color:{TEAL if k == truth else RED}'>"
                f"{'right' if k == truth else 'wrong'}</span><br>"
                f"Model said <b>{C[said]}</b> at {conf[i]:.0%}, "
                f"<span style='color:{TEAL if said == truth else RED}'>"
                f"{'right' if said == truth else 'wrong'}</span><br>"
                f"It was <b style='color:{TEAL}'>{C[truth]}</b></div>",
                unsafe_allow_html=True)
            st.write("")
            if itp is not None:
                probs, ms = infer(itp, A["X_test"][i])
                note(f"live, quantized TFLite, {ms:.2f} ms")
            else:
                probs = A["y_prob_q"][i]
            st.altair_chart(prob_chart(probs, C, truth, height=155),
                            use_container_width=True)
            if st.button("Next image", type="primary"):
                st.session_state.g_i = int(np.random.default_rng(
                    st.session_state.g_n * 7 + 13).integers(len(y_test)))
                st.session_state.g_ans = None
                st.rerun()

    if st.session_state.g_n:
        st.write("")
        s = st.columns(4)
        stat(s[0], f"{st.session_state.g_you}/{st.session_state.g_n}", "you", AMBER)
        stat(s[1], f"{st.session_state.g_mod}/{st.session_state.g_n}",
             "the model, same images", TEAL)
        stat(s[2], f"{st.session_state.g_you/st.session_state.g_n:.0%}", "your rate")
        stat(s[3], f"{M['probe_acc']:.0%}", "its rate on all 600")


# ================================================================== explore
with T[5]:
    view = st.radio("View", ["Every image", "Embedding map"], horizontal=True,
                    label_visibility="collapsed")
    st.write("")

    if view == "Every image":
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
            labels = [f"#{j} — {C[y_test[j]]} → {C[pred[j]]} ({conf[j]:.0%})"
                      for j in idxs]
            pickd = f3.selectbox(f"Image, {len(idxs)} matching", labels)
            i = int(idxs[labels.index(pickd)])
            st.write("")
            a, b = st.columns([1, 1.35])
            with a:
                im = thumb(i, 380)
                if im is not None:
                    _img(im, width=360)
                note(f"{C[y_test[i]]} → {C[pred[i]]} at {conf[i]:.1%}. "
                     f"Nearest training image {A['nn_sim'][i]:.3f}.")
            with b:
                if itp is not None:
                    probs, ms = infer(itp, A["X_test"][i])
                    lab(f"Trained head, live, {ms:.2f} ms")
                else:
                    probs = A["y_prob_q"][i]
                    lab("Trained head")
                st.altair_chart(prob_chart(probs, C, int(y_test[i]), height=150),
                                use_container_width=True)
                lab("Zero-shot, nothing trained")
                st.altair_chart(prob_chart(A["zs_scores"][i], C, int(y_test[i]),
                                           height=140), use_container_width=True)
            if M.get("confusions"):
                st.markdown("---")
                lab("Which confusions happen")
                st.dataframe(pd.DataFrame(M["confusions"],
                                          columns=["true", "predicted", "count"]),
                             hide_index=True, use_container_width=True)
    else:
        lede("Every held-out image in two dimensions. Crosses are mistakes.")
        e = A["emb2d"]
        d = pd.DataFrame({"x": e[:, 0], "y": e[:, 1],
                          "class": [C[k] for k in y_test],
                          "predicted": [C[k] for k in pred],
                          "confidence": conf, "index": np.arange(len(y_test)),
                          "outcome": np.where(pred == y_test, "correct", "wrong")})
        pts = alt.Chart(d).mark_circle(size=56, opacity=.72).transform_filter(
            alt.datum.outcome == "correct").encode(
            x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
            color=alt.Color("class:N",
                            scale=alt.Scale(range=[TEAL, "#6f8bb5", AMBER, "#a878b8"]),
                            legend=alt.Legend(title=None, orient="bottom",
                                              labelColor=DIM)),
            tooltip=["index", "class", "predicted",
                     alt.Tooltip("confidence:Q", format=".3f")])
        bad = alt.Chart(d).mark_point(shape="cross", size=190, strokeWidth=2.6,
                                      color=RED).transform_filter(
            alt.datum.outcome == "wrong").encode(
            x="x:Q", y="y:Q",
            tooltip=["index", "class", "predicted",
                     alt.Tooltip("confidence:Q", format=".3f")])
        st.altair_chart((pts + bad).properties(height=500)
                        .configure_view(strokeWidth=0), use_container_width=True)


# ================================================================== deployment
with T[6]:
    sz, lat = M["sizes_kb"], M.get("latency_ms", {})
    lede("The encoder stays on a server. Only the head ships to the edge.")
    c = st.columns(4)
    stat(c[0], f"{sz['keras']/1024:.1f} MB", "Keras checkpoint")
    stat(c[1], f"{sz['tflite_int8']:.0f} KB", "quantized", TEAL)
    stat(c[2], f"{sz['keras']/sz['tflite_int8']:.0f}×", "smaller")
    stat(c[3], f"{M['quant_acc'] - M['probe_acc']:+.2%}", "accuracy change",
         TEAL if abs(M["quant_acc"] - M["probe_acc"]) < 0.005 else AMBER)
    st.write("")

    da, db = st.columns([1.2, 1])
    with da:
        st.altair_chart(
            hbars(["Keras float32", "TFLite float32", "TFLite int8"],
                  [sz["keras"], sz["tflite_f32"], sz["tflite_int8"]],
                  ["a", "a", "b"], ["a", "b"], ["#333c48", TEAL],
                  height=160, fmt=",.0f", xtitle="kilobytes"),
            use_container_width=True)
    with db:
        st.dataframe(pd.DataFrame([
            {"artifact": "Keras float32", "KB": f"{sz['keras']:,.1f}",
             "accuracy": f"{M['probe_acc']:.4f}",
             "ms": f"{lat.get('f32_ms', float('nan')):.3f}"},
            {"artifact": "TFLite float32", "KB": f"{sz['tflite_f32']:,.1f}",
             "accuracy": f"{M['probe_acc']:.4f}",
             "ms": f"{lat.get('f32_ms', float('nan')):.3f}"},
            {"artifact": "TFLite int8", "KB": f"{sz['tflite_int8']:,.1f}",
             "accuracy": f"{M['quant_acc']:.4f}",
             "ms": f"{lat.get('int8_ms', float('nan')):.3f}"}]),
            hide_index=True, use_container_width=True)

    say("The embedding still has to come from somewhere. This is an edge classifier, "
        "not an edge pipeline.")

    if itp is not None:
        st.markdown("---")
        lab("Run it on this machine")
        n = st.slider("Images", 50, len(y_test), min(300, len(y_test)), step=50)
        if st.button("Benchmark", type="primary"):
            sub = A["X_test"][:n]
            t0 = time.perf_counter()
            out = np.stack([infer(itp, r)[0] for r in sub])
            dt = (time.perf_counter() - t0) / n * 1000
            r = st.columns(3)
            stat(r[0], f"{dt:.2f} ms", "per image, this CPU", TEAL)
            stat(r[1], f"{1000/dt:,.0f}", "images per second")
            stat(r[2], f"{(out.argmax(1) == y_test[:n]).mean():.1%}",
                 f"accuracy on {n}")