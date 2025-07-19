import os
import random
import shutil
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from captcha.image import ImageCaptcha
from sklearn.model_selection import train_test_split

# ── 1) Paths relative to this script ──────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "captcha_data"
REAL_DIR = DATA_DIR / "real"
SUB_DIR  = DATA_DIR / "real_sub"
SYN_DIR  = DATA_DIR / "synth"
OUT_DIR  = DATA_DIR / "output"

# ── 2) Ensure directories exist ───────────────────────────────────────────────
for d in (REAL_DIR, SUB_DIR, SYN_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── 3) Download & unzip via Kaggle API ────────────────────────────────────────
api = KaggleApi()
api.authenticate()
api.dataset_download_files(
    "parsasam/captcha-dataset",
    path=str(REAL_DIR),
    unzip=True
)

# ── 4) Gather all .jpg recursively (with fallback) ────────────────────────────
all_real = list(REAL_DIR.rglob("*.jpg"))
if not all_real:
    # maybe everything landed one folder deeper?
    children = [p for p in REAL_DIR.iterdir() if p.is_dir()]
    if len(children) == 1:
        all_real = list(children[0].rglob("*.jpg"))

if not all_real:
    # debug print
    print(f"No jpgs found under {REAL_DIR}, here’s the tree:")
    for root, dirs, files in os.walk(REAL_DIR):
        print(f"  {root!s}: {len(files)} files, subdirs={dirs}")
    raise RuntimeError(f"No real images found – check that `{REAL_DIR}` contains .jpg files")

# ── 5) Subsample 25% of the real captchas ────────────────────────────────────
n_real = len(all_real)
n_sub  = max(1, int(n_real * 0.25))
sub    = random.sample(all_real, n_sub)
for src in sub:
    shutil.copy(src, SUB_DIR)

# ── 6) Synthesize 3× as many new captchas ────────────────────────────────────
n_syn = 3 * n_sub
gen   = ImageCaptcha(width=200, height=80)
chars = "0123456789abcdefghijklmnopqrstuvwxyz"

for i in range(n_syn):
    length = random.choice([4, 6, 7, 8])
    text   = "".join(random.choices(chars, k=length))
    gen.write(text, str(SYN_DIR / f"{text}_{i:05d}.jpg"))

# ── 7) Combine & shuffle ─────────────────────────────────────────────────────
imgs = []
for label, folder in [("real", SUB_DIR), ("syn", SYN_DIR)]:
    for path in folder.glob("*.jpg"):
        imgs.append((label, str(path)))
random.shuffle(imgs)

# ── 8) 80/20 train-test split ─────────────────────────────────────────────────
train, test = train_test_split(imgs, train_size=0.8, random_state=42)

for split_name, subset in [("train", train), ("test", test)]:
    dst = OUT_DIR / split_name
    dst.mkdir(exist_ok=True)
    for label, path in subset:
        dst_path = dst / f"{label}_{Path(path).name}"
        shutil.copy(path, dst_path)

# ── 9) Summary ───────────────────────────────────────────────────────────────
print(f"Found real .jpgs:       {n_real}")
print(f"Subsampled real:        {n_sub}")
print(f"Generated synthetic:    {n_syn}")
print(f"Total images outputted: {len(imgs)} (80/20 split into train/test)")
print("Done bro")
