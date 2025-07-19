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

# ── 4) Gather all .png recursively (with fallback) ────────────────────────────

all_real = list(REAL_DIR.rglob("*.jpg"))

print(all_real)

if not all_real:

    children = [p for p in REAL_DIR.iterdir() if p.is_dir()]

    if len(children) == 1:

        all_real = list(children[0].rglob("*.jpg"))

if not all_real:

    print(f"No images found under {REAL_DIR}, here’s the tree:")

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
chars = "0123456789abcdefghijklmnopqrstuvwxyz"

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

def add_noise(img, amount=0.02):

    arr = np.array(img)

    noise = np.random.randint(0, 256, arr.shape, dtype='uint8')

    mask = np.random.rand(*arr.shape[:2]) < amount
    arr[mask] = noise[mask]

    return Image.fromarray(arr)

def draw_lines(draw, width, height, n=5):

    for _ in range(n):
        
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        color = tuple(random.randint(0, 150) for i in range(3))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=random.randint(1, 3))

def draw_dots(draw, width, height, n=30):

    for _ in range(n):

        x, y = random.randint(0, width), random.randint(0, height)
        r = random.randint(1, 3)
        color = tuple(random.randint(0, 150) for i in range(3))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

def random_warp(img):

    width, height = img.size

    dx = width * 0.1 * (np.random.rand(2) - 0.5)
    dy = height * 0.1 * (np.random.rand(2) - 0.5)

    x1 = int(dx[0])
    y1 = int(dy[0])
    x2 = width + int(dx[1])
    y2 = height + int(dy[1])

    # Use Image.Transform.QUAD and Image.Resampling.BICUBIC for compatibility with modern Pillow

    return img.transform(

        img.size,

        Image.Transform.QUAD,

        (0, 0, width, 0, x2, y2, 0, height),

        resample=Image.Resampling.BICUBIC

    )

for i in range(n_syn):

    length = random.choice([4, 6, 7, 8])
    text   = "".join(random.choices(chars, k=length))

    # Generate captcha(we gotta put it through distortions because the captcha dataset is too simple)

    gen = ImageCaptcha(width=200, height=80)
    base_img_path = SYN_DIR / f"{text}_{i:05d}.jpg"
    gen.write(text, str(base_img_path))

    # Open with PIL

    img = Image.open(base_img_path).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)
    
    # Augmentations/Distortions(see above)
    
    draw_lines(draw, width, height, n=random.randint(3, 7))
    draw_dots(draw, width, height, n=random.randint(20, 50))

    img = add_noise(img, amount=0.03)

    if random.random() < 0.7:

        if random.random() < 0.7:

            img = random_warp(img)

        if random.random() < 0.5:

            img = img.filter(ImageFilter.GaussianBlur(radius=int(random.uniform(0, 1.2))))

        img.save(base_img_path)

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

print(f"Found real .jpg:       {n_real}")
print(f"Subsampled real:        {n_sub}")
print(f"Generated synthetic:    {n_syn}")
print(f"Total images outputted: {len(imgs)} (80/20 split into train/test)")
print("Done bro")
