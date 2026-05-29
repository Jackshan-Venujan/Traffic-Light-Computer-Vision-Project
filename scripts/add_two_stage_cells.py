"""
One-off builder: appends Step 13 (two-stage pipeline) cells to
Traffic_Light_Complete_Workflow.ipynb. Safe to re-run — it removes any
previously-appended Step 13 cells before adding the new ones.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "Traffic_Light_Complete_Workflow.ipynb"
MARKER = "STEP-13-TWO-STAGE-PIPELINE"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


MD_HEADER = """## Step 13: Two-Stage Pipeline — housing detector + bulb-state classifier
<!-- __MARKER__ -->

The earlier filter (Step 7b) threw away the 7 bulb classes because they are
~3 px wide at training resolution — below YOLOv8's detection floor. The
remaining 7 housing classes also encode the lit state in their NAMES
(`go traffic light`, `stop traffic light`, …), which means the current
detector is being asked to do TWO jobs at once: localize the housing AND
read the lit bulb.

This step splits those two jobs:

- **Stage 1** — a single-class YOLO that just finds the housing box.
  Easier task → higher recall.
- **Stage 2** — a tiny MobileNetV3-Small that takes the cropped housing
  and classifies it into one of the 7 states. With a normalized 96×96
  crop the bulb is suddenly ~20 px wide — well within reach for a CNN.

End-to-end inference: detect housings with Stage 1 → crop each → classify
with Stage 2 → emit `(box, state)` pairs.
"""


CODE_13A = """# 13a. BUILD STAGE-1 DATASET (single-class 'traffic_light')
# __MARKER__
# Reuses the same images and the same boxes as data/processed/, but rewrites
# every class ID to 0. The result is YOLO-ready labels for a 1-class detector.
import shutil

STAGE1_DIR = PROJECT_ROOT / "data" / "stage1"
STAGE1_LBL = STAGE1_DIR / "labels"
SRC_LBL = DATA_PROCESSED_DIR / "labels"
SRC_IMG = DATA_PROCESSED_DIR / "images"

if STAGE1_LBL.exists():
    shutil.rmtree(STAGE1_LBL)

total_files = total_boxes = 0
for split in ("train", "val", "test"):
    src = SRC_LBL / split
    dst = STAGE1_LBL / split
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        continue
    for f in src.glob("*.txt"):
        new_lines = []
        for line in f.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            # Rewrite the class id (first token) to 0; keep bbox unchanged.
            new_lines.append("0 " + " ".join(parts[1:]))
            total_boxes += 1
        (dst / f.name).write_text(("\\n".join(new_lines) + "\\n") if new_lines else "")
        total_files += 1

# YOLO resolves labels by swapping /images/ -> /labels/ in the image path. So
# we mirror the images under data/stage1/images/ via a symlink, then point the
# yaml's image roots at that local tree. This way labels land in our new dir.
STAGE1_IMG = STAGE1_DIR / "images"
if STAGE1_IMG.is_symlink() or STAGE1_IMG.exists():
    if STAGE1_IMG.is_symlink():
        STAGE1_IMG.unlink()
    else:
        shutil.rmtree(STAGE1_IMG)
STAGE1_IMG.symlink_to(SRC_IMG, target_is_directory=True)

stage1_yaml = STAGE1_DIR / "data.yaml"
stage1_yaml.write_text(
    "path: " + str(STAGE1_DIR) + "\\n"
    "train: images/train\\n"
    "val:   images/val\\n"
    "test:  images/test\\n\\n"
    "nc: 1\\n"
    "names:\\n"
    "  0: traffic_light\\n"
)

print(f"Rewrote {total_files} label files, {total_boxes} boxes -> class 0")
print(f"Stage-1 yaml: {stage1_yaml}")
print(stage1_yaml.read_text())
"""


CODE_13B = """# 13b. TRAIN STAGE-1 (single-class housing detector)
# __MARKER__
from ultralytics import YOLO

# Clear stale YOLO label caches so labels are re-parsed fresh.
for split in ("train", "val"):
    p = STAGE1_LBL / f"{split}.cache"
    if p.exists():
        p.unlink()

stage1_model = YOLO('yolov8n.pt')
stage1_results = stage1_model.train(
    data=str(stage1_yaml),
    epochs=30,                  # 1-class converges faster than 7-class
    imgsz=1280,
    batch=8,
    patience=8,
    device=0,                   # 'cpu' if no GPU
    amp=False,                  # same AMP gotcha as Step 10 — leave OFF on RTX 5090
    project=str(RUNS_DIR),
    name='stage1_localizer',
)
print("\\nStage 1 done. best.pt at:",
      RUNS_DIR / 'stage1_localizer' / 'weights' / 'best.pt')
"""


CODE_13C = """# 13c. BUILD STAGE-2 CROP DATASET (housing crops -> bulb-state label)
# __MARKER__
# For every box in data/processed/labels/<split>, crop the corresponding region
# from the image with a small padding, and save it under
# data/stage2/<split>/<class_name>/crop_NNN.jpg.  The class name comes from
# the 7-class label list in data/processed/data.yaml.
import yaml
from PIL import Image

STAGE2_DIR = PROJECT_ROOT / "data" / "stage2"
if STAGE2_DIR.exists():
    shutil.rmtree(STAGE2_DIR)

with open(DATA_PROCESSED_DIR / "data.yaml") as f:
    proc_yaml = yaml.safe_load(f)
STAGE2_CLASSES = [proc_yaml["names"][i] for i in range(proc_yaml["nc"])]
print("Stage-2 classes:", STAGE2_CLASSES)

PAD_RATIO = 0.15      # 15% padding around the housing
MIN_CROP_PX = 16      # skip housings whose smaller side is < this many px

counts = {c: 0 for c in STAGE2_CLASSES}
skipped = 0
for split in ("train", "val", "test"):
    img_dir = SRC_IMG / split
    lbl_dir = SRC_LBL / split
    if not img_dir.exists() or not lbl_dir.exists():
        continue
    for lbl_file in lbl_dir.glob("*.txt"):
        text = lbl_file.read_text().strip()
        if not text:
            continue
        stem = lbl_file.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = img_dir / (stem + ext)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        W, H = img.size
        for line in text.splitlines():
            cls_id, cx, cy, bw, bh = line.split()
            cls_id = int(cls_id)
            cx, cy, bw, bh = float(cx), float(cy), float(bw), float(bh)
            cls_name = STAGE2_CLASSES[cls_id]
            # YOLO -> pixel xyxy
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            pw, ph = (x2 - x1), (y2 - y1)
            if min(pw, ph) < MIN_CROP_PX:
                skipped += 1
                continue
            px = pw * PAD_RATIO
            py = ph * PAD_RATIO
            cx1 = max(0, int(x1 - px))
            cy1 = max(0, int(y1 - py))
            cx2 = min(W, int(x2 + px))
            cy2 = min(H, int(y2 + py))
            crop = img.crop((cx1, cy1, cx2, cy2))
            out_dir = STAGE2_DIR / split / cls_name.replace(" ", "_")
            out_dir.mkdir(parents=True, exist_ok=True)
            n = counts[cls_name]
            crop.save(out_dir / f"crop_{n:06d}.jpg", quality=92)
            counts[cls_name] += 1

print("\\nCrops written per class (across all splits):")
for c, n in counts.items():
    print(f"  {c:30s}  {n}")
print(f"\\nSkipped {skipped} boxes below {MIN_CROP_PX}px (too small even for stage 2).")
"""


CODE_13D = """# 13d. TRAIN STAGE-2 — MobileNetV3-Small, ImageNet pretrained, 7-way head
# __MARKER__
import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

IMG_SIZE = 96
BATCH = 128
EPOCHS = 12

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(STAGE2_DIR / "train", transform=train_tf)
val_ds   = datasets.ImageFolder(STAGE2_DIR / "val",   transform=eval_tf)
test_ds  = datasets.ImageFolder(STAGE2_DIR / "test",  transform=eval_tf)
print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
print("Class -> index mapping:", train_ds.class_to_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                          num_workers=4, pin_memory=True)

# Class-balanced loss to handle imbalance (warning/go_left are rarer).
label_counts = collections.Counter(train_ds.targets)
weights = torch.tensor(
    [len(train_ds) / (len(label_counts) * label_counts[i]) for i in range(len(label_counts))],
    dtype=torch.float, device=DEVICE,
)
print("Class weights:", weights.cpu().numpy().round(2))

NUM_CLASSES = len(train_ds.classes)
net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
# Replace the final Linear(1024 -> 1000) with a 7-way head.
in_feats = net.classifier[3].in_features
net.classifier[3] = nn.Linear(in_feats, NUM_CLASSES)
net = net.to(DEVICE)

opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
loss_fn = nn.CrossEntropyLoss(weight=weights)


@torch.no_grad()
def evaluate(loader):
    net.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = net(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


best_val = 0.0
STAGE2_OUT = MODELS_DIR / "stage2_mobilenet.pt"
STAGE2_OUT.parent.mkdir(parents=True, exist_ok=True)

for epoch in range(1, EPOCHS + 1):
    net.train()
    running = 0.0
    seen = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = net(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        running += loss.item() * y.size(0)
        seen += y.size(0)
    sched.step()
    val_acc = evaluate(val_loader)
    print(f"epoch {epoch:02d}  train_loss={running/seen:.4f}  val_acc={val_acc*100:.2f}%")
    if val_acc > best_val:
        best_val = val_acc
        torch.save({
            "state_dict": net.state_dict(),
            "classes": train_ds.classes,
            "img_size": IMG_SIZE,
        }, STAGE2_OUT)

print(f"\\nBest val acc: {best_val*100:.2f}%")
test_acc = evaluate(test_loader)
print(f"Test acc: {test_acc*100:.2f}%")
print(f"Saved stage-2 weights to: {STAGE2_OUT}")
"""


CODE_13E = """# 13e. TWO-STAGE INFERENCE — wire stage 1 (YOLO) + stage 2 (MobileNetV3) together
# __MARKER__
import random
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO
from matplotlib.patches import Rectangle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

stage1_best = sorted(
    (RUNS_DIR / 'stage1_localizer').glob("**/weights/best.pt"),
    key=lambda p: p.stat().st_mtime,
)[-1]
print("Stage 1 model:", stage1_best)
stage1 = YOLO(str(stage1_best))

ckpt = torch.load(MODELS_DIR / "stage2_mobilenet.pt", map_location=DEVICE)
S2_CLASSES = ckpt["classes"]
S2_IMG = ckpt["img_size"]
stage2 = models.mobilenet_v3_small(weights=None)
stage2.classifier[3] = nn.Linear(stage2.classifier[3].in_features, len(S2_CLASSES))
stage2.load_state_dict(ckpt["state_dict"])
stage2 = stage2.to(DEVICE).eval()
print("Stage 2 classes:", S2_CLASSES)

s2_tf = transforms.Compose([
    transforms.Resize((S2_IMG, S2_IMG)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def two_stage_predict(image_path, conf=0.25, pad_ratio=0.15):
    \"\"\"Detect housings with stage 1, classify each crop with stage 2.

    Returns a list of dicts:
        [{box: (x1,y1,x2,y2), state: str, det_conf: float, cls_conf: float}]
    \"\"\"
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    det = stage1.predict(source=str(image_path), conf=conf, verbose=False)[0]
    out = []
    if det.boxes is None or len(det.boxes) == 0:
        return out
    boxes = det.boxes.xyxy.cpu().numpy()
    confs = det.boxes.conf.cpu().numpy()
    crops, geoms = [], []
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * pad_ratio, bh * pad_ratio
        cx1 = max(0, int(x1 - px))
        cy1 = max(0, int(y1 - py))
        cx2 = min(W, int(x2 + px))
        cy2 = min(H, int(y2 + py))
        crops.append(s2_tf(img.crop((cx1, cy1, cx2, cy2))))
        geoms.append(((int(x1), int(y1), int(x2), int(y2)), float(c)))
    batch = torch.stack(crops).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(stage2(batch), dim=1)
        cls_conf, cls_idx = probs.max(1)
    for (box, dconf), cidx, ccf in zip(geoms, cls_idx.cpu().numpy(), cls_conf.cpu().numpy()):
        out.append({
            "box": box,
            "state": S2_CLASSES[cidx],
            "det_conf": dconf,
            "cls_conf": float(ccf),
        })
    return out


# ---- Demo on 4 random test images ----
test_imgs = sorted((SRC_IMG / "test").glob("*.jpg"))
random.seed(0)
sample = random.sample(test_imgs, 4)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, ipath in zip(axes.flat, sample):
    img = Image.open(ipath).convert("RGB")
    preds = two_stage_predict(ipath)
    ax.imshow(img)
    for p in preds:
        x1, y1, x2, y2 = p["box"]
        state = p["state"].lower()
        color = "lime" if "go" in state else ("red" if "stop" in state else "gold")
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                               linewidth=2, edgecolor=color, facecolor='none'))
        label = p["state"].replace("_traffic_light", "").replace("_", " ")
        label += f"  d={p['det_conf']:.2f} c={p['cls_conf']:.2f}"
        ax.text(x1, max(0, y1 - 4), label, color='black', fontsize=8,
                bbox=dict(facecolor=color, alpha=0.75, pad=1))
    ax.set_title(f"{ipath.name}  ({len(preds)} detections)", fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "two_stage_demo.png", dpi=120, bbox_inches='tight')
plt.show()
print("\\nSaved demo: two_stage_demo.png")
"""


CELLS = [
    md(MD_HEADER.replace("__MARKER__", MARKER)),
    code(CODE_13A.replace("__MARKER__", MARKER)),
    code(CODE_13B.replace("__MARKER__", MARKER)),
    code(CODE_13C.replace("__MARKER__", MARKER)),
    code(CODE_13D.replace("__MARKER__", MARKER)),
    code(CODE_13E.replace("__MARKER__", MARKER)),
]


def main():
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]
    before = len(cells)
    # Drop any previously-appended Step 13 cells (so this script is idempotent).
    cells = [c for c in cells if MARKER not in "".join(c.get("source", []))]
    cells.extend(CELLS)
    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, indent=1))
    print(f"OK — appended {len(CELLS)} cells. Notebook: {before} -> {len(cells)} cells.")


if __name__ == "__main__":
    main()
