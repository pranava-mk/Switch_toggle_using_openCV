# 👋 Switch Toggle via Human Detection (OpenCV)

> Detect a person in the webcam frame and flip a virtual (or real) switch —
> built for Innovation Project League.

Uses a pretrained **MobileNet-SSD** Caffe model to detect people in a live
webcam feed and toggle a visual indicator (and, in concept, a real switch —
e.g. a smartphone flashlight) based on whether someone's present.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-41CD52)

---

## ✨ What it does

Three variants, same core detection loop:

| Script | What it adds |
|---|---|
| `human detection.py` | Base case — draws a bounding box + confidence % around detected people |
| `human_detection_switch_toggle.py` | Adds a red/green circle overlay that flips based on detection — the actual "switch toggle" visual |
| `Simulation.py` | Same detection, plus a separate PyQt5 window whose background color flips red/green |

## 🚀 Getting Started

One command — nothing to pre-clone, it clones itself:

```bash
curl -fsSL https://raw.githubusercontent.com/pranava-mk/Switch_toggle_using_openCV/main/install.sh | bash
```

This checks for `git`/`python3`/`pip` (offering an OS-appropriate install via
apt/pacman/brew if missing, always asking before touching the system),
clones the repo if you're not already in it, creates a `.venv`, installs the
pinned `requirements.txt`, then prints the exact commands to run. It will
**not** auto-launch any detection script — see "Usage" below; you need a
real webcam attached.

### Requirements

- Python 3.11+
- A webcam

### ⚠️ OpenCV version is pinned on purpose

`requirements.txt` pins `opencv-python==4.10.0.84`. **OpenCV 5.x removed
Caffe model support from `cv2.dnn`** (`readNetFromCaffe` no longer exists) —
confirmed by actually hitting `AttributeError` against the latest release
during testing. This project needs a 4.x `opencv-python`; don't bump it
without checking Caffe support still exists in whatever version you move to.

## 🖥️ Usage

```bash
.venv/bin/python "human detection.py"              # bounding boxes only
.venv/bin/python human_detection_switch_toggle.py  # + red/green toggle circle
.venv/bin/python Simulation.py                     # + separate PyQt5 status window
```

Press **`q`** to quit any of them.

## 🗂️ Files

```
Switch_toggle_using_openCV/
├── MobileNetSSD_deploy.caffemodel      # pretrained weights (~23MB)
├── MobileNetSSD_deploy.prototxt.txt    # model architecture definition
├── human detection.py
├── human_detection_switch_toggle.py
├── Simulation.py
├── requirements.txt
└── setup.sh
```

## ⚠️ Notes

- The `CLASSES` list in each script is a leftover/decorative list of
  MobileNet-SSD's object categories — it's never actually indexed in code
  (detection only checks `class_id == PERSON_CLASS_IDX`), so don't read it
  as the real class mapping if you're extending this to detect other
  objects; check the model's actual label order instead.
- `human_detection_switch_toggle.py`'s toggle is purely visual (an on-screen
  circle). Actually driving a real switch/flashlight is sketched as a
  comment in `human detection.py` but not implemented.
