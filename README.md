# Image Segmentation Challenge

## Structure

```
.
├── preprocess.py        # Step 1 — run once to build the tensor cache
├── challenge.py         # Step 2 — training, evaluation, prediction
│
└── src/
    ├── data.py          # make_loader(), save_masks()
    ├── model.py         # SegmentationNet  (UNet + ResNet-34, 5-channel input)
    ├── loss.py          # soft_dice, focal, total_loss (hybrid scribble + GT)
    ├── metrics.py       # batch_iou (torch) · dataset_iou (numpy)
    ├── inference.py     # run_inference() · refine_mask() (bilateral + morphology)
    └── visualise.py     # show_result()
```

## Data layout expected

```
data/
  images/           *.jpg
  scribbles/        *.png   — 0 = background · 1 = foreground · 255 = unlabeled
  ground_truths/    *.png
```

## Install

```bash
pip install torch torchvision segmentation-models-pytorch \
            scikit-learn scikit-image scipy tqdm \
            pillow matplotlib opencv-python
```

## Run

```bash
# Once — slow (SLIC + label propagation per image), result is cached
python preprocess.py

# Every training run — fast after the first preprocess
python challenge.py
```

## What each step does

**preprocess.py**

- Runs SLIC superpixels on each image (~2500 segments).
- Uses LabelSpreading (RBF kernel) to propagate scribble labels to all superpixels.
- Builds a 5-channel input tensor: RGB + distance-to-fg-scribble + distance-to-bg-scribble.
- Builds a soft label map (fg probability) and a per-pixel weight map.
- Saves everything to `data/preprocessed.pt`.

**challenge.py**

- Loads the `.pt` cache and splits 80 / 20 into train and validation sets.
- Trains SegmentationNet with a hybrid loss (scribble branch + GT branch).
- Applies early stopping when validation mIoU stops improving.
- Runs inference on the validation split and refines masks with bilateral filtering.
- Prints FG / BG / Mean IoU, shows a random example, saves all predictions.

<img width="1280" height="391" alt="telegram-cloud-photo-size-2-5296753587729932987-y" src="https://github.com/user-attachments/assets/0247e277-a060-4c3b-9cdb-b8c7478ce5df" />
