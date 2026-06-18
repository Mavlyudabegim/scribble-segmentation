Scribble-Supervised Image Segmentation
Binary image segmentation from sparse scribble annotations, using superpixel-based label propagation and a deep UNet model.

Overview
Standard segmentation models require dense pixel-level annotations, which are expensive to collect. This project uses scribble supervision — a few rough strokes drawn by a user — as the only labeling input. Sparse scribble labels are propagated across the image using graph-based label spreading, then used to train a UNet model with a custom hybrid loss.

Method

1. Preprocessing (preprocess.py)
   SLIC superpixels partition each image into ~2500 compact, colour-coherent regions.
   LabelSpreading (RBF kernel) propagates scribble labels {0=bg, 1=fg} from annotated superpixels to all unlabeled ones, producing a soft foreground probability map.
   Distance-based scribble channels encode proximity to fg/bg strokes as two additional input channels (exponential decay from each scribble).
   A per-pixel weight map is built from label confidence and proximity to scribble strokes — used during training to emphasize reliable regions.
   Results cached to .pt files so preprocessing runs only once.
2. Model (src/model.py)
   UNet with a pretrained ResNet-34 encoder (ImageNet weights).
   5-channel input: RGB + fg distance channel + bg distance channel.
   Single output channel — raw logit map for binary segmentation.
3. Loss (src/loss.py)
   Hybrid loss combining two supervision signals:

loss = λ · supervised(pred, scribble_labels, weight_map) + (1-λ) · supervised(pred, ground_truth, ones)
where supervised = Focal loss + Soft Dice loss and λ = 0.3.

Focal loss down-weights easy background pixels and focuses learning on object boundaries.
Soft Dice loss directly optimises the overlap metric, robust to class imbalance. 4. Training (challenge.py)
80/20 train/validation split (stratified, fixed seed).
Adam optimiser, lr=1e-4, weight_decay=1e-4.
Early stopping with patience 5 on validation mIoU.
Best checkpoint saved automatically. 5. Post-processing (src/inference.py)
Bilateral filtering on the raw probability map — smooths predictions while preserving colour edges.
Morphological closing fills small holes inside predicted foreground regions.
Morphological opening removes isolated noise blobs.
Results
Evaluated on a held-out 20% validation split (unseen during training):

Metric Score
Foreground IoU 0.816
Background IoU 0.956
Mean IoU 0.886
Training converged in 20 epochs with early stopping.

Project Structure
.
├── preprocess.py # Run once — builds tensor caches from raw data
├── challenge.py # Training, evaluation, and prediction
│
└── src/
├── data.py # make_loader(), save_masks()
├── model.py # SegmentationNet (UNet + ResNet-34)
├── loss.py # soft_dice, focal, total_loss
├── metrics.py # batch_iou (torch) · dataset_iou (numpy)
├── inference.py # run_inference() · refine_mask()
└── visualise.py # show_result()
Data Layout
dataset/
train/
images/ _.jpg
scribbles/ _.png (0=bg · 1=fg · 255=unlabeled)
ground_truth/ _.png
predictions/ ← written by challenge.py
test/
images/ _.jpg
scribbles/ \*.png
predictions/ ← written by challenge.py
preprocessed/
train.pt ← written by preprocess.py
test.pt ← written by preprocess.py
Setup
pip install torch torchvision segmentation-models-pytorch \
 scikit-learn scikit-image scipy tqdm \
 pillow matplotlib opencv-python
Usage

# Step 1 — preprocess (slow, runs once, results cached)

python preprocess.py

# Step 2 — train, evaluate, predict

python challenge.py
challenge.py will:

Train on 80% of the labeled data with early stopping.
Print FG / BG / Mean IoU on the held-out 20% validation split.
Display a random validation example (image + scribbles | ground truth | prediction).
Save predictions for both training and test sets.
