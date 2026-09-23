This Used Rasnet18 defualt weight

resolution 244,244

data augmentation
- rotate +-10 
- img shift
- size/scale
- light change(little)
- normalize to standard of rasnet input
- erotion/dialation จำลองตัวบาง/หนา
- randomErasing สุ่มลบส่วนภาพ

hightlight i think is
1. class balance by add weight each class
2. doup_out 30%
3. learning rate with decay
4. early stop
5. save data in train loop using torch so it doing on GPU instead of CPU(lower time lag at communicate) EX.running_loss ,correct
6. calc. val_loss in train loop to see overfit and early stop and save best model
7. have checkpoint to save both model and index(in case test data sort differently)


#### มีการอธิบายชุดข้อมูล อาทิเช่น จำนวนคลาส จำนวนข้อมูลแต่ละคลาส ตัวอย่างข้อมูลในแต่ละคลาส
- 72 classes
- imange in class have size around 10*10 to 35*35
- .jpg
- have over 62,711 picture
#### มีการวิเคราะห์ความท้าทายของชุดข้อมูลนี้
- serve unbalance classes
- low resolution
- simirality in some class
- distort image/character
#### มีการอธิบายถึงโครงสร้าง CNN ที่ใช้งานในภาพรวม
- rasnet18
-input image to 64 7x7 kernel filter stride 2
- pooling 2 stride 2
- use 3x3 conv to increase channel to finetune detail(64>128>256>512)
  - each layer conv 3 timed
  - after 64 layer(start 128) first conv stride 2
  - rasnet connectivity
- 7x7 pooling stride 7
- FC 1000(1000x1x1)
- softmax
##### add for this CNN
- drop out 30%
- linear output to 72 classes instead of 1000

#### มีการอธิบายถึงเทคนิคการถ่ายโอนความรู้ (Transfer Learning)  
- use rasnet18 model and it's defualt weight 
#### มีการใอธิบายถึงเทคนิคการสังเคราะห์ข้อมูล (Data Augmentation) 
- rotate +-10 
- img shift
- size/scale
- light change(little)
- normalize to standard of rasnet input
- erotion/dialation จำลองตัวบาง/หนา
- randomErasing สุ่มลบส่วนภาพ
#### มีการอธิบายถึงเทคนิคหรือแนวคิดที่น่าสนใจ
1. class balance by add weight each class
2. doup_out 30%
3. learning rate with decay
4. early stop
5. save data in train loop using torch so it doing on GPU instead of CPU(lower time lag at communicate) EX.running_loss ,correct
6. calc. val_loss in train loop to see overfit and early stop and save best model
7. have checkpoint to save both model and index(in case test data sort differently)
8. erotion/dialation จำลองตัวบาง/หนา
#### มีการอธิบายถึงการทำงานของ CNN โดยเน้นที่จุดเด่นของสถาปัตยกรรม 
- use 3x3 conv to increase channel to finetune detail(64>128>256>512)
  - each layer conv 3 timed
  - after 64 layer(start 128) first conv stride 2
  - rasnet connectivity(Residual Blocks)
- 7x7 pooling stride 7 (GAP)
- FC 1000(1000x1x1)(fully connected)
#### มีการอธิบายถึงเทคนิคที่ใช้งาน 
- doup_out 30%
- early stop
- learning rate with decay
- save data in train loop using torch so it doing on GPU instead of CPU(lower time lag at communicate) EX.running_loss ,correct
- calc. val_loss in train loop to see overfit and early stop and save best model
-  class balance by add weight each class
#### มีการอธิบายถึงขั้นตอนการฝึกสอนแบบจำลอง
- yea bro
#### มีการแสดงประสิทธิภาพของชุดฝึกสอนด้วยค่าความถูกต้อง (Accuracy Rate)

#### มีการแบ่งชุดข้อมูลออกเป็น Train และ Validation ในสัดส่วน 80% ต่อ 20% 

มีการแนะนำสมาชิกในกลุ่ม


---

# Updated version: augmentation + font data

Classifies an image of a single Thai character into one of **72 classes** (consonants, vowels,
tone marks, digits) by fine-tuning an ImageNet-pretrained ResNet-18.

## Results

| Model | Training data | Real 20% test (12,545 imgs) | Font test (432 imgs) |
|---|---|---|---|
| Original baseline | real only | — | 78.47% |
| `model_nofont.pt` | real (80%) + augmentation | **97.56%** | 89.81% |
| `model.pt` | real (80%) + font images + augmentation | 97.35% | **96.99%** |

## Data

```
dataset/round2/<class>/          real images, 72 class folders (62,707 images)
dataset/synth_fonts/             font-rendered images (generated, not in git: run render_fonts.py)
archive/synthetic_test_set/      font test set: 72 classes x 6 fonts = 432 images
archive2/test/                   same test images + test.csv (for TestingCNN_csv.py)
```

Folder names are TIS-620 codes: `161` = ก, `162` = ข, ... (`bytes([int(name)]).decode('cp874')`).
The real data is split **80% train / 20% test**, stratified per class.

## Files

| File | Purpose |
|---|---|
| `TrainingCNN.py` | Training (AdamW + OneCycle, label smoothing, class-balanced sampling, bf16) |
| `augment.py` | Augmentation: rotation, shear, perspective, elastic warp, blur, noise, cutout, stroke thickness, low-res, JPEG |
| `render_fonts.py` | Generates extra training images from Windows Thai fonts (test fonts excluded) |
| `preview_augment.py` | Saves `augment_preview.png` to inspect augmentations |
| `test.py`, `TestingCNN_csv.py` | Run a trained model on test images |
| `train_log*.csv` | Per-epoch training logs |

## Usage

```bash
pip install torch torchvision opencv-python pillow numpy

# real data only (20% real split = pure test set)
python TrainingCNN.py --no-synth --out model_nofont.pt --log train_log_nofont.csv

# with font images (render them first)
python render_fonts.py
python TrainingCNN.py
```

`model.pt` / `model_nofont.pt` store `{'model_state', 'class_to_idx'}`.
Input: RGB, resized to 224x224, ImageNet normalization.
