# Kaggle Synthetic Hard Test Set: Results

We tested both trained models on the Kaggle dataset
[pawaritpansing/synthetic-test-set](https://www.kaggle.com/datasets/pawaritpansing/synthetic-test-set).

- **2,556 images**, all **72 classes**
- Every image has one of **20 corruption types**: rotation, shear, erosion/dilation, salt-and-pepper
  and Gaussian noise, blur, faint contrast, scratch lines, perspective tilt, and combinations of these
- The images come in mixed formats (png, jpg, jpeg, bmp, webp)

## Overall accuracy

| Model | Training data | Correct | Accuracy |
|---|---|---|---|
| `model.pt` | real + font images + augmentation | 2425 / 2556 | **94.87%** |
| `model_nofont.pt` | real + augmentation | 2359 / 2556 | 92.29% |

The model trained with font images is **+2.6 points** better. This matches our earlier font test
(96.99% vs 89.81%).

## Accuracy by corruption type (`model.pt`)

| Corruption | Acc | Corruption | Acc |
|---|---|---|---|
| sp_noise | 86.0% | comp_noise_thin | 96.3% |
| comp_scratch_rot | 88.7% | gauss_noise | 96.4% |
| rot_cw | 91.9% | thin_erosion | 96.6% |
| comp_noise_thick | 92.5% | shear_pos | 96.8% |
| comp_persp_noise | 93.5% | comp_lowcon_noise | 97.1% |
| scratch_line | 93.5% | clean | 97.2% |
| comp_shear_blur | 94.7% | thick_dilation | 97.2% |
| rot_ccw | 94.9% | faint_contrast | 97.7% |
| comp_blur_rot | 95.6% | blur_defocus | 97.7% |
| persp_tilt | 95.7% | | |
| shear_neg | 96.0% | | |

- The hardest corruption is **salt-and-pepper noise** (86%), followed by **scratch + rotation** (89%).
- **Clockwise rotation** (91.9%) scores worse than counter-clockwise (94.9%).
- Blur, contrast changes and stroke thickness have little effect on accuracy.

## Weak classes and confusions

**`model.pt`**
- Weakest classes: `ๅ` 21%, `็` 57%, `๘` 81%, `ู` 84%, `ั` 84%, `ข` 86%, `ฃ` 88%
- Top confusions: `ๅ→า` ×39, `็→๘` ×12, `ั→้` ×9, `ว→า` ×5, `ู→้` ×5, `ข→ฃ` ×4
- `ๅ→า` alone accounts for **30% of all errors**. The two glyphs look almost identical in most
  fonts.

**`model_nofont.pt`**
- It fails almost completely on some classes: `ฃ` 0%, `ๅ` 6%, `ฑ` 12%, `๙` 18%
- The font-rendered training data fixes most of these rare glyphs.

## Takeaways

1. `model.pt` is the better model: 94.87% on this hard set.
2. Most of the remaining errors are between look-alike glyphs (`ๅ/า`, `็/๘`, `ั/้`), not caused by the corruptions.
3. Two possible improvements: stronger salt-and-pepper noise augmentation and clockwise rotation
   augmentation.

## How it was run

- Preprocessing is the same as `TestingCNN_csv.py`: RGB, resize to 224×224, ImageNet normalization.
- Predictions are mapped to labels using the `class_to_idx` stored in each checkpoint.
- The run used CPU inference with PyTorch 2.14 / torchvision 0.29.
