# Few-shot hyperspectral image classification

This repository contains a paired pilot experiment for spatially weighted episodic
training in same-scene hyperspectral pixel classification. The hypothesis is that
upweighting labeled queries farther from same-class supports encourages features
that generalize beyond immediately neighboring patches.

## Comparison

| Method | Training objective | Prediction |
| --- | --- | --- |
| `ce` | Cross-entropy over the five labeled centers per class | Cosine linear classifier |
| `uniform` | 2-support, 3-query prototype episodes with uniform query weights | Squared Euclidean distance to the five-support class mean |
| `shuffled` | Same episodes and weight values as `spatial`, permuted among queries within each class | Same prototype classifier |
| `spatial` | Query weights increase with the distance to the nearest same-class support | Same prototype classifier |

The encoder, initialization seed, and label budget are shared across methods.
Standardization uses only the labeled training centers. Five additional centers per
class are reserved in order to match the existing label split design. The pilot
uses fixed training steps, so reserved labels play no role in model selection.

Three evaluation cohorts reuse the same models and predictions. `random_test`
contains all remaining labeled centers, `no_overlap` requires a Chebyshev
distance of at least 5 pixels from training and reserved centers, and `guard_10`
requires at least 10 pixels. At patch size 5, the first guard prevents shared input
pixels between a test patch and a training patch. Distance filtering is a
within-scene evaluation. A spatially separated or independently acquired scene
is needed to estimate cross-region transfer.

## Reproduce

Install the dependencies with `python -m pip install -r requirements.txt`.
For CUDA, use a PyTorch build that supports the installed GPU. Use
`--device cpu` for a CPU run.

Supply the benchmark files in the following layout.

| Dataset folder | Image | Labels |
| --- | --- | --- |
| `ip` | `Indian_pines_corrected.mat` | `Indian_pines_gt.mat` |
| `pu` | `PaviaU.mat` | `PaviaU_gt.mat` |
| `houston13` | `GRSS2013.mat` | `GRSS2013_gt.mat` |
| `whuhh` | `WHU_Hi_HongHu.mat` | `WHU_Hi_HongHu_gt.mat` |

Run the five-seed pilot and aggregate it:

```sh
python experiment.py --data-root /path/to/datasets --output results
python summarize.py --results results
```

The exploratory inference-only experiment reuses the uniform episodic encoder.
It compares the ordinary prototype mean against an inverse-neighborhood-count
weighted mean and a shuffled-weight control on the no-overlap cohort:

```sh
python support_density.py --data-root /path/to/datasets --results results --output results/support-density
```

The JSON outputs identify train/reserved center indices and report per-cohort
sample counts, class counts, OA, AA, Kappa and confusion matrices. The summary
prints mean ± sample standard deviation and paired spatial-weight effects with
descriptive seed-level confidence intervals. Coverage accompanies metrics because
filtering can remove rare classes from a cohort.

## References

- Liu et al., *Deep Few-Shot Learning for Hyperspectral Image Classification*,
  IEEE TGRS, 2019. [DOI](https://doi.org/10.1109/TGRS.2018.2872830)
- Snell et al., *Prototypical Networks for Few-shot Learning*, NeurIPS, 2017.
  [Paper](https://proceedings.neurips.cc/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html)
- Geiß et al., *On the Effect of Spatially Non-Disjoint Training and Test Samples
  on Estimated Model Generalization Capabilities in Supervised Classification With
  Spatial Features*, IEEE GRSL, 2017. [DOI](https://doi.org/10.1109/LGRS.2017.2747222)
