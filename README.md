# HONTF: HOCP and HOTucker

MATLAB code for **Hypergraph regularized orthogonal non-negative tensor factorization for semi-supervised clustering**, Xianxiu Zhang, Baicheng Pan, Xiangguang Dai, Xiangqin Dai and Hangjun Che, *Internet of Things*, 38 (2026), 101986.

[Paper / DOI](https://doi.org/10.1016/j.iot.2026.101986)

This release includes both methods from the paper:

- **HOCP**: the main method using CP decomposition.
- **HOTucker**: the extension using Tucker decomposition.

Both demos run the complete COIL20 dataset with the same seeded sample ordering and labeled/unlabeled split. Core optimization formulas and objective routines are retained from the author's original experiment code.

## Requirements

- MATLAB with **Statistics and Machine Learning Toolbox** (`KDTreeSearcher`, `knnsearch`, `randsample`).
- Tested with **MATLAB R2024b Update 3**.
- The required dense Tensor Toolbox 3.6 components and GSPbox 0.7.0 helpers are included. No separate toolbox download, compilation, or FLANN binaries are needed for these demos.

## Quick start

Clone or download the repository, change MATLAB's current folder to its root, and run either or both methods:

```matlab
cp_result = demo_hocp_COIL20;
tucker_result = demo_hotucker_COIL20;
```

For ten complete training runs per method:

```matlab
cp_result = demo_hocp_COIL20(10);
tucker_result = demo_hotucker_COIL20(10);
```

Or, from a terminal in the repository folder:

```sh
matlab -batch "demo_hocp_COIL20; demo_hotucker_COIL20"
```

The entry functions call `setup_hontf` automatically. Each entry resets the seed to 0, so their first run uses identical samples and label selection. Later repeated runs consume method-specific random numbers and can have different splits between methods. The default is **one complete training run**; its standard deviation is zero and does not measure run-to-run variation.

## Data and semi-supervised split

`data/COIL20.mat` is copied from the author's shared dataset directory. It contains `fea` (1440 x 1024, one sample per row) and `gnd` (1440 x 1).

The shared preparation routine retains the original demo's row normalization, random class ordering and within-class label selection. It uses all 20 classes and all 72 samples per class. Samples are reshaped to 32 x 32 images for tensor factorization.

For each class, `floor(0.3 * 72) = 21` samples are labeled: **420 labeled and 1020 unlabeled samples**, an actual labeled fraction of 29.17%. The labeled rows precede the unlabeled rows, as required by the original algorithms. The labeled indicator is fixed and scaled so that its column Gram matrix is `I/2`.

The legacy core interfaces accept both labeled and unlabeled reference labels. They use the latter for sample counts and class enumeration; the demos ensure every class has labeled samples. Saved splits record the original sample indices, relabeled reference labels, class order and labeled mask for inspection.

## Settings

| Setting | HOCP | HOTucker |
| --- | --- | --- |
| Image size | 32 x 32 | 32 x 32 |
| Classes / samples | 20 / 1440 | 20 / 1440 |
| CP rank / Tucker dimensions | 20 | [32, 32, 20] |
| Hypergraph penalty `lambda` | 1000 | 1000 |
| Orthogonality penalty `beta` | 100 | 1 |
| Hypergraph neighbors | 5 | 5 |
| Maximum iterations | 500 | 200 |
| Relative stopping tolerance | 1e-5 | 1e-5 |
| Final k-means replicates | 20 | 20 |

Penalties and iteration limits come from the supplied COIL20 experiment scripts. They are demonstration settings, not a re-run of the paper's parameter search. The original CP entry selected 10 classes; this release selects all 20 to share the full dataset with Tucker. The redundant parameter loops in the original demos are omitted.

HOCP clusters rows of `C`. HOTucker clusters `C * G_(3)`, where `G_(3)` is the mode-3 matricization of the core tensor, as in the original Tucker demo.

## Results

Outputs are saved to `results/HOCP_COIL20.mat` and `results/HOTucker_COIL20.mat`, each containing a `result` struct. Generated results are ignored by Git.

- `metrics_all` and `metrics_unlabeled`: 7 x number-of-runs matrices. Rows are **ACC, NMI, purity, F-score, precision, recall, ARI**.
- `mean_all`, `std_all`, `mean_unlabeled`, `std_unlabeled`: aggregate metrics.
- `objective_history`, `predicted_labels`, `splits`, `training_seconds`: records for every run.
- `A`, `B`, `C`, and Tucker's numeric `G`: factor matrices/core for the **last** run.
- `options`, `seed`, `nRun`: experiment settings.
- `last_nIter`: the original core's legacy counter; `last_update_count` counts recorded objective values and can be one larger when stopping early.

The all-sample metrics follow the original demos and include labeled samples. Unlabeled-only metrics evaluate the unlabeled rows of the same clustering assignment. Both are reported explicitly. Original update order, returned factors and objective logging are preserved; the stored objective histories are the original implementation's diagnostics.

## Validation

Both methods completed on all 1440 samples from MATLAB's default path using only the bundled helper paths. The split was checked to be identical for their first run, and representation dimensions, finite values, nonnegativity and clustering metric ranges were checked.

| Method | Recorded updates | ACC, all | NMI, all | ARI, all | ACC, unlabeled |
| --- | --- | --- | --- | --- | --- |
| HOCP | 73 | 0.7847 | 0.8924 | 0.7039 | 0.7990 |
| HOTucker | 63 | 0.6222 | 0.6874 | 0.3264 | 0.5559 |

These are single-run results with seed 0, not the paper's ten-run aggregates. Numerical results may vary across MATLAB/toolbox versions and hardware.

## Layout and provenance

```text
hocp.m, CalcuObjhoCP.m            Original CP core and objective
hotucker.m, CalcuObjhoTucker.m    Original Tucker extension and objective
demo_hocp_COIL20.m                CP entry function
demo_hotucker_COIL20.m            Tucker entry function
setup_hontf.m                    Dependency paths
tools/                           Shared preparation, normalization and metrics
data/COIL20.mat                   Complete COIL20 dataset
third_party/gspbox/               Required graph helper subset
third_party/tensor_toolbox/       Dense tensor components and support routines
citation.bib                     Paper citation
```

The CP core comes from the author's `[13] HONTF-CP` directory; Tucker comes from `[14] HONTF-Tucker`. Shared data, metrics and hypergraph routines come from the surrounding experiment collection. `NormalizeFea` comes from its RNTF dependency directory. The four algorithm/objective files retain their original formulas; only source encoding and trailing whitespace are normalized for publication. The entry functions consolidate the original data preparation and add relative paths, a seed, saved results, output checks and unlabeled-only evaluation.

Third-party copyright notices and licenses are retained:

- [GSPbox 0.7.0 license](third_party/gspbox/LICENSE): GPL-3.0-or-later, including the locally modified hypergraph Laplacian.
- [Tensor Toolbox 3.6 license](third_party/tensor_toolbox/LICENSE): BSD 2-Clause, Sandia National Labs. The included subset supports these dense tensor demos; it is not a full Tensor Toolbox installation.
- `NormalizeFea` credits Deng Cai. The clustering/evaluation helpers retain their existing attribution, including the MathWorks notice in `litekmeans`.

No additional blanket license is assigned to other author-supplied code or data. Other datasets, parameter searches, ablations and figure-generation scripts are omitted.

## Citation

Please cite the paper when using this code:

```bibtex
@article{ZHANG2026101986,
  title={Hypergraph regularized orthogonal non-negative tensor factorization for semi-supervised clustering},
  journal={Internet of Things},
  volume={38},
  pages={101986},
  year={2026},
  issn={2542-6605},
  doi={10.1016/j.iot.2026.101986},
  url={https://www.sciencedirect.com/science/article/pii/S2542660526001162},
  author={Xianxiu Zhang and Baicheng Pan and Xiangguang Dai and Xiangqin Dai and Hangjun Che},
  keywords={Internet of things, Semi-supervised clustering, Non-negative tensor factorization, CP decomposition, Tucker decomposition}
}
```

## 中文运行说明

安装 MATLAB 和 Statistics and Machine Learning Toolbox，将当前目录切换到仓库根目录，运行 `demo_hocp_COIL20`（CP 主方法）或 `demo_hotucker_COIL20`（Tucker 扩展）。COIL20 数据集及运行所需辅助函数已包含，无需另外配置 Tensor Toolbox。

两个入口默认均使用完整的 1440 个样本和固定随机种子，首轮实验的有标签/无标签划分相同。每类选取 21 个有标签样本，共 420 个有标签样本、1020 个无标签样本。传入参数 `10` 可进行十次完整训练，例如 `demo_hotucker_COIL20(10)`；后续各轮的随机划分可能因两种算法消耗的随机数不同而不同。

结果分别保存至 `results/HOCP_COIL20.mat` 和 `results/HOTucker_COIL20.mat`，同时提供全部样本和仅无标签样本的聚类指标。此版本保留两种方法的原始核心更新公式，提供单个数据集的运行流程。
