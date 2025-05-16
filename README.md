# APCoTTA: Continual Test-Time Adaptation for Semantic Segmentation of Airborne LiDAR Point Clouds

Airborne laser scanning (ALS) point cloud segmentation is a fundamental task for large-scale 3D scene understanding. In realworld
applications, models are typically fixed after training. However, domain shifts caused by changes in the environment, sensor
types, or sensor degradation often lead to a decline in model performance. Continuous Test-Time Adaptation (CTTA) offers a
solution by adapting a source-pretrained model to evolving, unlabeled target domains during deployment. Despite its potential,
research on ALS point clouds remains limited, facing challenges such as the absence of standardized datasets for ALS CTTA tasks
and the risk of catastrophic forgetting and error accumulation during prolonged adaptation. To tackle these challenges, we propose
APCoTTA, the first CTTA method tailored for ALS point cloud semantic segmentation. We propose a dynamic trainable layer
selection module. This module utilizes gradient information to identify low-confidence layers with near-uniform distributions.
These layers are selected for training, and the remaining layers are kept frozen. This design helps retain critical source domain
knowledge, mitigating catastrophic forgetting. To further reduce error accumulation, we propose an entropy-based consistency loss.
Extremely low-confidence samples often cause unstable gradients and model collapse. By losing such samples based on entropy,
we apply consistency loss only to the reliable samples, enhancing model stability. In addition, we propose a random parameter
interpolation mechanism, which randomly blends parameters from the selected trainable layers with those of the source model.
This approach helps balance target adaptation and source knowledge retention, further alleviating forgetting. Finally, we construct
two benchmarks, ISPRSC and H3DC, to address the lack of CTTA benchmarks for ALS point cloud segmentation. Experimental
results demonstrate that APCoTTA achieves the best performance on two benchmarks, with mIoU improvements of approximately
9% and 14% over direct inference.


# Note
We will release our code soon.

# Cite
Please cite our work if you find it useful.
```bibtex
@misc{gao2025apcottacontinualtesttimeadaptation,
      title={APCoTTA: Continual Test-Time Adaptation for Semantic Segmentation of Airborne LiDAR Point Clouds}, 
      author={Yuan Gao and Shaobo Xia and Sheng Nie and Cheng Wang and Xiaohuan Xi and Bisheng Yang},
      year={2025},
      eprint={2505.09971},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.09971}, 
}
```
