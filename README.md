# WACV2025 MVMD code
![WACV-2025-Logo_Color-1024x315](https://github.com/user-attachments/assets/ecbf2956-da75-4a39-b4b0-e90124f3e9bd)

# MVMD: A Multi-View Approach for Enhanced Mirror Detection

[[PAPER](https://openaccess.thecvf.com/content/WACV2025/papers/Shen_MVMD_A_Multi-View_Approach_for_Enhanced_Mirror_Detection_WACV_2025_paper.pdf)] [[SUPPLEMENT](https://openaccess.thecvf.com/content/WACV2025/supplemental/Shen_MVMD_A_Multi-View_WACV_2025_supplemental.pdf)]

## Abstract
In 3D reconstruction, mirrors introduce significant challenges by creating distorted and fragmented spaces, resulting in inaccurate and unreliable 3D models. As 3D reconstruction typically relies on multi-view images to capture different perspectives of a scene, detecting and labeling mirrors in multi-view images before reconstruction can effectively address this issue. However, existing methods focus solely on single-image detection, overlooking the rich information provided by multi-view setups. To overcome this limitation, we propose MVMD, a novel Multi-View Mirror Detection method, along with the first database specifically designed for mirror detection in multi-view scenes.
The design of MVMD is grounded in the inherent associations between objects seen from different views and those reflected inside and outside of mirrors. These relationships are learned through cross- and self-attention mechanisms. MVMD consists of three key blocks: the Inter-Views Block tracks the shifts of objects within mirrors caused by changes in viewpoint; the Intra-View Block detects object reflections inside mirrors; and the Refinement Block sharpens mirror boundaries and enhances detected details.
Experimental results show that our method improves accuracy by up to 2.6% and IoU by up to 11.1%, compared to single-image mirror detection techniques. This substantial improvement makes MVMD particularly effective for computer vision tasks, especially in enhancing the accuracy of 3D reconstruction in mirror-dense environments. 

## Architecture
![Screenshot 2025-02-27 154312](https://github.com/user-attachments/assets/90835bd1-051f-4527-9219-b3ea99e44749)


## Experiment
[mvmd(compare).pdf](https://github.com/user-attachments/files/19016544/mvmd.compare.pdf)


## Citation
```bibtex
@InProceedings{Shen_2025_WACV,
    author    = {Shen, Yidan and Wen, Yu and Zhang, Chen and Fu, Xin and Hu, Renjie},
    title     = {MVMD: A Multi-View Approach for Enhanced Mirror Detection},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {9314-9323}
}
```
