# ODAM
- Install mmdetection according to [MMDetection Installation](https://mmdetection.readthedocs.io/en/latest/get_started.html), and put the odam_onestage_fcos.ipynb into mmdetection folder.
- To preserve gradient in the inference process, change all "detach=True" in ./mmdet/models/dense_heads/base_dense_head.py to "detach=False".
- make dir "./checkpoints", download corresponding FCOS weights from [FCOS](https://github.com/open-mmlab/mmdetection/tree/main/configs/fcos) based on the adopted config file, and put it into "./checkpoints".
- make dir "./data", download MS COCO validation dataset and annotation file and put into "./data".
