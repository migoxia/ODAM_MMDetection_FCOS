# ODAM Using Object Relationships
This repository implements explanations for object detectors using object relationships, based on the previous work ODAM: https://github.com/Cyang-Zhao/ODAM. This is a demo version using FCOS.
- Install mmdetection according to [MMDetection Installation](https://mmdetection.readthedocs.io/en/latest/get_started.html), and put the odam_onestage_fcos.ipynb into mmdetection folder.
- To preserve gradient in the inference process, change all "detach=True" in ./mmdet/models/dense_heads/base_dense_head.py to "detach=False".
- make dir "./checkpoints", download corresponding FCOS weights from [FCOS](https://github.com/open-mmlab/mmdetection/tree/main/configs/fcos) based on the adopted config file, and put it into "./checkpoints".
- make dir "./data", download MS COCO validation dataset and annotation file and put into "./data".
- for Faster-RCNN, modify method extract_feat in module mmdet.models.detectors.two_stage:
```
def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x_after_neck = self.neck(x)
        return x,x_after_neck
```
