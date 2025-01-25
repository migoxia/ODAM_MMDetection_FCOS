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
- for DETR, modify python3.8/site-packages/torch/nn/modules/transformer.py:
```
# in TransformerDecoder.forward(), line 389
cnt=1
qs=[]
ks=[]
vs=[]
for mod in self.layers:
    output,q,k, sa_block_output = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)
    qs.append(q)
    ks.append(k)
    vs.append(sa_block_output)
   
    cnt+=1

if convert_to_nested:
    output = output.to_padded_tensor(0., src.size())

if self.norm is not None:
    output = self.norm(output)

return output,qs,ks,vs
```

```
# in TransformerEncoderLayer.forward(), line 718
x = src
q=x
k=x
if self.norm_first:
    x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
    x = x + self._ff_block(self.norm2(x))
else:
    sa_block_output=self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
    x = self.norm1(x + sa_block_output)
    x = self.norm2(x + self._ff_block(x))
return x,q,k,sa_block_output
```
