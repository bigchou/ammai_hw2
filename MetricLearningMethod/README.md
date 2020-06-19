# Apply Metric Learning on few-shot task

### In this part, we introduce the angular softmax into the ProtoNet.

##### Meta-Training "ProtoNet + Angular Softmax" (only trained on miniImageNet)

Note. You should check that <strong>trunk.append(SphereFace(in_features=indim, out_features=indim, device_id=[0], m=3))</strong> exists in <strong>backbone.py</strong> and then run the following code.

```bash
    python train.py --dataset miniImageNet --model ResNet10_asoftmax --method protonet_asoftmax --n_shot 5 --train_aug
```

##### Meta-Test "ProtoNet + Angular Softmax" for Few-Shot task

Note. <strong>trunk.append(SphereFace(in_features=indim, out_features=indim, device_id=[0], m=3))</strong> should be commented out. Then, run the following code.

```bash
    python meta_test_few_shot_models.py --task fsl --model ResNet10_asoftmax --method protonet_asoftmax --train_aug --freeze_backbone
```