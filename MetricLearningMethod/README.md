# Apply Metric Learning on few-shot task

# In this part, we introduce the angular softmax into the ProtoNet.

Meta-Training "ProtoNet + Angular Softmax" (only trained on miniImageNet)

```bash
    python train.py --dataset miniImageNet --model ResNet10_asoftmax --method protonet_asoftmax --n_shot 5 --train_aug
```

Meta-Test "ProtoNet + Angular Softmax" for Few-Shot task

```bash
    python meta_test_few_shot_models.py --task fsl --model ResNet10_asoftmax --method protonet_asoftmax --train_aug --freeze_backbone
```