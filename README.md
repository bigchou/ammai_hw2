# NTU-AMMAI-CDFSL-HW2

### The repo is modified from https://github.com/JiaFong/NTU_AMMAI_CDFSL

### Datasets
   * mini-ImageNet: https://drive.google.com/file/d/1zGGCKzspL0GSZhiDEp8rpVnB4i-jL2k_/view?usp=sharing
   * EuroSAT: https://drive.google.com/file/d/1DUj510w7726Iga34gmmhLA4meVYULrBe/view?usp=sharing
   * ISIC: https://drive.google.com/file/d/15jXP_rDEi_eusIK-kCz-DD5ZBFD3WeGF/view?usp=sharing



## Description
   See https://docs.google.com/document/d/1rNuAb3D0dcXI776eKrj8iNpE2LQmCE63WU7lRTOQGIU/edit?usp=sharing

### Specific Tasks:

   **EuroSAT**

     • Shots: n = {5}

   **ISIC2018**

     • Shots: n = {5}


### Environment
   Python 3.7
   
   Pytorch 1.3.1

### Steps
   1. Download all the needed datasets via above links.
   2. Change configuration in config.py to the correct paths in your own computer.
   3. Train models on miniImageNet. (Note: You can only train your own model, other pretrained models are provided.)
   - **Standard supervised learning on miniImageNet**

       ```bash
           python ./train.py --dataset miniImageNet --model ResNet10  --method baseline --train_aug
       ```
   - **Train meta-learning method (protonet) on miniImageNet**
   
       The available method list: [protonet]

       ```bash
           python ./train.py --dataset miniImageNet --model ResNet10  --method protonet --n_shot 5 --train_aug
       ```
   4. Test
      You should know the following options:

      • --task: fsl/cdfsl, option for task 1(fsl) or task 2(cdfsl).

      • --model: ResNet10, network architecture.

      • --method: baseline/protonet/your-own-model.

      • --train_aug: add this if you train the model with this option.

      • --freeze_backbone: add this for inferring directly. (Do not add this if you want to fine-tune your model, you should only fine-tune models in task 2.)

      There are two meta-test files:

      * **meta_test_Baseline.py**:
      
        For Baseline, we will train a new linear classifier using support set.

        ```bash
            python meta_test_Baseline.py --task cdfsl --model ResNet10 --method baseline  --train_aug --freeze_backbone
        ```
         You can also train a new linear layer and fine-tune the backbone.

        ```bash
            python meta_test_Baseline.py --task cdfsl --model ResNet10 --method baseline  --train_aug
        ```

      * **meta_test_few_shot_models.py**:
      
        This method will apply the pseudo query set to the few-shot model you want to fine-tune with. 

        The available method list: [protonet]

        The available model list: [ResNet10]
        
        ```bash
            python meta_test_few_shot_models.py --task cdfsl --model ResNet10 --method protonet  --train_aug
        ```

   No matter which finetune method you chosse, a dataset contains 600 tasks.

   After evaluating 600 times, you will see the result like this: 600 Test Acc = 49.91% +- 0.44%.

### Results

| Models  | miniImageNet | EuroSAT | ISIC |
| ------------- | ------------- | ------------- | ------------- |
| Baseline (TA's) | 68.10% ± 0.67% | 75.69% ± 0.66% / 79.08% ± 0.61% | 43.56% ± 0.60% / 48.11% ± 0.64% | 
| ProtoNet (Ta's) | 66.33% ± 0.65% | 77.45% ± 0.56% / 81.45% ± 0.63% | 41.73% ± 0.56% / 46.72% ± 0.59% |
| TPN (Liu et al., 2019) | 69.43% | - / - | - / - |
| Baseline (Ours) | 66.53% ± 0.66% | 76.37% ± 0.69% / 78.95% ± 0.62% | 44.31% ± 0.58% / 48.87% ± 0.62% | 
| ProtoNet (Ours) | 66.20% ± 0.67% | 78.44% ± 0.67% / 82.09% ± 0.63% | 42.12% ± 0.56% / 46.93% ± 0.58% |
| TPN (Ours) | 67.30% ± 0.65% | 64.41% ± 0.80% / 82.07% ± 0.67% | 33.86% ± 0.54% / 44.09% ± 0.60% |
| ProtoNet + Angular Softmax | 69.32% ± 0.66% | - / - | - / - |



For EuroSAT and ISIC, the result w/o and w/ fine-tuning are the first and second accuracy, respectively.



### Commands

Meta-Train Baseline (only trained on miniImageNet)

```bash
    python ./train.py --dataset miniImageNet --model ResNet10  --method baseline --train_aug
```

Meta-Test Baseline for Few-Shot task

```bash
    python meta_test_Baseline.py --task fsl --model ResNet10 --method baseline  --train_aug --freeze_backbone
```

Meta-Test Baseline for Cross-Domain Few-Shot task

```bash
    python meta_test_Baseline.py --task cdfsl --model ResNet10 --method baseline  --train_aug 
```

Meta-Train ProtoNet (only trained on miniImageNet)

```bash
    python ./train.py --dataset miniImageNet --model ResNet10  --method protonet --n_shot 5 --train_aug
```

Meta-Test ProtoNet for Few-Shot task

```bash
    python meta_test_few_shot_models.py --task fsl --model ResNet10 --method protonet  --train_aug --freeze_backbone
```

Meta-Test ProtoNet for Cross-Domain Few-Shot task

```bash
    python meta_test_few_shot_models.py --task cdfsl --model ResNet10 --method protonet  --train_aug
```

Meta-Train TPN (only trained on miniImageNet)

```bash
    python ./train.py --dataset miniImageNet --model ResNet10  --method mytpn --n_shot 5 --train_aug
```

Meta-Test TPN for Few-Shot task

```bash
    python meta_test_few_shot_models.py --task fsl --model ResNet10 --method mytpn  --train_aug --freeze_backbone
```

Meta-Test TPN for Cross-Domain Few-Shot task

```bash
    python meta_test_few_shot_models.py --task cdfsl --model ResNet10 --method mytpn  --train_aug
```

Meta-Train TPN (Meta-Training using EuroSAT and miniImageNet)

```bash
    python ./train_adapt.py --dataset miniImageNet --model ResNet10  --method mytpnadapteurosat --n_shot 5 --train_aug
```

Meta-Test TPN for Few-Shot task (Note. the checkpoint is obtained after meta-training on EuroSAT and miniImageNet)

```bash
    python meta_test_few_shot_models.py --task fsl --model ResNet10 --method mytpnadapteurosat  --train_aug --freeze_backbone
```

Meta-Test TPN for Cross-Domain Few-Shot task (Note. the checkpoint is obtained after meta-training on EuroSAT and miniImageNet)

```bash
    python meta_test_few_shot_models.py --task cdfsl --model ResNet10 --method mytpnadapteurosat  --train_aug
```

Meta-Train TPN (Meta-Training using ISIC and miniImageNet)

```bash
    python ./train_adapt.py --dataset miniImageNet --model ResNet10  --method mytpnadaptisic --n_shot 5 --train_aug
```

Meta-Test TPN for Few-Shot task (Note. the checkpoint is obtained after meta-training on ISIC and miniImageNet)

```bash
    python meta_test_few_shot_models.py --task fsl --model ResNet10 --method mytpnadaptisic  --train_aug --freeze_backbone
```

Meta-Test TPN for Cross-Domain Few-Shot task (Note. the checkpoint is obtained after meta-training on ISIC and miniImageNet)

```bash
    python meta_test_few_shot_models.py --task cdfsl --model ResNet10 --method mytpnadaptisic  --train_aug
```


### Metirc Learning?

Please see <strong>MetricLearningMethod</strong> folder.


### Reference

[1] [NTU_AMMAI_CDFSL](https://github.com/JiaFong/NTU_AMMAI_CDFSL)

[2] [TPN-pytorch](https://github.com/csyanbin/TPN-pytorch)

[3] [face.evoLVe.PyTorch](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)