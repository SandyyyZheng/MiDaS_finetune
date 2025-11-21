## Finetune & Evaluation on MiDaS

This repository contains the code for [finetune](finetune) and [evaluation](eval) on MiDaS. We use [LoRA](https://github.com/microsoft/LoRA) for finetuning, and multiple metrics for evaluating. We finetune the models on the [NYU Depth V2](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2) dataset.

Thanks to coders [dvdhfnr](https://gist.github.com/dvdhfnr) and [praeclarum](https://gist.github.com/praeclarum) in [this gist](https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0), we refer to their implementation of the loss functions. For evaluation, we refer to the code provided [here](https://github.com/isl-org/DPT/blob/main/EVALUATION.md).

Please read [this documentation](finetune/README.md) for detailed instructions. 

Before you finetune MiDaS, please make sure to properly set up the environment and download the weights according to the instructions in the [original repository](https://github.com/isl-org/MiDaS). 