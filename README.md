# Learning to Infer Generative Template Programs for Visual Concepts

By [R. Kenny Jones](https://rkjones4.github.io/), [Siddhartha Chaudhuri](https://www.cse.iitb.ac.in/~sidch/), and [Daniel Ritchie](https://dritchie.github.io/)

![Overview](https://rkjones4.github.io/img/template/main_result.png)

We develop a neurosymbolic method that learns how to infer Template Programs; partial programs that capture visual concepts in a domain-general fashion. 
 
## About the paper

[Paper](https://rkjones4.github.io/pdf/template.pdf)

[Project Page](https://rkjones4.github.io/template.html)

Presented at [ICML 2024](https://icml.cc/).

## Bibtex
```
@inproceedings{jones2024TemplatePrograms,
  title= {Learning to Infer Generative Template Programs for Visual Concepts},
  author= {Jones, R. Kenny and Chaudhuri, Siddhartha and Ritchie, Daniel},
  booktitle = {International Conference on Machine Learning (ICML)},
  year= {2024}
}
```

# General Info

This repository contains code and data for the experiments in the above paper.

*Code*: We release a reference implementation of our template programs framework, and support for two concept-related tasks: few-shot generation and co-segmentation. We also provide training scripts that were used to develop these models across three visual programming domains: 2D primitive layouts (layout), Omniglot characters (omni), and 3D shape structures (shape).

*Data*: We release our target datasets that we finetune towards for each domain.

# Data and Pretrained Models

Google drive download links for:

[Pretrained Models](https://drive.google.com/file/d/1h9xKqO40vgMJ6A_np10DzyFjQR-pBtk_/view?usp=sharing)

[Layout Target Data](https://drive.google.com/file/d/1aJDFDIL58-qBpjeFucuAvnPJKrn8r7qQ/view?usp=sharing)

[Omni Target Data](https://drive.google.com/file/d/1m3sFXQaDf8nHYy4JNlgs9sJj_BLhrYxN/view?usp=sharing)

[Shape Target Data](https://drive.google.com/file/d/1UX0hXNO4IBvcwUmIUMtz6vtlKhwaZ4M0/view?usp=sharing)

Please unzip each of these files from the root of this directory.

For each domain, we include the following pretrained models:
- pre_net.pt --> network pretraind on synthetic data
- ft_inf_net.pt -> network trained on reconstruction task (main model)
- ft_gen_net.pt -> network trained on unconditional generative task (used during wake-sleep step of fine-tuning)
- ft_magg_net.pt -> network trained on conditional generative task (used during few-shot generation)

# Concept-related Tasks

**Commands**

Each task/training command started from main.py requires setting an experiment name and domain name (EXP_NAME and DOMAIN).

EXP_NAME can be any string (results will be saved to model_output/{EXP_NAME})

Domain can be one of: layout / omni / shape (see main.py)

By default the `shape' domain will use the primitive soup input. To instead use a voxel input, add the following arguments to any call to main.py :

```
 -vt voxel -vd 64
```

**Few-shot generation**

The main logic for our few-shot generation task can be found in fsg_eval.py

To run a few-shot generation experiment, you can use a command like:

```
python3 main.py -en {EXP_NAME} -dn {DOMAIN} -mm fsg_eval -lmp pretrained_models/{DOMAIN}/ft_inf_net.pt -lgmp pretrained_models/{DOMAIN}/ft_magg_net.pt
```

This will save few-shot generation results to model_output/{EXP_NAME} . In the vis folder each image is a separate concept input: the top row is the input, the middle row is the reconstruction, the bottom row is the few-shot generation.

*Notes*:

To run few-shot generation with the more expensive inference procedure (used in the paper) include the following arguments:

```
 -tbm 40 -ebm 10
```

When running few-shot generation for voxelized shapes, rendering voxel fields with matplotlib can take a prohibitively long time. 

**Co-segmentation**

The main logic for our co-segmentation task can be found in coseg_task.py

To run a co-segmentation experiment, you can use a command like:

```
python3 main.py -en {EXP_NAME} -dn omni -mm coseg -lmp pretrained_models/{DOMAIN}/ft_inf_net.pt
```

This will print out co-segmentation metrics and save results to model_output/{EXP_NAME} . In the vis folder each image is a separate concept input: the top row is the input, the second row is the reconstruction, the third row is the parse, the fourth row transfers the parse regions to the input shapes, the fifth row is the predicted segmentation, the sixth row is the ground-truth segmentation.

*Notes*:

Once again, this can be run with the more expensive inerence procedure (used in the paper) by including the following arguments:

```
 -tbm 40 -ebm 10
```

For shapes, we save only three rows (for time consideration), these correspond to rows 4-6 from the above description.

# Training new models

**Pretraining**

To start a new pretaining run, you can use the following command:

```
python3 main.py -en {EXP_NAME} -dn {DOMAIN} -mm pretrain 
```

To visualize what the synthetic concepts look like, you can use a command like:

```
python3 executors/common/test.py {DOMAIN} vis 10 5
```

This will render 10 synthetic concepts that each have 5 members.

**Finetuning**

To start a new finetuning run, you can use the following command:

```
python3 main.py -en {EXP_NAME} -dn {DOMAIN} -mm finetune -lmp pretrained_models/{DOMAIN}/pre_net.pt
```

**Conditional generative model**

To train a new mean aggregation network, which is used for the few-shot generation task, first you should complete a finetuning run. Say this finetuning run was saved under the experiment name FT_EXP_NAME, then you can train a new conditional generative model with the following command:

```
python3 main.py -en {EXP_NAME} -dn {DOMAIN} -mm train_magg -lmp pretrained_models/{DOMAIN}/pre_net.pt -mtdp model_output/{FT_EXP_NAME}/train_out/
```

# Code Structure

**Folders**:

*data* --> target set data and logic

*domains* --> visual programming domains

*executors* --> execution logic for each domains

*model_output* --> where experiment logic is saved

*models* --> logic for our networks

**Files**:

*coseg_task.py* --> logic for co-segmentation

*finetune.py* --> logic for finetuning

*fsg_eval.py* --> logic for few-shot generation

*main.py* --> main training/task entrypoint

*pretrain.py* --> logic for pretraining

*prob_infer.py* --> logic for inference step during finetuning

*search.py* --> logic for inference during few-shot generation / co-segmentation

*train_magg_net.py* --> logic for training conditional generative network for few-shot generation

*train_prob_plad.py* --> logic for training reconstruction network during finetuning

*train_utils.py* --> general training related utility functions

*utils.py* --> general utility functions

*wake_sleep.py* --> logic for training and sampling unconditional generative network during finetuning

# Dependencies

This code was tested on Ubuntu 20.04, an NVIDIA 3090 GPU, python 3.9.7, pytorch 1.9.1, and cuda 11.1

The environment this code was developed in can be found in env.yml

# Acknowledgments

We would like to thank the anonymous reviewers for their helpful suggestions and all of our perceptual study particpants for their time. Renderings of 3D shapes were produced using the Blender Cycles renderer. This work was funded in parts by NSF award #1941808 and a Brown University Presidential Fellowship. Daniel Ritchie is an advisor to Geopipe and owns equity in the company. Geopipe is a start-up that is developing 3D technology to build immersive virtual copies of the real world with applications in various fields, including games and architecture. Part of this work was done while R. Kenny Jones was an intern at Adobe Research.

