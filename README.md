# MSTGCN-LAM-DMGAB
Leveraging Pretrained Local Augmentation Modules for Enhanced Traffic Flow Prediction in MSTGCN
## Introduction
We propose a pretrainable local augmentation module (LAM) that augments a multi-component spatial-temporal graph convolution network (MSTGCN), which is equipped with a dynamic multi-graph attention block (DMGAB). This module adeptly captures the distribution characteristics of long-term historical time series and leverages the adjacency relationships among nodes to generate supplementary features, encapsulating local information for each node. This augmentation significantly enriches the contextual information for predictions. Additionally, we propose a cluster feature correlation graph to uncover hidden correlations in long-term time series data. Experiments on the METR-LA dataset demonstrate significant reductions in prediction errors.
## Datasets
### Histocical Records
- METR-LA is a traffic speed dataset collected from loopdetectors located on the LA County road network [14]. It contains data of 207 selected sensors over a period of 4 months from Mar to Jun in 2012. The traffic information is recorded at the rate of every 5 minutes, and the total number of time slices is 34,272.
