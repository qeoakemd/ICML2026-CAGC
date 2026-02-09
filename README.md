# ICML2026-CAGC

This contain anonymous github repository for submission "Communication-Efficient Approximate Gradient Coding for Distributed Learning in Heterogeneous Systems" to ICML 2026.

To reproduce our results, users can freely adjust both the model architecture and dataset size through simple configuration flags in the released code. Because our study centers on convergence speed rather than task-specific accuracy, the methodology is model- and dataset-agnostic and can be applied to any architecture–dataset pair. (Users can change the model (to ResNet-50 ,Yolov5, etc.), dataset size, testset size, learning rate, batch size, crop size, etc. manually, according to the computer resources.)

Download COCO dataset
Generate file named 'data' and move the dataset into 'data'
Run 'Simulation_COCO_.py'
