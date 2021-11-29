***Active Learning Using Uncertainty Estimation***

This repo contains the code related to our paper, **Efficacy of Bayesian Neural Networks in Active Learning** If you are using this code, please consider citing our work:

Rakesh, V., & Jain, S. (2021). [Efficacy of Bayesian Neural Networks in Active Learning](https://openaccess.thecvf.com/content/CVPR2021W/LLID/html/Rakesh_Efficacy_of_Bayesian_Neural_Networks_in_Active_Learning_CVPRW_2021_paper.html). In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 2601-2609).

In our work, we use the following neural networks to estimate uncertainty and consequently perform active learning (AL):

1. Monte Carlo Dropout (MCD)
2. Ensemble-based (EN)
3. Bayesian neural network (BNN)

MCD and EN based codes can be found inside the folder ```MCD_EN_ActiveLearning``` and the codes related to BNN can be found inside ```BNN_ActiveLearning```. Readme files can be found inside their respective folders.