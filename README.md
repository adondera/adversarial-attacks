# Gradient Based Adverserial Attacks

Implemented attacks:
1. [ZOO: Zeroth Order Optimization Based Black-box Attacks](https://dl.acm.org/doi/10.1145/3128572.3140448):  
**Summary**: Estimate the gradients of the target DNN model based on querying the original model multiple times. To decrease 
rom 2p queries (p=dim of representation), authors used Newton and Adam Optimization - Zoo-ADAM <br>
**Inputs from the model needed**: Probability distribution over class labels  <br>
**Targeted or untargeted**: Both (Only untargeted attacks are implemented) <br>
**Code Refernced From**: [GitHub - IBM/ZOO-Attack](https://github.com/IBM/ZOO-Attack) and [GitHub - ZOO-Attack_PyTorch ](https://github.com/as791/ZOO_Attack_PyTorch)

2. [Query-Efficient Hard-label Black-box Attack:An Optimization-based Approach](https://arxiv.org/abs/1807.04457):  
**Summary**: Improvement over ZOO. Instead of top-K class probabilities or class labels, access to only one hard label. 
Solved using Randomized Gradient Free (RGF) Method which is based on ZOO. Also effective for discrete or non-continuos 
models like s Gradient Boosting Decision Trees (GBDT) <br>
**Inputs from the model needed**: Only the final hard class label <br>
**Targeted or untargeted**: Both (Only untargeted attacks are implemented) <br>
**Code Refernced From**: [Blackbox attacks for deep neural network models](https://github.com/LeMinhThong/blackbox-attack)

3. [Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models](https://arxiv.org/abs/1712.04248):  
**Summary**: <br>
**Inputs from the model needed**: Only the final hard class label <br>
**Targeted or untargeted**: Both (Only untargeted attacks are implemented) <br>
**Code Refernced From**: [Blackbox attacks for deep neural network models](https://github.com/LeMinhThong/blackbox-attack)
