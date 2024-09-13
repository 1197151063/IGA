# IGA Pytorch

## Official pytorch Implementation of IGA

The original version of this code base was from LightGCN-pytorch: https://github.com/gusye1234/LightGCN-PyTorch.



## Abstract

Graph Neural Networks (GNNs) has become a prominent backbone model in recommender systems, credited to its exceptional ability to capture intricate topological signals within user-item interactions. However, many existing studies mistakenly assume that these interactions are inherently reliable, overlooking the fact that a significant portion of user-item engagements, such as accidental clicks, are actually noisy. These noisy edges can mislead the network to overfit incorrect interaction patterns, weakening the graph's collaborative signals and ultimately degrading recommendation performance. To tackle these challenges, we introduce a novel Instance-dependent Graph Augmentation (IGA) method for recommendation. The proposed IGA innovatively employs a edge pruning strategy to filter out unreliable interactions based on the memorization effect of Deep Neural Networks (DNNs) and further utilizes the memorization-dependent information to generate augmented views for self-supervised tasks. Comprehensive experiments and ablation studies validate the effectiveness and robustness of proposed method.



## Requirements

torch_geometric==2.5.3

numpy==1.24.3

scipy==1.10.1

torch-sparse==0.6.17+pt20cu118

torch==2.0.1






