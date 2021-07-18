# PTGCN
Position-enhanced and Time-aware Graph Convolutional Network for Sequential Recommendations

The sequential recommendation (also known as the next-item recommendation), which aims to predict the following item to recommend in a session according to users’ historical behavior, plays a critical role in improving session-bassed recommender systems. Most of the existing deep learning-based ap-proaches utilize the recurrent neural network architecture or self-attention to model the sequential pat-terns and temporal influence among a user’s historical behavior and learn the user’s preference at a specific time. However, these methods have two main drawbacks. First, they focus on modeling users’ dynamic states from a user-centric perspective and always neglect the dynamics of items over time. Second, most of them deal with only the first-order user-item interactions and do not consider the high-order connectivity between users and items, which has recently been proved helpful for the se-quential recommendation. To address the above problems, in this article, we attempt to model us-er-item interactions by a bipartite graph structure and propose a new recommendation approach based on a Position-enhanced and Time-aware Graph Convolutional Network (PTGCN) for the sequential recommendation. PTGCN models the sequential patterns and temporal dynamics between user-item interactions by defining a position-enhanced and time-aware graph convolution operation and learning the dynamic representations of users and items simultaneously on the bipartite graph with a self-attention aggregator. Also, it realizes the high-order connectivity between users and items by stacking multi-layer graph convolutions. To demonstrate the effectiveness of PTGCN, we carried out a comprehensive evaluation of PTGCN on three real-world datasets of different sizes compared with a few competitive baselines. Experimental results indicate that PTGCN outperforms several state-of-the-art models in terms of two commonly-used evaluation metrics for ranking. In particular, it can make a better trade-off between recommendation performance and model training efficiency, which holds great potential for online session-based recommendation scenarios in the future.

Next, we introduce how to run our model for provided example data or your own data.

# Environment

Python 3.7

Pytorch 1.7.1

Numpy 1.15.0

# Usage
As an illustration, we provide the data and running command for MovieLens-1M.

# Input data
ratings.dat：UserID::MovieID::Rating::Timestamp.

# Contact
Liwei Huang, dr_huanglw@163.com

# Citation
If you use PTGCN in your research, please cite our paper:

Liwei Huang, Yutao Ma, Yanbo Liu, Shuliang Wang, Deyi Li. 2021. Position-enhanced and Time-aware Graph Convolutional Network for Sequential Recommendations. http://arxiv.org/abs/2107.05235
