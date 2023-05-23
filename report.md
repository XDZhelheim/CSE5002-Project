# CSE5002 Project Report

12232414 董正

---

### Introduction

This project is a multi-class classification task of the nodes in a graph. In detail, the task is to utilize graph structure and six categorial features of each node to predict its class label.

To solve the problem, I first cleaned the original data and constructed the dataset. When dealing with the categorical features, one-hot embedding and trainable embedding were used and compared. Then the dataset was fed into predictive models to perform classification, where the models include a machine-learning model, a simple non-linear neural network, and two graph-based neural networks. A PyTorch training framework with clear pipeline and detailed logs was applied to train and test the models.

To summarize, the pipeline of this work is:

```
1. Raw data -> data cleaning
2. Cleaned data -> dataset construction
								including graph adjacency, node features, and train/val/test split
3. Model developing and coding
4. Dataset & model -> training framework -> model accuracy
```

This project is open-source on Github: https://github.com/XDZhelheim/CSE5002-Project.

P.S. If cannot access, please remind me to make the repo public, thanks.

### Data Preprocessing

This section introduces how to clean the raw data and how to create the dataset for model training/testing.

#### Data Cleaning

This part is shown in a Jupyter notebook [scripts/data_clean.ipynb](https://github.com/XDZhelheim/CSE5002-Project/blob/main/scripts/data_clean.ipynb).

The raw data contains four files.

- `adjlist.csv`

  Contains the adjacency list of each node.

  Format: `node_id,neighbor_1,neighbor_2,...`

  This file is used to create the adjacency matrix, which is used in GCNs.

- `attr.csv`

  Contains the attributes of each node.

  Format: `node_id,degree,gender,major,second_major,dormitory,high_school`

  Based on the name and data type (integer) of the attributes in this file, we can deduce that all the $C=6$ attributes are **categorical features**.

  Although they are of integer type, their values are not continuous, especially the `high_school` column, i.e. we should convert them to class labels as `0, 1, 2, ...`. This can be done via `pandas` or use the `LabelEncoder` in the `sklearn` package.

  It has 5298 rows, so the total number of nodes is $N=5298$.

  From this file, we can get the input feature matrix $X\in\N^{N\times C}$ of the neural models.

- `label_train.csv & label_test.csv`

  Contains the label for each node.

  Format: `node_id,label`

  First I use `pandas` to read this two files together into one `DataFrame` to get all the labels $Y\in\N^{N}$.

  Via `value_counts()` function, we can see the label distribution:

  ```bash
  >>> df_label.value_counts()
  
  label
  2008     1006
  2007      889
  2009      867
  2006      840
  2005      744
  2004      528
  2003      179
  2002       90
  2010       58
  2001       29
  2000       23
  1999        7
  1996        4
  1998        4
  1995        4
  1977        3
  1997        3
  1975        3
  1994        2
  1968        2
  1981        2
  1947        1
  1956        1
  1976        1
  1980        1
  1979        1
  1928        1
  1987        1
  1989        1
  1990        1
  1993        1
  1900        1
  dtype: int64
  
  >>> len(df_label.loc[df_label["label"]<2000])
  
  45
  ```

  We can see that there are many <2000 labels, but they only appear once or twice. If we take them into consideration, the number of classes will be very large for our classification model. What's more, they are very hard to learn because they only have one or two samples each, as mentioned before. Therefore, I decide to discard them:

  ```bash
  >>> df_label.loc[df_label["label"]<2000]=-1
  >>> df_label.value_counts()
  
  label
   2008    1006
   2007     889
   2009     867
   2006     840
   2005     744
   2004     528
   2003     179
   2002      90
   2010      58
     -1      45
   2001      29
   2000      23
  dtype: int64
  ```

  By setting them to -1, they are considered as **outliers**. When training the models, they will be ignored. But as a tradeoff, this 45 nodes that labeled as -1 will never be classified correctly, i.e. the model can never achieve 100% accuracy because I discarded them.

  As a result, there remains $n=11$ classes: 2000 ~ 2010.

#### Dataset Construction

This part is given in [scripts/gen_dataset.py](https://github.com/XDZhelheim/CSE5002-Project/blob/main/scripts/gen_dataset.py), and some also in [scripts/data_clean.ipynb](https://github.com/XDZhelheim/CSE5002-Project/blob/main/scripts/data_clean.ipynb).

After data cleaning, we have the following data:

- Adjacency list (or graph structure)
- Input matrix $X\in\N^{N\times C}$
- Label vector $Y\in\N^{N}$, containing $n=11$ classes

*Adjacency matrix.* Through the adjacency list, we can create a adjacency matrix $A\in[0, 1]^{N\times N}$. If node $i$ is connected to node $j$, then $A_{i,j}=1$, otherwise $A_{i,j}=0$. We can do this by iterating on the rows of `adjlist.csv`. Note that the diagonal will also be set to 1.

*Input matrix.* Because all the features are categorical, a straightforward solution is to use one-hot encoding. But the encode dimension will be very large. See the number of classes of each feature:

```bash
>>> df.nunique()

degree             6
gender             3
major             43
second_major      44
dormitory         64
high_school     2506
dtype: int64
```

So if we use one-hot encoding, the feature dimension is $C'=6+3+43+44+64+2506=2666$. The input matrix is $X'\in[0, 1]^{N\times C'}$.

Considering the curse of dimensionality and its sparsity, I decided to use trainable embeddings. Details will be given in the next section. But the one-hot input is kept as a comparison in the Experiment section. 

Therefore, finally the input matrix is still $X\in\N^{N\times C}$, not transformed.

*Label.* Another question is how to split train/validation/test dataset. The input $X$ cannot be split or divided into batches because it is a whole graph, which is non-separable. So we must split on the labels. I used 3500 labels for training, 500 for validation and the remaining for testing.

Note that the outliers (-1 values) should be masked in training, i.e. do not calculate loss on them. **But do not mask them in validation and test!**

How to use train/val/test labels:

```python
train_indices = np.argwhere(train_labels != -1).squeeze() # 3500-num_outliers
val_indices = np.arange(train_num, train_num + val_num) # 500
test_indices = np.arange(train_num + val_num, 5298) # 1298

# When calculating loss and accuracy (pesudo code):

# predict all, but only calculate loss on training labels
train_loss = loss_func(y_pred[train_indices], y[train_indices])
# predict all, but only calculate accuracy on val/test labels
val_acc = accuracy(y_pred[val_indices], y[val_indices])
test_acc = accuracy(y_pred[test_indices], y[test_indices])
```

### Methodology

This section introduces the input embeddings and the applied models including graph-based and non-graph models.

Notations: $X\in\N^{N\times C}, Y\in\N^{N}, N=5298, C=6, n=11$

First I tried a machine learning method: random forest (`sklearn.ensemble.RandomForestClassifier`). The training accuracy is 98.25%, but the testing accuracy is only 31.90%, indicating severe overfitting. So I turned to neural networks.

#### Embeddings of Categorical Features

Due to the drawbacks of one-hot encoding, trainable embeddings are applied to deal with the categorical input features.

In detail, each feature is assigned with a trainable parameter matrix, and the feature value is used as an index to extract an embedding vector in the matrix. For example, for "high_school", there will be an embedding matrix $E_{hs}\in\R^{2506\times d}$, where $d$ is the embedding dimension. This is implemented using `nn.Embedding` in PyTorch.

As a result, for the six features, there are six embedding matrices $E_{deg}\in\R^{6\times d}$, $E_g\in\R^{3\times d}$, $E_m\in\R^{43\times d}$, $E_{sm}\in\R^{44\times d}$, $E_{dorm}\in\R^{6\times d}$, and $E_{hs}\in\R^{2506\times d}$.

We concat the embedding vectors to generate the input embeddings $E\in\R^{N\times 6d}$:
$$
E=\mathop{\mathrm{\vert\vert}}_{i=1}^C \mathrm{EmbeddingLayer}(X_{:, i}),
$$
where $\vert\vert$ denotes the concatenation operation.

#### Non-graph Model: MLP

[models/MLP.py](https://github.com/XDZhelheim/CSE5002-Project/blob/main/models/MLP.py)

A 2-layer MLP is served as a baseline, which is 
$$
\hat Y=Softmax(W_2(ReLU(W_1E+b_1))+b_2),
$$
where $W_1\in\R^{6d\times h}, b_1\in\R^{h}, W_2\in\R^{h\times n}, b_2\in\R^{n}$, and $h$ stands for hidden dimension. The output is $\hat Y\in\R^{N\times n}$.

**Note:** Softmax function is not needed in the model code when using `nn.CrossEntropyLoss`. This is because this loss function in PyTorch will perform Softmax inside. See the [official doc](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

#### Graph Convolutional Networks

[models/GCN.py](https://github.com/XDZhelheim/CSE5002-Project/blob/main/models/GCN.py)

Two GCN models are applied, namely traditional GCN using graph Laplacian, and Adaptive Diffusion Graph Convolution Network (ADFGCN).

Finally I chose ADFGCN because of these reasons:

- It is a SOTA GCN method in my research field (Spatial-Temporal Graph Learning)
- It is simple yet effective, while having good interpretability
- I am very familiar with its code implementation...

The detailed deduction of the GCNs will be introduced in the following paragraphs.



GCN proposed by Kipf et al. [1] is defined as:
$$
Z=\tilde LXW,
$$
where $\tilde L$ is the normalized graph Laplacian, $X$ is the input feature matrix and $W$ is the parameter matrix.

Li et al. [2] proposed a diffusion convolution layer which proves to be effective. They modeled the diffusion process of graph signals with $K$ finite steps. The diffusion convolution is defined as 
$$
Z=\sum_{k=0}^K P^kXW_k,
$$
where $P^k$ represents the power series of the transition matrix. In the case of an undirected graph, $P=A/rowsum(A)$. In the case of a directed graph, the diffusion process have two directions, the forward and backward directions, where the forward transition matrix $P_f=A/rowsum(A)$ and the backward transition matrix $P_b=A^T/rowsum(A^T)$. With the forward and the backward transition matrix, the diffusion graph convolution is
$$
 Z=\sum_{k=0}^K P_f^kXW_{k1}+P_b^kXW_{k2}.
$$
Based on diffusion convolution, Wu et al. [3] further proposed adaptive diffusion convolution. It uses a self-adaptive adjacency matrix $A_{adp}\in\R^{N\times N}$ which does not require any prior knowledge and is learned end-to-end through stochastic gradient descent. By doing so, the model can discover hidden spatial dependencies by itself. It is achieved by randomly initializing two node embeddings with learnable parameters $E_1, E_2\in\R^{N\times D}$. The self-adaptive adjacency matrix is calculated as
$$
 A_{adp}=Softmax(ReLU(E_1E_2^T)).
$$
The source node embedding is referred to as $E_1$, while the target node embedding is called $E_2$. To obtain the spatial dependency weights between the source nodes and the target nodes, it computes the product of $E_1$ and $E_2$. Weak connections are eliminated using the ReLU activation function, and the Softmax function is utilized to normalize the self-adaptive adjacency matrix. The resulting normalized self-adaptive adjacency matrix can be interpreted as the transition matrix of a hidden diffusion process. Therefore, combining diffusion convolution and self-adaptive adjacency matrix, adaptive diffusion convolution is defined as
$$
Z=\sum_{k=0}^K P_f^kXW_{k1}+P_b^kXW_{k2}+A_{adp}^kXW_{k3}.
$$
In addition, residual and skip connection is applied when using $L$ GCN layers. To sum up, the forwarding procedure of ADFGCN is:
$$
\begin{aligned}
	E&=\mathop{\mathrm{\vert\vert}}_{i=1}^C \mathrm{EmbeddingLayer}(X_{:, i})\\
	Z^{(0)}&=WE+b\\
    Z_{conv}^{(l)}&=\sum_{k=0}^K P_f^kZ^{(l-1)}W_{k1}+P_b^kZ^{(l-1)}W_{k2}+A_{adp}^kZ^{(l-1)}W_{k3}\\
    Z^{(l)}&=Z_{conv}^{(l)}+Z^{(l-1)},\ \mathrm{i.e.\ residual}\\
    Z_{all}&=\mathop{\mathrm{\vert\vert}}_{l=1}^L Z_{conv}^{(l)},\ \mathrm{i.e.\ skip\ connection} \\
    \hat Y&=Softmax(W_2(ReLU(W_1Z_{all}+b_1))+b_2),
\end{aligned}
$$
where $l=1, 2, \cdots, L$. All the $W$ and $b$ are parameters.

The tensor shape of each step is:

|          Tensor           | Shape ($N=5298$)                       |
| :-----------------------: | :------------------------------------- |
|            $E$            | $(N, C\cdot d)$ where $C=6$            |
| $Z_{conv}^{(l)}, Z^{(l)}$ | $(N, h)$ where $h$ is hidden dimension |
|         $Z_{all}$         | $(N, L\cdot h)$ where $L$ is #layers   |
|         $\hat Y$          | $(N, n)$ where $n=11$                  |

Also, Softmax is not needed in the code, reasons are explained before.

### Experiments

This section gives comparisons on different models and one-hot vs. trainable embeddings.

#### Settings

[configs/ADFGCN.yaml](https://github.com/XDZhelheim/CSE5002-Project/blob/main/configs/ADFGCN.yaml)

The experiments were performed on a server equipped with an Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz and an NVIDIA GeForce RTX 2080Ti graphics card. The PyTorch version is 1.11 with Python 3.8.

For ADFGCN, I adopted $L=2$ layers adaptive diffusion convolution whose order is $K=1$. The hidden size of GCN $h=32$, the input embedding dimension $d=8$, node embedding dimension $D=16$, and learning rate was 0.01 and would be decreased to 0.001 after 10 epochs. To avoid overfitting, dropout ratio was set to 0.1, l2 regularization coefficient $\lambda=0.001$, and the max norm in gradient clipping was set to 5. Adam algorithm was employed to control the overall training process, and the loss function was *Cross Entropy*. The training process would be early-stopped if the validation loss was not decreasing for 50 epochs, then the best model on validation data would be saved.

#### Performance Analysis

[logs/](https://github.com/XDZhelheim/CSE5002-Project/tree/main/logs)

Table of test accuracy:

|                       |  MLP   | Laplacian GCN |   ADFGCN   |
| :-------------------: | :----: | :-----------: | :--------: |
|   One-hot Encoding    | 30.51% |    13.79%     |   62.79%   |
|  Trainable Embedding  | 33.28% |    68.57%     | **70.80%** |
| Random Forest: 31.90% |        |               |            |

Based on the table above, we can draw some conclusions:

- Comparing ML model Random Forest and DL model MLP, MLP did not give a performance improvement. This indicates that the models without considering the graph structure are  equally bad.
- Comparing MLP and GCNs, it is obvious that GCNs have much higher results. This strongly proves that **using both two sources of  information is better than using a single source of information**.
- Comparing one-hot encoding and trainable embedding, the latter gives much better performance, especially for Laplacian GCN. This shows its effectiveness.
- Comparing Laplacian GCN and ADFGCN, we can see that ADFGCN is very robust because it can even give a considerable accuracy when using one-hot. Also, it gives the best accuracy 70.8% when using trainable embeddings.

### Conclusion

In this project, I first analyzed the raw data and performed data cleaning. Based on the cleaned data, the training/validation/testing dataset were created. To deal with categorical input features, I tried one-hot encoding and trainable embedding. Models including Random Forest, MLP and two GCNs are tested and compared.

However, there are still many aspects that can be improved. One of them is **label imbalance**. I did not address this issue. This might be solved by manually modifying the CrossEntropyLoss, by giving a weight to each class.

### References

[1] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” arXiv preprint arXiv:1609.02907, 2016

[2] Y. Li, R. Yu, C. Shahabi, and Y. Liu, “Diffusion convolutional recurrent neural network: Data driven traffic forecasting,” in International Conference on Learning Representations, 2018.

[3] Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang, “Graph wavenet for deep spatial-temporal graph modeling,” arXiv preprint arXiv:1906.00121, 2019.
