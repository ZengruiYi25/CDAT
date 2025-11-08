# CDAT: Efficient Convolutional Dual-Attention Transformer for Automatic Modulation Recognition
The repaired version of our paper's code. 
[Efficient Convolutional Dual-Attention Transformer for Automatic Modulation Recognition](https://link.springer.com/article/10.1007/s10489-024-06202-6)

## Abstract
Automatic modulation recognition (AMR) involves identifying the modulation of electromagnetic signals in a noncollaborative manner. Deep learning-based methods have become a focused research topic in the AMR field. Such models are frequently trained using standardized data, relying on many computational and storage resources. However, in real-world applications, the finite resources of edge devices limit the deployment of large-scale models. In addition, traditional networks cannot handle real-world
signals of varying lengths and local missing data. Thus, we propose a network structure based on a convolutional Transformer with a dual-attention mechanism. This proposed structure effectively utilizes the inductive bias of the lightweight convolution and the global property of the Transformer model, thereby fusing local features with global features to get high recognition accuracy. Moreover, the model can adapt to the length of the input signals while maintaining strong robustness against incomplete signals. Experimental results on the open-source datasets RML2016.10a, RML2016.10b, and RML2018.01a demonstrate that the proposed network structure can achieve 95.05%, 94.79%, and 98.14% accuracy, respectively, with enhancement training and maintain greater than 90% accuracy when the signals are incomplete. In addition, the proposed network structure has fewer parameters and lower computational cost than benchmark methods.

## Instructions
This is not a complete project file. It only provides referable code and file paths.

Place the processed data in the ***Datasets*** directory, including train_data, train_label, test_data, and test_label. Specifically, the data are in the format $N \times 2 \times L$, where $N$ and $L$ denote the number of signals and their lengths, respectively. The label is in the format $N\times 2$, with the first column containing the class label and the second column containing the SNR. 

The ***result.txt*** presents the accuracy under each SNR in the original paper.

## Train
Once you have configured the environment and dataset, you can directly run ***Main_CDAT.py***.

The information of the training process and the model parameters will be saved in ***./Argus***.

## Notice
!!!This is not the original code, but a slightly modified version (the performance should be close). Through our verification, under different dataset partitions, the Overall Accuracy (OA) obtained by this model is higher than that reported in the paper, but the Highest Accuracy (HA) may decrease, which is possible.

!!!The latest version will be continuously updated in the near future.

If this project helps you, please give it a star.

If you are interested in our work, please cite:

````bibtex
@article{yi2025efficient,
  title={Efficient convolutional dual-attention transformer for automatic modulation recognition},
  author={Yi, Zengrui and Meng, Hua and Gao, Lu and He, Zhonghang and Yang, Meng},
  journal={Applied Intelligence},
  volume={55},
  number={3},
  pages={231},
  year={2025},
  publisher={Springer}
}
 ```
