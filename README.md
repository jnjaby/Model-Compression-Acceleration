# Model-Compression-Acceleration

# Papers

## Quantization
- Product Quantization for Nearest Neighbor Search,PAMI,2011 [[paper]](https://hal.inria.fr/inria-00514462v2/document)
  * 介绍Product Quantization, 可以关注background部分
- Compressing Deep Convolutional Networks using Vector Quantization,ICLR,2015 [[paper]](https://arxiv.org/pdf/1412.6115.pdf)
  * 关于Vector Quantization早期比较有影响力的工作，用k-means学习centroids
- Deep Learning with Limited Numerical Precision, ICML, 2015 [[paper]](https://pdfs.semanticscholar.org/dec1/59bb0d83a506ec61fb8745388e585f48be44.pdf?_ga=2.16188209.660876135.1502713025-632431917.1498533020)
  * 在MNIST，CIFAR上进行16-bit fixed-point实验
- Ristretto: Hardware-Oriented Approximation of Convolutional Neural Networks, ArXiv, 2016 [[paper]](https://arxiv.org/pdf/1604.03168.pdf)
- Fixed Point Quantization of Deep Convolutional Networks, ICML, 2016 [[paper]](https://pdfs.semanticscholar.org/d88d/3d8450f2032b3a59d0006693381877bfc1da.pdf?_ga=2.82169745.660876135.1502713025-632431917.1498533020)
  * 推导quantization error在网络中的传播，根据这个error选取layer-wise的bit-width
- Quantized Convolutional Neural Networks for Mobile Devices, CVPR, 2016 [[paper]](https://pdfs.semanticscholar.org/2353/28f8bc8b62e04918f9b4f6afe3c64cfdb63d.pdf?_ga=2.115328896.660876135.1502713025-632431917.1498533020)
  * Vector Quantization的一种，sub-vector和codebook近似
- Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights, ICLR, 2017 [[paper]](https://arxiv.org/pdf/1702.03044.pdf)
  * weights限制为0或2的幂，采用循环量化和逐步代偿的idea
- Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding, ICLR, 2016 [[paper]](https://arxiv.org/pdf/1510.00149.pdf)
  * ICLR'16 best paper，在model compression里较重要的工作，结合quantization和下面提到的pruning，能把AlexNet压缩30多倍
- BinaryConnect: Training Deep Neural Networks with binary weights during propagations, NIPS, 2015 [[paper]](https://pdfs.semanticscholar.org/a573/3ff08daff727af834345b9cfff1d0aa109ec.pdf?_ga=2.32656573.200323026.1503209786-632431917.1498533020)
- BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1, ArXiV, 2016 [[paper]](https://arxiv.org/pdf/1602.02830.pdf)
- XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks, ECCV, 2016 [[paper]](https://pdfs.semanticscholar.org/9e56/cc1142e71fad78d1423791f99a5d2d2e61d7.pdf?_ga=2.259605641.200323026.1503209786-632431917.1498533020)
  * Quantization中最极端的一种，只用1 bit表示数字
  * BinaryConnect中只有weights是二值化的，BNN中weights和activations都是二值化的，这两个实验都在小数据集上进行
  * XNOR-Net思路上跟BNN一致，但对layer加了scale补偿信息损失，在ImageNet上进行实验performance有10个点的损失。
- Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations, ArXiv, 2016 [[paper]](https://arxiv.org/pdf/1609.07061.pdf)
- DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients, ArXiv, 2016 [[paper]](https://arxiv.org/pdf/1606.06160.pdf)
  * 作为XNOR-Net上改进的工作，采用不同bit-width的weights和activations补偿信息损失，在ImageNet上1-bit weights和4-bit activations要比XNOR-Net好很多
  * 二值化网络的改进还有一些工作，如三值化等，此处不再一一列举

## Pruning
- Optimal Brain Damage, NIPS, 1990 [[paper]](https://pdfs.semanticscholar.org/17c0/a7de3c17d31f79589d245852b57d083d386e.pdf?_ga=2.267651469.200323026.1503209786-632431917.1498533020)
- Learning both Weights and Connections for Efficient Neural Network, NIPS, 2015 [[paper]](http://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf)
- Pruning Filters for Efficient ConvNets, ICLR, 2017 [[paper]](https://arxiv.org/pdf/1608.08710.pdf)
- Sparsifying Neural Network Connections for Face Recognition, CVPR, 2016 [[paper]](https://pdfs.semanticscholar.org/d8e6/9677fe51836847f63e5ef84c8d3d68942d12.pdf?_ga=2.259031433.200323026.1503209786-632431917.1498533020)
- Learning Structured Sparsity in Deep Neural Networks, NIPS, 2016 [[paper]](https://pdfs.semanticscholar.org/35cd/36289610df4f221c309c4420036771fcb274.pdf?_ga=2.34365986.200323026.1503209786-632431917.1498533020)
## Knowledge Distallation
- Distilling the Knowledge in a Neural Network, ArXiv, 2015 [[paper]](https://arxiv.org/pdf/1503.02531.pdf)
- FitNets: Hints for Thin Deep Nets, ICLR, 2015 [[paper]](https://arxiv.org/pdf/1412.6550.pdf)
- Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer, ICLR, 2017 [[paper]](https://arxiv.org/pdf/1612.03928.pdf)
- Face Model Compression by Distilling Knowledge from Neurons, AAAI, 2016 [[paper]](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/11977/12130)
- In Teacher We Trust: Learning Compressed Models for Pedestrian Detection, ArXiv, 2016 [[paper]](https://arxiv.org/pdf/1612.00478.pdf)
- Like What You Like: Knowledge Distill via Neuron Selectivity Transfer, ArXiv, 2017 [[paper]](https://arxiv.org/pdf/1707.01219.pdf)

## Network Architecture
- SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5MB model size, ArXiv, 2016 [[paper]](https://arxiv.org/pdf/1602.07360.pdf)
- Convolutional Neural Networks at Constrained Time Cost, CVPR, 2015 [[paper]](https://pdfs.semanticscholar.org/9a1b/08883a74b25f35f1df9553718899e2bdb944.pdf?_ga=2.268584178.200323026.1503209786-632431917.1498533020)
- Flattened Convolutional Neural Networks for Feedforward Acceleration, ArXiv, 2014 [[paper]](https://arxiv.org/pdf/1412.5474.pdf)
- Design of Efficient Convolutional Layers using Single Intra-channel Convolution, Topological Subdivisioning and Spatial "Bottleneck" Structure, ArXiv, 2016 [[paper]](https://arxiv.org/pdf/1608.04337.pdf)
- Xception: Deep Learning with Depthwise Separable Convolutions, ArXiv, 2017 [[paper]](https://arxiv.org/pdf/1610.02357.pdf)
- MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, ArXiv, 2017 [[paper]](https://arxiv.org/pdf/1704.04861.pdf)
- ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices, ArXiv, 2017 [[paper]](https://arxiv.org/pdf/1707.01083.pdf)

## Matrix Factorization(Low-rank Approximation)
严格来说Matrix Factorization在形式上应当属于Network Architecture的一种，但两条line出发点稍有不同，部分文章也很难严格区分属于哪一类，姑且如此列出
- Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation, NIPS,2014 [[paper]](https://pdfs.semanticscholar.org/e5ae/8ab688051931b4814f6d32b18391f8d1fa8d.pdf?_ga=2.149358257.660876135.1502713025-632431917.1498533020)
- Speeding up Convolutional Neural Networks with Low Rank Expansions, BMVC, 2014 [[paper]](https://pdfs.semanticscholar.org/d1a8/f0d257d434add438867ffeca4f2a4b40e5ae.pdf?_ga=2.10872371.660876135.1502713025-632431917.1498533020)
- Deep Fried Convnets, ICCV, 2015 [[paper]](https://pdfs.semanticscholar.org/27a9/9c21a1324f087b2f144adc119f04137dfd87.pdf?_ga=2.269034738.200323026.1503209786-632431917.1498533020)
- Accelerating Very Deep Convolutional Networks for Classification and Detection, TPAMI, 2016 [[paper]](https://pdfs.semanticscholar.org/3259/b108d516f4700411f92e574a0f944462f0bc.pdf?_ga=2.215762068.200323026.1503209786-632431917.1498533020)
- Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition, ICLR, 2015 [[paper]](https://arxiv.org/pdf/1412.6553.pdf)
