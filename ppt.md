# SAND: Spatially Adaptive Network Depth for Fast Sampling of Neural Implicit Surfaces

## Introduction: Implicit neural representations

我们要解决的核心问题是如何快速准确的渲染一个模型.

传统的方法通常通过把模型存储为很多小三角面的并集,然后通过光线追踪或把面投影到屏幕上渲染.

隐式神经网络表达技术通过其他方法:设模型在空间中占据的点的集合为$V$(严谨的话,可以是闭集),则可以定义一个标量场$f(\vec x)=\mathrm{sgn}(x)\mathrm{dis}(x,\partial V)$为点$x$到模型表面的有符号距离(一般里面是负的外面是正的),则$f(\vec x)=0$确定了模型表面.

而隐式神经网络表示就是监督训练一个网络来拟合$f$,来把传统的模型转化为训练好的权重数据.

渲染时常使用ray marching方法:光线投射的隐式神经网络版.

优势:无损表达光滑表面,利用泛化能力减小存储消耗

## Results

### Suzanne 猴头

6.2 万面。

| 网络 | RM 总时长 (s) | RM 网络 (s) | RM 零网络查询占比 | RM 平均深度 | RM 时长比 |
|---|---|---|---|---|---|
| baseline (无纹理) | 2.86 | 2.47 | 0.000 | 8.00 | — |
| SAND (无纹理) | 10.17 | 4.71 | 0.868 | 3.69 | 3.56x |
| SAND+纹理扩展 | 10.81 | 5.06 | 0.863 | 4.97 | 3.78x |

| 网络 | Ray Marching | RM 深度图 |
|---|---|---|
| baseline (无纹理) | ![rm_base_geo](report/suzanne/rm_base_geo.png) | — |
| SAND (无纹理) | ![rm_sand_geo](report/suzanne/rm_sand_geo.png) | ![rm_depth_sand_geo](report/suzanne/rm_depth_sand_geo.png) |
| SAND+纹理扩展 | ![rm_sand_color](report/suzanne/rm_sand_color.png) | ![rm_depth_sand_color](report/suzanne/rm_depth_sand_color.png) |

### R2 机器人

6.6 万面、13 个非封闭部件、2048² 贴图。实心化 v2 预处理。

| 网络 | RM 总时长 (s) | RM 网络 (s) | RM 零网络查询占比 | RM 平均深度 | RM 时长比 |
|---|---|---|---|---|---|
| baseline (无纹理) | 3.80 | 3.48 | 0.000 | 8.00 | — |
| SAND (无纹理) | 9.82 | 4.42 | 0.916 | 3.95 | 2.58x |
| SAND+纹理扩展 | 9.36 | 4.41 | 0.884 | 6.32 | 2.46x |

| 网络 | Ray Marching | RM 深度图 |
|---|---|---|
| baseline (无纹理) | ![rm_base_geo](report/r2/rm_base_geo.png) | — |
| SAND (无纹理) | ![rm_sand_geo](report/r2/rm_sand_geo.png) | ![rm_depth_sand_geo](report/r2/rm_depth_sand_geo.png) |
| SAND+纹理扩展 | ![rm_sand_color](report/r2/rm_sand_color.png) | ![rm_depth_sand_color](report/r2/rm_depth_sand_color.png) |


| 模型 | 网络 | 参数量 | 迭代数 | 训练时长 (s) | ms/iter | 最终 loss |
|---|---|---|---|---|---|---|
| Suzanne 猴头 | baseline (无纹理) | 461825 | 10000 | 1603.86 | 160.39 | 0.006387 |
| Suzanne 猴头 | SAND (无纹理) | 465423 | 10000 | 2829.50 | 282.95 | 0.008446 |
| Suzanne 猴头 | SAND+纹理扩展 | 476988 | 10000 | 2268.58 | 226.86 | 0.492926 |
| R2 机器人 | baseline (无纹理) | 461825 | 10000 | 1600.17 | 160.02 | 0.064338 |
| R2 机器人 | SAND (无纹理) | 465423 | 10000 | 2081.81 | 208.18 | 0.008331 |
| R2 机器人 | SAND+纹理扩展 | 476988 | 10000 | 2261.68 | 226.17 | 0.349134 |


### Explanation

baseline即直接同规模网络拟合距离场

为什么我们的比baseline慢?
1. 最重要的:baseline在震r荡同规模缺少octtree的策略,需要拟合整个空间,结果压根没收敛,导致平均marching次数远小于收敛了的sand
2. baseline的batch开的比较大

## Technique

论文使用的技术,和我们使用的技术


### Octtree

在空间中建立八叉树,每个节点代表空间中的一个立方体,其八个儿子是把自己的空间分成八个字立方体.

建树时,当且仅当模型表面与当前立方体有交时进行细分,可以快速过滤掉模型内部/外部的空间,精细化训练模型表面附近的空间.使得拟合更为精细.

Octtree被用来
- 生成神经网络的训练集:允许我们在靠近模型表面的空间采样,聚焦表面细节训练,不用拟合通过Octree过滤掉的空间远处的情况.
- 决定网络推理深度,见后.

### New Network Structure

传统的MLP拟合模型必须跑完整个深度,即使某处模型的表达很简单(比如,可线性表达的平面)

论文将MLP模型改造为T-MLP,MLP可以用如下公式表达:

$$
\begin{gathered}
\vec x_k=\operatorname{F}(W_k\vec x_{k-1}+\vec b_k) \\
L=\|\vec x_n-\vec x^\star\|
\end{gathered}
$$

论文改造为:

$$
\begin{aligned}
\vec x_k&=\operatorname{F}(W_k\vec x_{k-1}+\vec b_k) \\
\vec y_1&=O \vec x_1+\vec b_o \\
\forall k>1,\vec y_k&=(O_{k,1}\vec x+\vec b_{k,1})*(O_{k,2}\vec x+\vec b_{k,2})\\
L_k&=\|\vec y_n-\vec y^\star\| \\
L&=\sum_k L_k
\end{aligned}
$$

显然这是改为后面的层拟合残差,这样越往后越是精确逼近.

注意激活函数$F(x)$一般取$\sin \omega x$第三行用的是对应项相乘.论文说是为了解决残差过小,且实验这样更优.

也许还有浅层神经网络不善于拟合点积的原因.

网络训练好,提前对每个Octtree叶子节点存储需要递推的层数:在节点内取样,取对于给定误差,每个点所需推理层数的最大值,作为以后该节点内推理深度.

### Texture

原论文是不带颜色的,我们给标准的距离场基础上增加了rgb通道的拟合,一起训练,使得支持渲染有纹理的模型.

## Bugs

### Bad Model

一开始选了个神秘奇异模型.注意到模型常常不是单个联通集.我们使用高斯环绕数判断点在内部还是外部,但仍然遇到了一些错综复杂的面使得内外混到一起了,导致渲染有穿模现象.

解决方案:换一个好模型.

![bad_model](report/robot_union2/rm_sand_color.png)

### Wrong Direction of Normal Vector

我们发现渲染结果有神秘噪点:当使用ray marching算法时,我们是通过有限差分求距离场的梯度算的表面法向量.而梯度方向可能朝外或朝内.朝内的表面无光照到表现为黑噪点.

解决方案:注意到能看到的表面的法向量一定与相机同侧.

![black_points](report/other/black_points.png)
![black_points](report/other/black_points_fixed.png)

## Play time

![Play](http://124.222.61.175:16200/)

## Thanks


