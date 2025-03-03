# extract

# Rules

1. important不可乱用 + such as
2. key information + 解释
3. 状语前置
4. 一句话三行
5. Equation → 方法名
6. Equation (1)
7. 方法部分只需要详细步骤不需要总结型话术
8. 复数不加the
9. 时态保持一致
10. GNN GCN 不加s
11. Close intrinsic relationship
12. 正文与参考文献之间需要＋空格

# 专有名词

1. Drug–target interaction (DTI)
2. computational/in silico approaches
3.  生物实验：biochemical experimental methods
4. 体外验证：vitro validation
5. 潜在的药物靶标作用对：potential DTI pairs/candidates
6. drug discovery
7. 分子对接：molecule docking ***/*** docking-based approaches
8. 基于配体的：ligand-based approaches
9. 验证实验：
10. convolutional neural networks (CNNs)
11. 氨基酸序列：amino acid subsequences
12. calculate probablity
13. predict interaction
14. Graph Transformer （大写）
15. high-cost
16. make great progress (单数

# 替换词

1. 有效的：effective; efficient; valid; resultful;
2. 快速的：fast; clipping; frequent; sudden; fleet;
3. 传统的：traditional；conventional
4. 应用：combine

# 摘要

# 背景（intro）

1. 蛋白质-配体相互作用指的是蛋白质与小分子化合物之间的相互作用。化合物会与蛋白特定的结合位点发生相互作用，从而形成稳定的蛋白质-配体复合物。
2. 蛋白质的功能通常依赖于其三级结构，包括酶活性和配体（如小分子抑制剂）的结合。为了预测配体结合（位点、姿态和亲和力），首先需要确定蛋白质的三维结构。
3. in silico method (v1)
    1. biochemical experimental methods：extremely costly and time-consuming
    2. in silico or computational approaches：efficiently identify potential DTI candidates for guiding in vivo validation
    3. Traditional computational methods
        1. molecular docking-based approaches：
            1. means：基于锁钥原理 通过分子间空间**形态互补**和**能量匹配 预测ligand 与 target 是否interact**
            2. limited  when the 3D structure of the protein lacks
        2. ligand-based approaches：
            1. 基于配体的靶标预测，是指将需要预测靶标的化合物与具有已知靶标的化合物的结构特征进行比较分析，从而根据相似性原理预测潜在靶标。
            2. limited when a target has only a small number of known binding ligands
4. in silico method (v2)
    1. Currently, the ligand-based, docking simulation, and chemogenomic approaches are the three main classes of computational methods for predicting DTIs.

## results

### FragXsiteDTI

1. The computational results on two datasets demonstrate the predictive power of our FragXsiteDTI compared to several state-of-the-art models and across multiple evaluation metrics. 
2. Also, our model is **the first and the only one** providing an information-rich interpretation of the interaction in terms of the critical parts of the target protein and drug molecule in a drug-target pair. 

# Datasets

- 数据来源（数量已统计
    - biosnap：DrugBAN
    - human、drugbank：FragxsiteDTI  code

# 方法

## Framework

### DrugBAN

- The proposed DrugBAN framework is shown in Figure 1a.

# Representation

## drug

### SMILES-based

1. [SMILES & InChI | 化学结构的线性表示法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/86709568)
2. **Limitations of SMILES-based molecular generators**
- Yet SMILES do present challenges; namely, they do not detail the three-dimensional structure of a molecule beyond atomic connections, and there are several ways to represent the same
molecule.
- The lack of structural information and inherent redundancy in SMILES can cause SMILES-based models to struggle to fully understand the chemical and structural relationships between molecules.

### Graph-based

1. more in line with molecular structure
2. **molecule rep(**
    1. The idea behind the molecular graph representation lies in mapping the atoms and bonds that make up  a molecule into sets of nodes and edges.

### Motif

1. Fragment-based Drug Discovey (FBDD)
- FBDD方法的核心思想是，通过筛选和优化小的化学片段，这些片段能够与目标蛋白的活性位点结合并展示某种程度的生物活性。由于片段比传统的高通量筛选（HTS）中的化合物**更小且结构更简单**，FBDD可以更高效地探索化学空间，发现新的结合位点和新颖的化学结构。
1. **motif-based MRL**
- motif = frequent fragmentsn with some domain-specific structures or patterns
- functional groups (frequently-occurred subgraphs in molecular graphs) often carry indicative
information about the molecular properties.

### One-hot

- DrugBAN
    - **Each atom is represented as** a 74-dimensional integer vector **describing** eight pieces of information: the atom type, the atom degree, the number of implicit Hs, the formal charge, the number of radical electrons, the atom hybridization, the number of total Hs and whether the atom is aromatic.

### Substructure

1. **MRL(**
    1. Treating molecular graphs as regular attributed graphs would overlook the special substructure patterns of molecules, **such as motifs and functional groups**.

## protein

[(77 封私信 / 80 条消息) 蛋白质的一、二、三和四级结构分别是什么？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/641866834/answer/3427282225)

### 一级结构

- 蛋白质的一级结构是后面所有高级结构的基础

### 三级结构

- 整个蛋白质分子在空间中的折叠形态。这种结构是由氨基酸侧链之间的各种相互作用所决定的，包括氢键、疏水作用、离子键和范德华力等。**三级结构的确定对蛋白质的功能至关重要**。

### Pockets（3D）

1. motivation
    - **必要性：**contribute most to the interaction between protein and drug
        - 虽然现有方法在DTI预测中蛋白表征主要依赖蛋白质序列，但决定相互作用的关键是蛋白质结构。蛋白质3D结构中的口袋信息对于药物靶标识别、激动剂设计、虚拟筛选和受体-配体结合分析至关重要，是配体与蛋白质发生相互作用的关键位点。
        - 蛋白质口袋是蛋白-配体相互作用的先决条件，先前的蛋白质表征主要使用蛋白质序列，但在空间结构中相邻的残基在一维序列中很可能是不相邻的。因此，直接使用pockets结构作为蛋白质的表征可以更有有效的表示出蛋白与配体相互作用的信息，同时可以一定程度上缓解蛋白表征的信息冗余问题。
    - **充分性：**研究表明，绝大部分蛋白都是通过有结构化亚袋的口袋中结合相互作用的配体。
    - **new：**研究表明，约一半全蛋白可以在一个包含有结构化亚袋的口袋中**同时结合多个相互作用的配体**。
2. **simple method：**
- **表面特征的表示**：蛋白质表面的凸包边缘是由蛋白质原子位置的几何特征决定的。这些边缘三角形反映了蛋白质表面的凹凸不平，而活性位点或口袋通常位于这些凹凸区域。因此，凸包边缘的三角形可以作为识别潜在口袋位置的起点。
- **局部凹陷的指示**：在凸包的边缘，三角形的内部区域可能代表蛋白质表面的局部凹陷，这些凹陷区域很可能是配体结合的口袋。通过分析这些三角形，可以初步确定口袋的位置。
- **简化的口袋识别**：凸包边缘的三角形提供了一种简化的方式来识别和描述蛋白质表面的复杂结构。这种方法不需要深入到原子层面的细节，而是通过宏观的几何形状来初步界定口袋。
1. Protein Binding Pocket Dynamics
- protein binding pockets are crucial for their interaction specifificity
- A cavity on the surface or in the interior of a protein that possesses
suitable properties for binding a ligand is usually referred to as a
binding pocket.

# Graph Transformer

[Graph Transformer-CSDN博客](https://blog.csdn.net/weixin_41922868/article/details/129429447)

1. allows the usage of explicit domain information as edge features
    1. 以edge attribute的形式提供的丰富特征信息
    2. 强调了化学键的重要性
2. fuse node positional features using Laplacian eigenvectors

## Laplacian Matrix: 利用特征值和特征向量来识别图的结构和模式

### 推导

[GNN入门之路: 01.图拉普拉斯矩阵的定义、推导、性质、应用 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/368878987)

[GNN入门之路：02.谱域方法：Spectral CNN，ChebyNet，GCN - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/369382428)

[理解图的拉普拉斯矩阵 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/362416124)

1. 拉普拉斯算子的定义

![Untitled](Untitled.png)

![Untitled](Untitled%201.png)

![Untitled](Untitled%202.png)

1. 图·拉普拉斯算子的含义：节点与邻居节点之间信号的差异

![Untitled](Untitled%203.png)

![Untitled](Untitled%204.png)

![Untitled](Untitled%205.png)

1. 符号定义

![Untitled](Untitled%206.png)

1. 拉普拉斯矩阵特征分解：
- 归一化→用特征值和特征向量表示

![Untitled](Untitled%207.png)

- 从图傅里叶变换角度：特征值&特征向量→描述波动性

![Untitled](Untitled%208.png)

![Untitled](Untitled%209.png)

### 拉普拉斯矩阵的解读

- [GCN 笔记：图上的傅里叶变换_拉普拉斯归一化-CSDN博客](https://blog.csdn.net/qq_40206371/article/details/118230105)
    - **拉普拉斯矩阵反映了当前节点对周围节点产生扰动时所产生的累积增益。**直观上也可以理解为某一节点的权值变为其相邻节点权值的期望影响，形象一点就是**拉普拉斯矩阵可以刻画局部的平滑度。**
    - 本质上，也是一种**message passing**
- **矩阵分解**：
- **特征值**
    - 反映了**特征向量**的平滑度，值越小代表对应的**特征向量变化越平缓，取值差异不明显**
    - 特征值都大于等于零，归一化的拉普拉斯矩阵的特征值区间为 [0, 2]
    - **零特征值**：特征值中0出现的次数就是图连通区域的个数
        - 每个连通分量对应一个独立的零特征值
        - 其特征向量被称为平凡特征向量
    - **最小非零特征值**：最小非零特征值是图的代数连通度
        - 代数连通度（Algebraic Connectivity），也称为Fiedler值，是拉普拉斯矩阵的**第二小特征值**。它是衡量图连通性的一种重要指标，反映了图的全局连通性和紧密性
            - **代数连通度越大**：图越紧密，节点之间的连接越强
            - **代数连通度越小**：图越松散，存在较弱的连接，甚至可能接近分裂成多个部分
    - **k smallest non-trvial：用于谱聚类**
        - [图神经网络和图表征学习笔记（三）拉普拉斯矩阵与谱方法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/166669588)
        - 较小的特征值和对应的特征向量用于捕捉**全局结构信息**，从而有效地进行降维和嵌入
- LPE

# 双线性注意力网络（BAN）

# 蛋白质表征

### 自监督预训练

reference: 

[Structure based Multiview-CL（ICLR-2023）](https://www.notion.so/Structure-based-Multiview-CL-ICLR-2023-d2177325408341838acc7044a448d5f2?pvs=21)

- 基于序列：
    - 整个空间（PLM）：捕捉底层大规模蛋白质序列语料库中的生物化学和共同进化知识→微调用于下游任务
    - MSA：利用蛋白质家族内的序列来捕捉同源序列的保守和可变区域，这暗示了蛋白质家族的特定结构和功能
- 基于结构：构建残基级/原子级/蛋白质表面图→以自监督的方式学习图表示→从未标记的图中获取知识

## 分子碎片

**motivation**

- 用于高效地将大型有机分子分割成多样化的高质量片段，这对于药物发现范式中的高质量片段库构建至关重要。

method

- 生成所有可能的块：首先识别所有可切割键(B)，将分子切割成最小构建块，并使用用户可定义的参数maxSR来决定环结构的完整性。
    - BRICS规则：分子片段化，具有生物活性的小分子，生成49个待切割的键
- 通过将分子简化为无向图，使用Simple算法枚举所有符合指定最大节点数的连通子图，然后将子图映射回原始分子以提取片段。
    - 最大节点数：控制子图大小
    - 连通子图：保证符合原结构的前提下组合最小构建块（乱组合会导致原结构改变）

## GCN

[【图-注意力笔记，篇章1】一份PPT带你快速了解Graph Transformer：包括Graph Transformer 的简要回顾且其与GNN、Transformer的关联 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/611215525)

- 初始表征h0：one-hot编码属性信息
- 更新：
    - 简单汇聚
        
        $$
        H_i^{(l+1)}=σ(∑\nolimits_{j∈neighbors(i)}H_j^{(l)}W^{(l)})
        $$
        
    - 一般形式
    
    $$
    H^{(l+1)} = \sigma\left(\hat{D}^{-\frac{1}{2}} \hat{A} \hat{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)
    $$
    

# Metrics

## FragXsiteDTI

*Evaluation metrics*. We conducted evaluations of our models using various metrics, such as the Area

Under the Receiver Operating Characteristic Curve (AUC), precision, recall, and the F1 score.

## Recall-假阴性

在药物靶标预测问题中，假阴性（False Negative, FN）的含义如下：

### 含义

- 假阴性指的是实际为有效药物靶标的样本，被模型错误地预测为无效药物靶标的情况。

### 举例说明

- 设想一个药物靶标预测系统，目标是预测某个蛋白质是否是一个有效的药物靶标。
- 如果某个蛋白质确实是一个有效的药物靶标（即在实际情况中它可以与药物结合并产生治疗效果），但是模型预测它为无效（即预测结果为该蛋白质不是药物靶标），这种情况就是假阴性。

### 影响

- **错失潜在的治疗靶标**：假阴性会导致潜在的有效药物靶标被忽视，可能会错失开发新药的机会。
- **延长研发周期**：由于有效靶标没有被正确识别，可能需要更长时间才能找到合适的靶标，从而延长药物研发的整体周期。
- **增加研发成本**：为了补偿假阴性带来的损失，可能需要进行更多的实验和筛选步骤，从而增加研发成本。

### 重要性

在药物靶标预测问题中，减少假阴性非常重要，因为：

- 确保尽可能多地识别出所有潜在有效的药物靶标，以提高药物发现的成功率。
- 降低错过重要治疗靶标的风险，尤其是在涉及重大疾病或紧急医疗需求的情况下。

### 平衡 Precision 和 Recall

为了在药物靶标预测中减少假阴性，通常需要提高Recall（召回率），即在所有实际有效的药物靶标中，尽可能多地正确识别出来。然而，这通常会牺牲一些Precision（精确率），即预测为有效的靶标中有部分可能是错误的（假阳性）。在实践中，需要在Precision和Recall之间找到一个合理的平衡，以最大化预测模型的整体性能。

# Result

## compare study

### DrugBAN

- Table 1 shows the comparison on the BindingDB and BioSNAP datasets. DrugBAN has consistently outperformed baselines in terms of AUROC, AUPRC and accuracy, while its performance in sensitivity and specificity is also competitive.
- **The results indicate that** data-driven representation learning can capture more important information than pre-defined descriptor features in in-domain DTI prediction. Moreover, DrugBAN can capture interaction patterns through its pairwise interaction module, further improving prediction performance.

### FragXsiteDTI

- Notably, our proposed model demonstrates superior prediction performance compared to all these models. It achieves competitive results with AttentionSiteDTI, which currently holds the top performance among them.
- Building upon this foundation, our model further enhances accuracy, highlighting the quality of features extracted and learned during the end-to-end training process of fragXsiteDTI.

## ablation study

### DrugBAN

- Here, we conduct an ablation study to investigate the influences of
bilinear attention and domain adaptation modules on DrugBAN.
- To validate the effectiveness of bilinear attention, we study three variants of DrugBAN that differ in the joint representation computation between drug and protein: one-side drug，…
    - The one-side attention is equivalent to the neural attention mechanism
    introduced in ref.
    - As shown in the first four rows of Table 2, the results demonstrate that bilinear attention is the most effective method to capture interaction information for DTI prediction.
- To examine the effect of CDAN,

### AttentionMGTDTA

- To validate the contribution and effectiveness of each module in AttentionMGT-DTA, we conducted ablation studies on the Davis dataset. We performed ablation experiments by removing pretrained protein embeddings, cross-attention and joint-attention modules. The performances of the models with different modules are listed in Table 9.
- Specifically, the first model AttentionMGT-DTA*𝑐𝑜𝑛𝑐𝑎𝑡* concatenated the protein graph embeddings and pretrained sequence embeddings instead of using the cross-attention mechanism. AttentionMGT-DTA*𝑠𝑖𝑛𝑔𝑙𝑒* is the model without pretrained protein embedding features. For the third model AttentionMGT-DTA*𝑚𝑎𝑥*, the joint-attention mechanism was removed by adding a max pooling layer after graph transformers, and the pretrained embedding was also aggregated into a vector with the same dimension by the maximizing function. Analogously, AttentionMGT-DTA*𝑚𝑒𝑎𝑛* was the model utilizing the mean pooling method to replace the joint-attention module.
- As shown in Table 9, AttentionMGT-DTA **maintained the best performance compared with the variant models.**
- **Removal of the cross attention module** significantly decreased the prediction performance
of the model, which clearly indicated the importance of multi-modal
interaction.

## Parameter study

### AttentionMGT-DTA

- We explored the effect of the hyperparameters in AttentionMGTDTA: (1) The threshold of protein graph construction; (2) The number of heads and layers of the graph transformer module; (3) The dimension of embeddings of drugs and proteins. We implemented the parameter analysis experiment on the Davis dataset.
- we estimated the impact of two hyperparameters in the graph transformer module with grid
search: the number of attention heads with search range {1, 2, 4, 8} and the number of layers with search range {2, 3, 5, 10}.
- 

# 可视化

## FragXsiteDTI

- Protein：
    - In this study, we leverage the attention mechanism to enhance the model’s ability to predict the likelihood of specific protein binding sites interacting with a given ligand. This likelihood is quantified through the attention matrix computed within the model. The visualization of this attention mechanism can be observed in Figure 2, where it is presented as a heatmap for the protein with PDB code of 4BHN when interacting with a drug characterized by the molecular formula of *C*21*H*23*Cl*2*N*5*O*2. This figure also includes the projection of the heatmap onto the protein structure.
- Drug
    - We also have an attention matrix for each drug molecule that determines which fragments of that drug have the highest probabilities for interaction with the particular protein. These candidate fragments can explain the chemical properties that caused the interaction or be used for designing and generating new drugs. Figure 3 demonstrate an example of certain drug (*C*21*H*23*Cl*2*N*5*O*2) that binds with the target protein (4BHN).

## DrugBAN

- A further strength of DrugBAN is to enable molecular level insights and interpretation critical for drug design efforts, utilizing the com ponents of the bilinear attention map to visualize the contribution of each substructure to the final predictive result.
- Here, we examine the top three predictions (PDB IDs: 6QL2 (ref. 37), 5W8L (ref. 38) and 4N6H
(ref. 39)) of co-crystalized ligands from the Protein Data Bank (PDB)40.
- 5W8L：

## MolTrans

### result

![Untitled](Untitled%2010.png)

### method

- drug 2-nonyl n-oxide, and the protein cytochrome b-c1 complex unit 1
- visualize the interaction map by filtering scalars that are larger than a threshold
- 

## LDH-A（人类乳酸脱氢酶）

背景

- 肿瘤细胞通常依赖于糖酵解过程来合成三磷酸腺苷（ATP）
- **LDH酶**: 是一个关键的糖酵解酶
- "通过敲低或沉默 LDHA 基因降低 LDH 活性已被证明能够在缺氧条件下减少肿瘤细胞的生长，并在肿瘤异种移植模型中抑制肿瘤生长”
- 总结：LDH-A参与癌细胞生产ATP的主要环节->为癌细胞供能->使用LDH-A抑制剂在癌症治疗中有一定作用

摘要

论文报告了一类新型的基于吡唑的人类乳酸脱氢酶（LDH）抑制剂的发现和药物化学优化。通过定量高通量筛选（qHTS）范式识别出活性分子，并通过基于结构的设计和多参数优化，开发出具有强效酶促和基于细胞的LDH酶活性抑制的化合物。

- 总结：通过一系列生化方法发现并优化了基于吡唑的化合物→这些化合物可以作为LDH-A抑制剂
- 吡唑结构：我们模型错误预测的top2结构