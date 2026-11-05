# 手写数字上的准确率路径

数据换成 sklearn 的 8×8 手写数字，像素留在 0 到 1。按源图像逐像素做标准化会把滚进空白边缘的墨水放大到几百，分类器学不动，所以这里不再除以那个标准差。源分布是原图像。目标分布把每张图沿水平方向滚两个像素，标签不变。参考窗前一半是源，后一半是目标。池子里一半原图、一半滚过的图。测试集全是滚过的图。

查询规则接的是 deep active learning 里能在同一个分类网络上算的那些：不确定性、dropout 版、BALD、BatchBALD、core-set、k-center、k-means、cluster-margin、BADGE、baseline、BIM、DeepFool、ALBL。每条规则跑两遍，一遍权重是平的，一遍权重是参考窗上拟合的伪结果残差。VAAL、WAAL、learning-loss、GCN 和半监督那一组要换训练器，没有接进这个网络。

种子 0, 1。左图是各规则的平均路径，色带是规则之间的标准差，虚线是 10 类的随机猜测。k-center 和 core-set 是同一条最远点规则，平均和右图只计 core-set 一次。右图是终点准确率的差，伪结果权重减去无权重。

![Test accuracy by round](accuracy_paths.png)

路径下的面积：无权重 0.305，伪结果权重 0.237。抽到的目标行比例：无权重 0.433，伪结果权重 0.338。池子里这个比例是 0.50。

伪结果权重没有抬高终点准确率。无权重平均 0.606，加权平均 0.464。
终点最高的是 `adversarial_bim_po`，准确率 0.853，比起点高 +0.768。

| 方案 | 查询 | 权重 | 起点 | 终点 | lift | 面积 | 目标行 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| adversarial_bim_po | adversarial_bim | po_risk | 0.085 | 0.853 | +0.768 | 0.466 | 0.997 |
| margin_po | margin | po_risk | 0.085 | 0.825 | +0.740 | 0.456 | 0.819 |
| adversarial_deepfool_po | adversarial_deepfool | po_risk | 0.085 | 0.780 | +0.695 | 0.411 | 0.684 |
| margin | margin | uniform | 0.085 | 0.770 | +0.685 | 0.398 | 0.600 |
| adversarial_deepfool | adversarial_deepfool | uniform | 0.085 | 0.770 | +0.685 | 0.395 | 0.572 |
| least_confidence_dropout | least_confidence_dropout | uniform | 0.085 | 0.752 | +0.667 | 0.363 | 0.622 |
| adversarial_bim | adversarial_bim | uniform | 0.085 | 0.748 | +0.662 | 0.380 | 0.516 |
| albl | albl | uniform | 0.085 | 0.728 | +0.643 | 0.320 | 0.578 |
| random | random | uniform | 0.085 | 0.725 | +0.640 | 0.376 | 0.503 |
| least_confidence | least_confidence | uniform | 0.085 | 0.722 | +0.637 | 0.367 | 0.628 |
| margin_dropout_po | margin_dropout | po_risk | 0.085 | 0.722 | +0.637 | 0.440 | 0.841 |
| margin_dropout | margin_dropout | uniform | 0.085 | 0.700 | +0.615 | 0.408 | 0.591 |
| badge | badge | uniform | 0.085 | 0.690 | +0.605 | 0.319 | 0.381 |
| baseline | baseline | uniform | 0.085 | 0.650 | +0.565 | 0.328 | 0.422 |
| albl_po | albl | po_risk | 0.085 | 0.633 | +0.548 | 0.271 | 0.472 |
| kmeans | kmeans | uniform | 0.085 | 0.625 | +0.540 | 0.344 | 0.334 |
| kmeans_po | kmeans | po_risk | 0.085 | 0.625 | +0.540 | 0.317 | 0.341 |
| random_po | random | po_risk | 0.085 | 0.620 | +0.535 | 0.298 | 0.294 |
| baseline_po | baseline | po_risk | 0.085 | 0.595 | +0.510 | 0.266 | 0.325 |
| entropy_dropout | entropy_dropout | uniform | 0.085 | 0.590 | +0.505 | 0.264 | 0.506 |
| entropy | entropy | uniform | 0.085 | 0.545 | +0.460 | 0.255 | 0.453 |
| badge_po | badge | po_risk | 0.085 | 0.522 | +0.437 | 0.233 | 0.259 |
| coreset | coreset | uniform | 0.085 | 0.515 | +0.430 | 0.237 | 0.269 |
| kcenter | kcenter | uniform | 0.085 | 0.515 | +0.430 | 0.237 | 0.269 |
| cluster_margin_po | cluster_margin | po_risk | 0.085 | 0.320 | +0.235 | 0.149 | 0.131 |
| least_confidence_po | least_confidence | po_risk | 0.085 | 0.300 | +0.215 | 0.115 | 0.150 |
| batch_bald | batch_bald | uniform | 0.085 | 0.265 | +0.180 | 0.151 | 0.131 |
| bald | bald | uniform | 0.085 | 0.258 | +0.172 | 0.148 | 0.122 |
| entropy_po | entropy | po_risk | 0.085 | 0.253 | +0.167 | 0.116 | 0.128 |
| cluster_margin | cluster_margin | uniform | 0.085 | 0.250 | +0.165 | 0.139 | 0.141 |
| coreset_po | coreset | po_risk | 0.085 | 0.223 | +0.138 | 0.116 | 0.069 |
| kcenter_po | kcenter | po_risk | 0.085 | 0.223 | +0.138 | 0.116 | 0.069 |
| least_confidence_dropout_po | least_confidence_dropout | po_risk | 0.085 | 0.200 | +0.115 | 0.109 | 0.084 |
| entropy_dropout_po | entropy_dropout | po_risk | 0.085 | 0.158 | +0.072 | 0.104 | 0.075 |
| batch_bald_po | batch_bald | po_risk | 0.085 | 0.145 | +0.060 | 0.077 | 0.050 |
| bald_po | bald | po_risk | 0.085 | 0.113 | +0.027 | 0.077 | 0.034 |

k-center 与 core-set 的数字相同，因为两者都是嵌入上的最远点。上面的平均没有把它算第二次。

重复次数 2。
