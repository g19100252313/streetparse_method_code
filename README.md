# 街景语义解析与空间量化方法代码说明

本代码包提供 ResNet-50 编码、多尺度上下文、细节门控融合、边界辅助监督及公共环境要素空间量化的实现，支持训练、验证、单图推理、消融实验与多种子统计。本文档聚焦方法结构与运行步骤，性能指标以实际运行产生的评价文件为准。

## 代码结构

| 路径 | 功能 |
| --- | --- |
| `streetparse/models.py` | ResNet-50、ASPP、细节门控、语义与边界双分支 |
| `streetparse/losses.py` | 语义损失、边界目标生成与辅助监督 |
| `streetparse/data.py` | Cityscapes 读取、标签转换、同步数据增强 |
| `streetparse/labels.py` | 19类语义标签与10组环境要素映射 |
| `streetparse/quantify.py` | 占比、归一化质心、网格分布及配对误差 |
| `streetparse/metrics.py` | mIoU、语义边界 F1 |
| `streetparse/engine.py` | 训练、检查点恢复及验证 |
| `streetparse/scenes.py` | 四类场景划分及分组统计 |
| `streetparse/cli.py` | 统一命令入口 |
| `experiments.py` | 内部调参、正式训练、对比与消融计划 |
| `subset_experiments.py` | 固定子集实验入口 |
| `tests/` | 方法、数据、场景与训练逻辑检查 |

## 环境安装

在解压后的代码根目录打开 PowerShell，准备 Python 环境。按计算机的 CUDA 环境安装对应版本的 PyTorch 与 torchvision，再安装其他依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements-lock.txt` 保存原开发环境的版本清单，供环境核对。训练默认自动选择设备，`--device cuda` 指定 GPU，`--device cpu` 指定 CPU。ImageNet 预训练骨干在首次使用时下载；`--no-pretrained` 启用随机初始化。

## 数据准备

压缩包仅提供代码与文档，数据集需单独准备。数据路径通过 `--data` 指定。支持以下两种目录结构：

```text
Cityscapes/
  data/
    leftImg8bit_trainvaltest.zip
    gtFine_trainvaltest.zip
```

或解压后的结构：

```text
Cityscapes/
  leftImg8bit/train/<city>/...
  leftImg8bit/val/<city>/...
  gtFine/train/<city>/...
  gtFine/val/<city>/...
```

本机数据路径为 `E:\wksp\tests\test1\Cityscapes`。以下命令使用该路径，迁移后替换为对应位置。

```powershell
.\.venv\Scripts\python.exe -m streetparse.cli check-data --data E:\wksp\tests\test1\Cityscapes --full
```

训练使用官方训练划分，评价使用具有公开精细标签的官方验证划分。内部调参城市为 darmstadt、jena、weimar。

## 本方法训练与验证

```powershell
.\.venv\Scripts\python.exe -m streetparse.cli train --data E:\wksp\tests\test1\Cityscapes --architecture proposed --boundary-weight 0.2 --seed 2026 --epochs 100 --output runs/proposed_seed2026
.\.venv\Scripts\python.exe -m streetparse.cli evaluate --data E:\wksp\tests\test1\Cityscapes --checkpoint runs/proposed_seed2026/last.pt --output outputs/proposed_seed2026
```

`0.2` 为启动示例的边界损失权重；正式流程通过内部调参选择权重。默认训练采用 Adam、初始学习率 1e-4、poly 衰减、512×1024 裁剪、micro-batch 1、梯度累积4次与 CUDA 混合精度。图像与标签同步执行几何增强，标签使用最近邻插值。

正式调参与多种子流程如下，默认仅生成计划，`--run` 执行任务：

```powershell
.\.venv\Scripts\python.exe experiments.py tune --data E:\wksp\tests\test1\Cityscapes --run
.\.venv\Scripts\python.exe experiments.py select
.\.venv\Scripts\python.exe experiments.py final --data E:\wksp\tests\test1\Cityscapes --variants proposed --run
```

最终训练种子为2026、2027、2028。去掉 `--variants proposed` 则执行代码提供的全部对比与消融配置。GSCNN、HRNet-OCR、SegFormer、Mask2Former、STDC-Seg 的作者实现不在本包中。

## 单图推理

```powershell
.\.venv\Scripts\python.exe -m streetparse.cli predict --checkpoint runs/proposed_seed2026/last.pt --input path/to/image.png --output outputs/prediction
```

输入也支持图像目录。输出语义标签图、要素分组图、配色图、叠加图和空间量化 JSON。预测概率先恢复原始尺寸，再确定像素类别。

## 空间量化与统计口径

19类语义归并为植被、天空、建筑、墙体与围栏、人行道、道路、地形、杆件与交通设施、人员、车辆10组。量化结果记录要素占比、归一化质心、10×10网格占比以及左中右区域分布。指标描述图像视域内的组成与布局。

配对评价采用参考标签定义的有效像素域；质心误差仅统计预测与参考均存在的有效配对，并记录存在性统计。占比误差使用百分点，质心采用归一化距离，网格误差为有效网格的平均绝对偏差。边界 F1 使用原尺寸2像素匹配容差。

```powershell
.\.venv\Scripts\python.exe -m streetparse.cli quantify --labels outputs/prediction/image_trainIds.png --reference path/to/image_gtFine_labelIds.png --output outputs/paired.json
.\.venv\Scripts\python.exe -m streetparse.cli aggregate --reports outputs/proposed_seed2026/summary.json outputs/proposed_seed2027/summary.json outputs/proposed_seed2028/summary.json --output outputs/proposed_aggregate.json
```

评价结果由 `summary.json`、`per_image_group.csv`、逐图量化 JSON 与示例图组成。多种子汇总计算运行均值及样本标准差。

## 固定入口

```powershell
.\.venv\Scripts\python.exe -m streetparse.cli prepare-subset --data 
.\.venv\Scripts\python.exe subset_experiments.py --data E:\wksp\tests\test1\Cityscapes --variants proposed --run
```

## 代码检查与包内容

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streetparse.cli --help
```

本包未附带数据集、样本清单、训练权重、预训练缓存、实验结果、论文文件或图表生成脚本。解压后创建的 `runs/`、`outputs/` 保存本机实际运行产物。核心源码保持原项目实现；本说明不将预设数值或生成式示意图作为代码实测结果。
