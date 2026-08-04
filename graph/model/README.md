# NEMSIS KG-assisted Protocol Prediction

该项目根据四类症状/临床印象预测单个 Protocol，并支持加入年龄和首个生命体征。

## 模型

- 训练集构建概念 KG，不使用验证集或测试集的标签边。
- 节点：临床代码、ICD 类别、Protocol。
- 关系：四种字段特定的 `code → protocol` 关系及反向关系，以及
  `code ↔ ICD category`。
- 两层 relation-aware message passing 生成概念表示。
- 每条 PCR 不作为永久图节点；通过字段注意力池化动态生成 PCR 表示，
  因而可以归纳预测新病例。
- 相同四字段组合通过稳定哈希分到同一个集合，比例约为 7:1:2。

## 基础训练

```bash
python train_kg_protocol.py \
  --data /home/xiangling/nemsis_cache_files/nemsis_pcr2si_protocol_single_label.txt \
  --cache cache/protocol_dataset.npz \
  --output-dir outputs/kg_protocol \
  --epochs 20 \
  --batch-size 4096
```

## 加入年龄和初始生命体征

首次运行需要顺序扫描 `ComputedElements.txt` 和约 52 GB 的
`FACTPCRVITAL.txt`，之后直接使用缓存：

```bash
python train_kg_protocol.py \
  --data /home/xiangling/nemsis_cache_files/nemsis_pcr2si_protocol_single_label.txt \
  --raw-dir /data_8TB_2/xiangling_workspace/NEMSIS_ASCII_25 \
  --use-raw-context \
  --cache cache/protocol_dataset_with_context.npz \
  --output-dir outputs/kg_protocol_context
```

连续特征只使用训练集均值和标准差处理。缺失值用训练均值填充，并为每一项
额外加入缺失指示变量。

## 输出

- `best_model.pt`：按验证集 Macro-F1 选择的模型
- `metrics.json`：训练历史、数据切分数量和测试指标
- `vocabulary.json`：代码、Protocol 标签和上下文特征名称

测试指标包括 Top1、Top3、Top5、Accuracy、Macro Precision、Macro Recall
和 Macro F1。Top1 与 Accuracy 对单标签任务在数学上相同，二者仍分别输出。

