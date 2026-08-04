# Full-schema NEMSIS KG

这是独立于 `train_kg_protocol.py` 的完整 Schema 实现。

## 构图

```bash
python full_schema_kg_protocol.py \
  --stage build --rebuild \
  --labels /home/xiangling/nemsis_cache_files/nemsis_pcr2si_protocol_single_label.txt \
  --raw-dir /data_8TB_2/xiangling_workspace/NEMSIS_ASCII_25 \
  --kg cache/nemsis_full_schema.sqlite
```

构图会顺序扫描约 100 GB 原始表，耗时主要来自 `FACTPCRVITAL.txt`、
`FACTPCRMEDICATION.txt` 和 `FACTPCRPROCEDURE.txt`。SQLite 文件可以反复用于训练。

## 训练

```bash
python full_schema_kg_protocol.py \
  --stage train \
  --labels /home/xiangling/nemsis_cache_files/nemsis_pcr2si_protocol_single_label.txt \
  --raw-dir /data_8TB_2/xiangling_workspace/NEMSIS_ASCII_25 \
  --kg cache/nemsis_full_schema.sqlite \
  --output-dir outputs/full_schema_kg \
  --device cuda
```

默认预测邻域只包含决策时可获得的信息。`used_protocol` 永远不会输入预测器；
Medication、Procedure 和 HospitalDiagnosis 存在于完整 KG 中，但默认不作为预测
特征。`--include-post-treatment` 仅用于量化时间泄漏的消融实验，不应用于正式结果。

ProtocolCategory 无法从当前 0–45 标签可靠恢复。可以提供 JSON 映射：

```json
{"0": "Airway", "1": "Cardiac Arrest", "23": "Medical"}
```

并在构图时传入 `--protocol-category-json protocol_categories.json`。未提供映射的
Protocol 会连接到 `ProtocolCategory:UNMAPPED`，不会虚构医学类别。

