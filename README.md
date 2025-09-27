# Mitsui-Commodity-Pred

本项目聚焦大宗商品及相关资产的多资产收益率差预测。当前阶段已完成数据预处理与特征构建的基础设施，Transformer 模型部分留待后续迭代。

## 数据预处理
- **缺失值处理**：`src/data/cleaning.py` 中的 `fill_series_with_trend` 会根据最近 `window` 个有效观测计算平均增量，沿时间轴外推填补缺口，避免未来数据泄露。
- **数据管线**：通过 `build_all_cleaned(window=5)` 一次性生成 `train/test/all` 三份清洗结果，保存为 `artifacts/clean_*.pkl`。
- **目标构建**：`src/data/targets.py` 将 `target_pairs.csv` 中的资产差价表达式解析为 `TargetSpec`，在对数价格空间构造 424 维目标价差矩阵。

## 因子 / 特征生成
特征生成入口位于 `src/features/pipeline.py`，主要模块如下：
- **技术指标** (`technical.py`): 基于 TA-Lib 计算均线、MACD、布林带、RSI 等约 45 个指标。
- **经典金融因子** (`factors.py`): 包含收益率、波动率、Sharpe、动量、反转、区间极值等统计量。
- **PCA 潜在因子** (`pca.py`): 使用标准化后的收益矩阵提取前 5 个主成分载荷，并扩展为每个 target 的静态特征。
- **遗传编程特征** (`gp.py`): 借助 gplearn 的 `SymbolicTransformer` 在采样的训练子集上生成 15 个组合表达式，兼容最新 scikit-learn。

执行 `build_and_save_features()` 会在 `artifacts/features/` 下生成以下文件：
- `target_prices.pkl`：424 个目标序列的对数价差。
- `base.pkl`：技术指标 + 金融因子。
- `pca.pkl`：PCA 载荷特征。
- `gp.pkl`：遗传编程特征。
- `all.pkl`：合并后的完整特征矩阵。

## 使用示例
```powershell
# 激活环境
d:\py\genAI\venv\Scripts\Activate.ps1

# 安装依赖（首次）
pip install -r requirements.txt

# 生成清洗后的价格表
python - <<'PY'
from src.data.pipeline import build_all_cleaned
build_all_cleaned(window=5)
PY

# 生成特征矩阵
python - <<'PY'
from src.features.pipeline import build_and_save_features
build_and_save_features()
PY
```

## 代码结构概览
```
src/
├── config.py              # 路径与全局配置
├── data/
│   ├── cleaning.py        # 缺失值填补等数据清洗逻辑
│   ├── loading.py         # 数据读取工具
│   ├── pipeline.py        # 清洗管线入口（保存 clean_*.pkl）
│   └── targets.py         # target 解析与目标矩阵构建
└── features/
    ├── technical.py       # TA-Lib 技术指标
    ├── factors.py         # 金融统计因子
    ├── pca.py             # PCA 潜在因子及 artifacts
    ├── gp.py              # 遗传编程特征生成
    └── pipeline.py        # 特征整体拼装与持久化
```

---
后续工作计划包括：构建序列化训练集、实现多资产 Transformer 模型、评估与推理脚本等。