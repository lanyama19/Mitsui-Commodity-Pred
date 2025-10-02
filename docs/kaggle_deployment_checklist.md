# Kaggle Deployment Checklist

## 1. 打包与上传依赖
- 使用 `scripts/export_lightgbm_params.py` 固化 lag→参数映射；运行 `scripts/train_lightgbm_full.py` 产出 `artifacts/lightgbm_full/lag_*/tunedfull_*` 模型与 `*_summary_1960.json`。
- 将以下目录/文件整理到一个压缩包（例如 `mitsui-lightgbm-bundle.zip`）并在 Kaggle **Datasets** 中创建私人数据集：
  - `configs/lightgbm_lag_params.json`
  - `artifacts/lightgbm_full/lag_*/tunedfull_*`（含 `models/*.txt` 与 `preprocessing/*.json`）
  - 运行时需要的特征加工脚本：`src/`、`scripts/build_features_v2.py`、`src/features/pipeline_v2.py` 等
  - `requirements.txt` 中必要的第三方包列表（Kaggle Notebook 可直接 `pip install`）
  - 可选：将 `artifacts/features_v2/all_train.pkl` 和其他中间件压缩后上传，便于快速 warm-start
- 在 Kaggle Notebook 中通过 `kaggle datasets download` 或 `Add data` 方式挂载该数据集。

## 2. Notebook 初始化流程
1. `pip install -r requirements.txt`（根据 Kaggle CPU/GPU 环境选择安装子集：`lightgbm`, `polars`, `pyarrow`, `scikit-learn`, `numpy`, `pandas`）。
2. 从数据集解压模型与配置至工作目录，确保保持 `artifacts/lightgbm_full/lag_*/tunedfull_*` 的目录结构。
3. 载入 `configs/lightgbm_lag_params.json`，构建一个按 lag 组织的推理字典：包含 LightGBM Booster（用 `lgb.Booster(model_file=...)`）与对应的中位数填充/特征列表。
4. 加载训练期的特征缓存：
   - 若上传了 `all_train.pkl`，直接读入并保存在内存或 `Polars LazyFrame`，用于提供历史窗口。
   - 若仅上传原始 `train.csv` 和脚本，则在启动阶段运行 `scripts/build_features_v2.py --feature-version v2 --output-path ...` 复现特征。首个 batch 没有 1 分钟限制，可在 `predict` 的第一次调用里完成。

## 3. 在线推理数据流对接
- Kaggle 网关会按 `date_id` 顺序逐天推送 `test.csv` 的同日全部行，以及 4 个 `lagged_test_labels` 批次（同日窗口内的滞后标签）。
- 你的 `predict` 需要维护一个状态容器：
  - 合并历史原始数据：初始化时将 `train`（`train.csv`）或其特征版本读入；每次收到 `test_batch` 后将其追加到历史缓存。
  - 复用我们离线特征流水线：
    ```python
    from src.features.pipeline_v2 import build_feature_pipeline
    pipeline = build_feature_pipeline(lag=lag, horizon=1, ...)
    features_today = pipeline.transform(history_tail)
    ```
    确保 `history_tail` 覆盖所有需要的滚动窗口长度（见各特征模块默认窗口，例如 `src/features/technical_v2.py`）。
  - 对于滞后标签，利用网关提供的 `label_lags_*_batch` 直接拼入特征，不要再自行对 test 进行 shift，以免造成泄露。
- 将产生的特征矩阵按 target 拆分，应用训练期保存的同名特征顺序与中位数填充，再调用加载的 Booster 预测；拼接 424 维输出行返回。
- 建议使用 `polars` 进行增量更新，并为每个 lag 维护一个最近窗口的缓存（例如 `deque` 或 `DataFrame.tail(n)`），减少每批全量重算。

## 4. 线下验证与调试
- 使用 `demo_submission.py` 的 `inference_server.run_local_gateway` 在本地模拟 Kaggle 推理流程：将 `predict` 替换为你的封装函数，确认响应在 5 分钟内完成。
- 可编写单元测试确保特征滚动窗口与训练期一致，例如检查同一 `date_id`（≤1960）预测输出与离线训练时保存的 `train_predictions.pkl` 一致。
- 若需回滚到较小数据集验证，可截取 `date_id <= 100` 的切片建立 smoke 流程，然后再运行全量。

## 5. 额外的过拟合防护建议
- 继续保留当前的 `λ₁/λ₂` 与子采样设置；若观察到训练 IC 持续高于在线验证，可：
  1. 在参数中引入 `min_gain_to_split`（LightGBM 的 `min_gain_to_split`）来阻止弱增益分裂。
  2. 增加 `feature_fraction` 与 `bagging_fraction` 的随机性（例如添加 `bagging_seed` 与 `feature_fraction_bynode`）。
  3. 采用目标分箱或 Winsorization，减少极端标签的影响（可在特征流水线中加入 `clip` 步）。
  4. 引入交叉验证 stacking：训练多个 `seed` 的模型在 Kaggle 端平均，以降低方差。
- 在 Kaggle Notebook 中可根据上线表现调整 `weight_decay`（更大的衰减让模型更关注近月数据，但会增大波动）。

完成上述准备后即可在 Kaggle Notebook 中运行 `inference_server.serve()` 启动评分。

