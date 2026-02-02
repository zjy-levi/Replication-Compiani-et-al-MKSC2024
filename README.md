# Python Replication — Demand-Side Estimation (Expedia / Table 5)

## 中文简介
本仓库包含我们对论文“Will MKSC Online Search and Optimal Product Rankings: An Empirical Framework”中**需求侧模型（Table 5）**的 Python 复现代码：数据预处理、（条件）对数似然、Monte Carlo 积分，以及两种优化器驱动脚本。

### 复现范围
- 需求侧估计主程序：`scripts/run_estimation.py`（默认 L-BFGS-B 风格）
- 另一套“内点/信赖域”式优化：`scripts/run_estimation_ip.py`（trust-constr 风格）
- Bootstrap 驱动：`scripts/run_bootstrap.py`

### 目录结构
- `scripts/`：核心实现（数据处理、似然、估计驱动）
- `data/`：本地数据与输出（**不随仓库发布**，见 `.gitignore`）
- `logs/`：运行日志（**不随仓库发布**，见 `.gitignore`）
- `run.sh`：本地便捷脚本（**不随仓库发布**；命令已写入 `README.txt`）

### 数据说明（仓库不包含数据）
由于数据体积/许可原因，本仓库不包含 `data/`。你需要在本地准备。数据可从paper online appendix中下载 (https://doi.org/10.1287/mksc.2022.0071.)：
- 训练数据：`data/preparedata_new.parquet`（或 `.csv`）
- 初始值：`data/initial_value.csv`（或 `.mat`）
- 可选：`data/randcustsample_seed123.csv`、`data/eps_draws_seed123.csv`（用于更接近 MATLAB 的抽样与 eps 抽样）

数据列要求由 `scripts/data_utils.py` 使用（例如 `customeri`, `hotel`, `position`, `prop_starrating`, `prop_review_score`, `price_usd`, `click`, `book` 等）。

### 环境与依赖
在本机开发/测试环境：
- Python：3.11.7
- OS：Ubuntu 20.04（kernel 5.15）
- CPU：Intel Xeon Platinum 8268（96 线程可见）
- 内存：约 503 GiB
- GPU：未使用/未检测到

Python 依赖（最低集合）：
- `numpy`, `pandas`, `scipy`, `tqdm`
- 读取 parquet 需要 `pyarrow`（或 `fastparquet`）
- 可选加速：`numba`（未安装时会自动降级为纯 Python）

### 如何运行
请直接看 `README.txt`（只包含可复制粘贴的命令）。建议先用小样本/小迭代跑通，再用 Table 5 参数长跑。

---

## English Summary
This repo contains our Python replication of the paper’s **demand-side model estimation (Table 5)**: preprocessing, conditional log-likelihood with Monte Carlo integration, and two optimizer drivers.

### Scope
- Main estimation driver: `scripts/run_estimation.py`
- Interior-point / trust-region style driver: `scripts/run_estimation_ip.py`
- Bootstrap driver: `scripts/run_bootstrap.py`

### Layout
- `scripts/`: core implementation
- `data/`: local-only data + outputs (**not pushed**, see `.gitignore`)
- `logs/`: local-only logs (**not pushed**, see `.gitignore`)
- `run.sh`: local helper script (**not pushed**); commands are documented in `README.txt`

### Data (not included)
You must provide your own `data/` folder locally, e.g. Data can be collected from online Appendix of the original paper (https://doi.org/10.1287/mksc.2022.0071.).
- `data/preparedata_new.parquet` (or `.csv`)
- `data/initial_value.csv` (or `.mat`)
- Optional: `data/randcustsample_seed123.csv`, `data/eps_draws_seed123.csv`

Required columns are those consumed by `scripts/data_utils.py` (e.g., `customeri`, `hotel`, `position`, covariates, `click`, `book`, etc.).

### Environment & dependencies
Tested environment:
- Python 3.11.7
- Ubuntu 20.04 (kernel 5.15)
- CPU: Intel Xeon Platinum 8268 (96 threads visible)
- RAM: ~503 GiB
- GPU: not used / not detected

Python dependencies:
- `numpy`, `pandas`, `scipy`, `tqdm`
- Parquet IO: `pyarrow` (or `fastparquet`)
- Optional acceleration: `numba` (auto-fallback if unavailable)

### How to run
See `README.txt` for copy-paste commands. Start with a small sanity run before the full Table-5-sized optimization.

