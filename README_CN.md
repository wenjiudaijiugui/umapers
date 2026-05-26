# umapers

[English Version](README.md)

一个以 Rust 为中心的 UMAP 实现，并提供面向公平比较的 benchmark，
用于和 Python `umap-learn` 进行可复现对比。

仓库包含：

- `rust_umap/` 下的 Rust 库与 CLI
- `benchmarks/` 下的可复现 benchmark 框架
- `umap_rs/` 下的 `umapers` Python 包

## 仓库结构

- `rust_umap/`: Rust UMAP crate 与 CLI 二进制
- `umap_rs/`: 基于 PyO3 + maturin 的 Python 绑定，发布/导入名为 `umapers`
- `benchmarks/`: 面向公平比较的 benchmark 脚本与报告
- `reports/`: 生成的 benchmark 与回归产物
- `UMAP_MATHEMATICAL_DOCUMENTATION*.md`: 数学说明文档

## 快速开始

```bash
cd rust_umap
cargo build --release
cargo test
```

## Python 绑定

Python 绑定是刻意保持轻量的：

- Python 侧负责 dense 数组和 CSR 输入的归一化
- Rust 侧尽量负责校验和主要计算路径
- 预计算 kNN 接口会保留，但被定位为高级接口，不是默认入口

### 本地安装

```bash
PYTHON_BIN="$(command -v python3 || command -v python)"
if [ -z "$PYTHON_BIN" ]; then
  echo "python3/python not found" >&2
  exit 1
fi

uv venv --python "$PYTHON_BIN" .venv
uv pip install --python .venv/bin/python --upgrade pip maturin
uv run --python .venv/bin/python maturin develop --manifest-path umap_rs/Cargo.toml
```

### API 分层

#### Main API

这是稳定的公开 API，也是大多数用户应该优先学习的接口层。

- `from umapers import Umap, fit_transform`
- `Umap.fit(data)`
- `Umap.fit_transform(data, out=None)`
- `Umap.transform(query, out=None)`
- `Umap.inverse_transform(embedded_query, out=None)`
- `fit_transform(data, **kwargs)`

典型 dense 用法：

```python
import numpy as np
from umapers import Umap

rng = np.random.default_rng(42)
x = rng.normal(size=(400, 16)).astype(np.float32)

model = Umap(
    n_neighbors=15,
    n_components=2,
    n_epochs=120,
    metric="euclidean",
    random_seed=42,
    init="random",
)

emb = model.fit_transform(x)
print("embedding shape:", emb.shape, "dtype:", emb.dtype)
```

Main API 还支持：

- `fit` 与 `fit_transform` 的 CSR 稀疏输入
- `fit_transform`、`transform`、`inverse_transform` 的 `out=` 输出缓冲
- `ann_mode="auto" | "exact" | "approximate"` 这层 Python 便捷映射

#### Advanced API

`Umap.fit_transform_with_knn(...)` 是公开的高级接口。

它适合这些场景：

- 做公平 benchmark
- 在多组参数实验中复用同一张 kNN 图
- 对接外部近邻搜索流程

它要求的输入是：

- `data`: 形状为 `(n_samples, n_features)`，会被规整成 `float32`
- `knn_indices`: 形状为 `(n_samples, k)`，会被规整成 `int64`
- `knn_dists`: 形状为 `(n_samples, k)`，会被规整成 `float32`
- `knn_metric`: 必须和模型的 `metric` 一致

下面是使用共享精确 kNN 图的示例：

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from umapers import Umap

x = np.random.default_rng(42).normal(size=(300, 16)).astype(np.float32)
k = 15

nbrs = NearestNeighbors(
    n_neighbors=k + 1,
    algorithm="brute",
    metric="euclidean",
    n_jobs=1,
)
nbrs.fit(x)
dists, idx = nbrs.kneighbors(x)

knn_indices = idx[:, 1 : k + 1].astype(np.int64)
knn_dists = dists[:, 1 : k + 1].astype(np.float32)

model = Umap(
    n_neighbors=k,
    n_components=2,
    metric="euclidean",
    random_seed=42,
    init="random",
    use_approximate_knn=False,
)

emb = model.fit_transform_with_knn(
    x,
    knn_indices,
    knn_dists,
    knn_metric="euclidean",
)
```

这个接口被刻意保持在比“任意图输入 API”更窄的范围：

- 它接受的是预计算 kNN，而不是任意稀疏图
- 它是有价值的接口，但不是推荐的 quickstart 路径
- 做严格绑定对比时，建议把 `algo_exact_shared_knn_exact` 视为公平锚点

#### Internal API

下面这些属于内部实现细节，不承诺公开兼容性：

- `umapers._umapers.UmapCore`
- `umapers._api`
- 绑定包内部的 helper 和所有 `_` 前缀符号

## 当前范围边界

当前文档与 benchmark 明确覆盖的 Python 绑定能力包括：

- dense 单数据集 `Umap` 工作流
- dense 训练模型上的 `inverse_transform`
- 作为高级接口的 dense 预计算 kNN 拟合
- CSR 稀疏 `fit` / `fit_transform` MVP
- 稀疏训练后对 dense query 的 `transform`，支持 `euclidean`、`manhattan`
  和 `cosine`

当前范围有意小于完整 `umap-learn` 对等能力：

- 稀疏训练后的 `inverse_transform` 还不支持
- Python 包目前不暴露 parametric UMAP 或 aligned UMAP
- 与 `pynndescent` 的 ANN 质量和性能对等仍然不在范围内
- 绑定层当前不提供通用“任意图输入”API

更完整的范围决策见 `docs/adr/ADR-L8-scope-alignment.md`。

## Benchmark 报告

生态级 Python 绑定对比报告位于：

- `benchmarks/report_ecosystem_python_binding.md`
- `benchmarks/report_ecosystem_python_binding.json`

本地运行生态级对比：

```bash
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
"$PYTHON_BIN" benchmarks/compare_ecosystem_python_binding.py \
  --python-bin "$PYTHON_BIN" \
  --warmup 1 \
  --repeats 3 \
  --sample-cap-consistency 2000
```

输出文件：

- `benchmarks/report_ecosystem_python_binding.json`
- `benchmarks/report_ecosystem_python_binding.md`

`e2e_mixed_knn_strategy` 更适合生态级 smoke。若要做最严格的绑定对比，
优先看 `algo_exact_shared_knn_exact`。

如果要本地运行 core Rust 对 Python `umap-learn` 的公平对比：

```bash
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
"$PYTHON_BIN" benchmarks/compare_real_impls_fair.py \
  --python-bin "$PYTHON_BIN" \
  --warmup 1 \
  --repeats 3 \
  --sample-cap-consistency 2000
```

## 本地验证

```bash
PYTHON_BIN="$(command -v python3 || command -v python)"

cargo test --manifest-path rust_umap/Cargo.toml

uv venv --python "$PYTHON_BIN" .venv
uv pip install --python .venv/bin/python --upgrade pip
uv pip install --python .venv/bin/python -r benchmarks/requirements-bench.txt pytest maturin

uv run --python .venv/bin/python python -m py_compile \
  benchmarks/compare_real_impls_fair.py \
  benchmarks/compare_ecosystem_python_binding.py \
  benchmarks/run_umap_rs.py \
  benchmarks/run_umap_rs_algo.py

uv run --python .venv/bin/python maturin develop --manifest-path umap_rs/Cargo.toml
uv run --python .venv/bin/python python -I -m pytest -q umap_rs/tests/test_binding.py
```

如果要运行完整的本地回归与 release-prep 流程，请继续查看 `benchmarks/`
下的脚本，以及 `reports/` 下生成的报告。
