# Adapter conversion benchmarks

問題は `--instance`、サイズは `--size` で選択します。
すべて固定seedからOMMX v3 Instanceを直接生成します。

## Instance

| Instance | 変数型 | 目的関数 | 制約・測定目的 | Formulation | 推奨サイズ |
| --- | --- | --- | --- | --- | --- |
| `knapsack` | Binary | 1次 | Binaryの線形変換と不等式 | `regular` | 100 / 400 / 900 |
| `production` | Integer | 1次 | 上下限付きIntegerと複数の不等式 | `regular` | 100 / 400 / 900 |
| `blending` | Continuous | 1次 | Realと上下限の線形変換 | `regular` | 100 / 400 / 900 |
| `assignment` | Binary | 1次 | 通常制約とOneHotの比較 | `regular` / `one-hot` | 10 / 20 / 30 |
| `facility-location` | Binary + Continuous | 1次 | 混合変数型と連結不等式 | `regular` | 10 / 20 / 30 |
| `portfolio` | Continuous | 2次 | Realの2次目的関数と予算上限 | `regular` | 50 / 100 / 200 |
| `tsp` | Binary | 2次 | 通常制約とOneHotの比較 | `regular` / `one-hot` | 10 / 20 / 30 |

`size` は `knapsack`、`production`、`blending` では変数数、
`assignment` と `tsp` では一辺の要素数、`facility-location` では施設数と顧客数、
`portfolio` では資産数を表します。

制約表現は `--formulation regular` または `--formulation one-hot` で選択します。
`one-hot` を選択できるのは `assignment` と `tsp` だけです。
v3のOneHotはfirst-classな `OneHotConstraint` として表現します。
`regular` では同じ数式を通常の等式制約として、`one-hot` ではOneHot特殊制約として生成します。
Adapterは `AdditionalCapability.OneHot` を宣言しているため、後者は `amplify.one_hot` に直接変換されます。
この比較では、同じ数理構造を通常制約とOneHot特殊制約で表した場合の変換時間とメモリの違いを測定します。

## 測定対象

`instance-to-model` はAdapterの生成だけを測定します。
`result-to-solution` はAmplifyでの求解を測定外で一度行い、`adapter.decode(result)`だけを測定します。
時間測定ではウォームアップ後の計測中にGCを停止し、メモリ測定ではTracker開始前に1回ウォームアップします。
求解準備で使用するsolver time limitは、すべてのInstanceで既定値の120秒です。

## 処理時間

```console
mkdir -p benchmark_results
for size in 100 400 900; do
  uv run --frozen python benchmarks/timing.py instance-to-model \
    --instance knapsack --formulation regular --size "$size" \
    | tee "benchmark_results/v3-knapsack-instance-to-model-timing-${size}.csv"
done

for size in 10 20 30; do
  uv run --frozen python benchmarks/timing.py instance-to-model \
    --instance tsp --formulation one-hot --size "$size" \
    | tee "benchmark_results/v3-one-hot-instance-to-model-timing-${size}.csv"
done
```

`Result -> Solution` の測定にはAmplify tokenが必要です。

```console
export AMPLIFY_TOKEN=YOUR_TOKEN
for size in 10 20 30; do
  uv run --frozen python benchmarks/timing.py result-to-solution \
    --instance tsp --formulation regular --size "$size" \
    | tee "benchmark_results/v3-regular-result-to-solution-timing-${size}.csv"
done

for size in 10 20 30; do
  uv run --frozen python benchmarks/timing.py result-to-solution \
    --instance tsp --formulation one-hot --size "$size" \
    | tee "benchmark_results/v3-one-hot-result-to-solution-timing-${size}.csv"
done
```

## ピークメモリ

サイズごとに別プロセスで実行します。

```console
uv run --frozen --with memray python benchmarks/memory.py instance-to-model \
  --instance tsp --formulation regular --size 20

uv run --frozen --with memray python benchmarks/memory.py instance-to-model \
  --instance tsp --formulation one-hot --size 20
```
