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
| `portfolio-cardinality` | Continuous + Binary | 2次 | Continuousの2次目的関数とBinaryの基数・連結制約 | `regular` | 50 / 100 / 200 |
| `unit-commitment` | Integer + Binary | 2次 | Integerの2乗項とBinaryの起動・連結制約 | `regular` | 50 / 200 / 450 |
| `clique` | Binary | 定数0 | 1次等式と2次等式制約の変換 | `regular` | Instance → Model: 50 / 100 / 200、Result → Solution: 10 / 20 / 30 |
| `tsp` | Binary | 2次 | 通常制約とOneHotの比較 | `regular` / `one-hot` | 10 / 20 / 30 |
| `one-hot-preparation` | Binary | 1次 | Indicator/SOS1のPreparationとPreparation後のAdapter変換 | `one-hot` | 10 / 20 / 30 |

`size` は `knapsack`、`production`、`blending` では変数数、
`assignment` と `tsp` では一辺の要素数、`facility-location` では施設数と顧客数、
`portfolio` と `portfolio-cardinality` では資産数、`unit-commitment` では発電機数、
`clique` では頂点数、`one-hot-preparation` ではOneHotグループ数と各グループの変数数を表します。
`clique` は、2次制約の変換負荷を測る `instance-to-model` では
50 / 100 / 200、Amplifyが実行可能解を返せる規模でデコードを測る
`result-to-solution` では 10 / 20 / 30 を使用します。

制約表現は `--formulation regular` または `--formulation one-hot` で選択します。
`one-hot` を選択できるのは `assignment`、`tsp`、`one-hot-preparation` だけです。
v3のOneHotはfirst-classな `OneHotConstraint` として表現します。
`regular` では同じ数式を通常の等式制約として、`one-hot` ではOneHot特殊制約として生成します。
Adapterの `INPUT_CLASS` はOneHotを直接受け取るため、後者は `amplify.one_hot` に変換されます。
この比較では、同じ数理構造を通常制約とOneHot特殊制約で表した場合の変換時間とメモリの違いを測定します。

### Preparation

`one-hot-preparation` は、Preparation性能を測定するための専用Instanceです。
`--special-constraints` で制約family、`--preparation` でdirect/preparedを選びます。

| Case | `preparation=none` | `preparation=recommended` のsource |
| --- | --- | --- |
| `none` | OneHotのみ | 選択不可 |
| `indicator` | Indicator lowering済み通常制約 | OneHot + Indicator |
| `sos1` | SOS1 lowering済み通常制約 | OneHot + SOS1 |
| `indicator-sos1` | lowering済み通常制約 | 前半グループにIndicator、後半グループにSOS1 |

特殊制約を含む各caseは、direct/preparedともOneHot `size` 個と通常制約 `size` 個をactiveに持ちます。
`indicator-sos1` でも各グループにはIndicatorまたはSOS1の一方だけを割り当て、制約数を揃えます。
directとpreparedから生成されるAmplify Modelは同一で、preparedだけがremoved constraintsとprovenanceを保持します。
IndicatorとSOS1はOneHotから導かれる冗長制約なので、全ケースの実行可能領域と最適値は同一です。
PreparationはコピーしたInstanceへ
`OMMXFixstarsAmplifyAdapter.recommended_preparation_policy()` を適用します。

## 測定対象

`prepare` はInstanceの生成、コピー、Policy生成を測定外とし、`Instance.prepare()`だけを測定します。
`instance-to-model` は `preparation=recommended` の場合だけPreparationを測定外で済ませ、Adapterの生成だけを測定します。
`result-to-solution` は必要なPreparationとAmplifyでの求解を測定外で一度行い、`adapter.decode(result)`だけを測定します。
時間測定では、プロセス内でウォームアップ前の初回実行時間と、ウォームアップ後20回の中央値を記録します。
メモリ測定では、ウォームアップ前の初回実行と、その実行をウォームアップとした2回目のピークメモリを記録します。
時間計測中はGCを停止します。
求解準備で使用するsolver time limitは、Amplify APIの上限に合わせて既定値の10秒です。

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

for special_constraints in indicator sos1 indicator-sos1; do
  for size in 10 20 30; do
    uv run --frozen python benchmarks/timing.py prepare \
      --instance one-hot-preparation --formulation one-hot \
      --special-constraints "$special_constraints" \
      --preparation recommended --size "$size" \
      | tee "benchmark_results/v3-${special_constraints}-prepare-timing-${size}.csv"

    for preparation in none recommended; do
      uv run --frozen python benchmarks/timing.py instance-to-model \
        --instance one-hot-preparation --formulation one-hot \
        --special-constraints "$special_constraints" \
        --preparation "$preparation" --size "$size" \
        | tee "benchmark_results/v3-${special_constraints}-${preparation}-instance-to-model-timing-${size}.csv"
    done
  done
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

for special_constraints in indicator sos1 indicator-sos1; do
  for size in 10 20 30; do
    for preparation in none recommended; do
      uv run --frozen python benchmarks/timing.py result-to-solution \
        --instance one-hot-preparation --formulation one-hot \
        --special-constraints "$special_constraints" \
        --preparation "$preparation" --size "$size" \
        | tee "benchmark_results/v3-${special_constraints}-${preparation}-result-to-solution-timing-${size}.csv"
    done
  done
done
```

## ピークメモリ

サイズごとに別プロセスで実行します。

```console
uv run --frozen --with memray python benchmarks/memory.py instance-to-model \
  --instance tsp --formulation regular --size 20

uv run --frozen --with memray python benchmarks/memory.py instance-to-model \
  --instance tsp --formulation one-hot --size 20

uv run --frozen --with memray python benchmarks/memory.py prepare \
  --instance one-hot-preparation --formulation one-hot \
  --special-constraints indicator-sos1 \
  --preparation recommended --size 20

uv run --frozen --with memray python benchmarks/memory.py instance-to-model \
  --instance one-hot-preparation --formulation one-hot \
  --special-constraints indicator-sos1 \
  --preparation recommended --size 20
```
