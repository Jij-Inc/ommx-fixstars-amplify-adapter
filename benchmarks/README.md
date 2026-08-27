# Adapter conversion benchmarks

問題は `--instance`、サイズは `--size` で選択します。
すべて固定seedからOMMX v2 Instanceを直接生成します。

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
| `one-hot-preparation` | Binary | 1次 | v3 Preparation後と同じ通常制約を持つ比較基準 | `one-hot` | 10 / 20 / 30 |

`size` は `knapsack`、`production`、`blending` では変数数、
`assignment` と `tsp` では一辺の要素数、`facility-location` では施設数と顧客数、
`portfolio` と `portfolio-cardinality` では資産数、`unit-commitment` では発電機数、
`clique` では頂点数、`one-hot-preparation` ではOneHotグループ数と各グループの変数数を表します。
`clique` は、2次制約の変換負荷を測る `instance-to-model` では
50 / 100 / 200、Amplifyが実行可能解を返せる規模でデコードを測る
`result-to-solution` では 10 / 20 / 30 を使用します。

制約表現は `--formulation regular` または `--formulation one-hot` で選択します。
`one-hot` を選択できるのは `assignment`、`tsp`、`one-hot-preparation` だけです。
v2のOneHotは通常の等式制約と `ConstraintHints.OneHot` の組で表現します。

現在のAdapterは `ConstraintHints` を参照せず、すべての制約を通常制約として変換します
(`amplify.one_hot` は未使用)。そのため `regular` と `one-hot` で生成されるAmplifyモデルは
同一であり、この比較はヒント付与が変換時間・メモリにオーバーヘッドを生まないことの確認が目的です。

### Preparation比較用counterpart

`one-hot-preparation` はv3側のPreparation性能測定専用Instanceと同じ決定変数、目的関数、
OneHotグループを持つ比較基準です。OMMX v2にはfirst-classなIndicator/SOS1と
`Instance.prepare()`がないため、`--special-constraints` は特殊制約そのものではなく、
対応するv3 Preparation後と同じ通常不等式をInstance生成時に直接追加します。

| Case | v2 Instanceに追加する通常制約 |
| --- | --- |
| `none` | 追加なし |
| `indicator` | v3でIndicatorからlowerされるBig-M不等式 |
| `sos1` | v3でSOS1からlowerされるat-most-one不等式 |
| `indicator-sos1` | 前半グループにIndicator相当、後半グループにSOS1相当 |

全ケースでPreparationは行わず、CSVの `preparation` は `none` です。
各caseで追加する通常制約は常に `size` 個で、OneHot等式と合わせたactive制約は
常に `2 * size` 個です。
追加制約はOneHotから導かれる冗長制約なので、4ケースの実行可能領域と最適値は同一です。

## 測定対象

`instance-to-model` はAdapterの生成だけを測定します。
`result-to-solution` はAmplifyでの求解を測定外で一度行い、`adapter.decode(result)`だけを測定します。
`end-to-end` はv2の `solve()` を使い、Adapter変換、Amplify求解、decodeを含む公開API全体を測定します。
v2にはPreparationがないため、このAPIはv3の `solve_without_preparation()` に対応します。
E2Eは通信とリモート求解の変動を含む補助指標であり、Adapter固有の変化は分解測定で判断します。
時間測定では、プロセス内でウォームアップ前の初回実行時間と中央値を記録します。
既定値はE2E以外がwarmup 3回・repeat 20回、E2Eがwarmupなし・repeat 3回です。
実際の値はCSVの `warmup` / `repeat` 列にも記録されます。
メモリ測定では、ウォームアップ前の初回実行と、その実行をウォームアップとした2回目のピークメモリを記録します。
時間計測中はGCを停止します。
求解準備で使用するsolver time limitは、Amplify APIの上限に合わせて既定値の10秒です。

## 処理時間

```console
mkdir -p benchmark_results
for size in 100 400 900; do
  uv run --frozen python benchmarks/timing.py instance-to-model \
    --instance knapsack --formulation regular --size "$size" \
    | tee "benchmark_results/v2-knapsack-instance-to-model-timing-${size}.csv"
done

for size in 10 20 30; do
  uv run --frozen python benchmarks/timing.py instance-to-model \
    --instance tsp --formulation one-hot --size "$size" \
    | tee "benchmark_results/v2-one-hot-instance-to-model-timing-${size}.csv"
done

for special_constraints in none indicator sos1 indicator-sos1; do
  for size in 10 20 30; do
    uv run --frozen python benchmarks/timing.py instance-to-model \
      --instance one-hot-preparation --formulation one-hot \
      --special-constraints "$special_constraints" --size "$size" \
      | tee "benchmark_results/v2-${special_constraints}-instance-to-model-timing-${size}.csv"
  done
done
```

`Result -> Solution` の測定にはAmplify tokenが必要です。

```console
export AMPLIFY_TOKEN=YOUR_TOKEN
for size in 10 20 30; do
  uv run --frozen python benchmarks/timing.py result-to-solution \
    --instance tsp --formulation regular --size "$size" \
    | tee "benchmark_results/v2-regular-result-to-solution-timing-${size}.csv"
done

for special_constraints in none indicator sos1 indicator-sos1; do
  for size in 10 20 30; do
    uv run --frozen python benchmarks/timing.py result-to-solution \
      --instance one-hot-preparation --formulation one-hot \
      --special-constraints "$special_constraints" --size "$size" \
      | tee "benchmark_results/v2-${special_constraints}-result-to-solution-timing-${size}.csv"
  done
done

for size in 10 20 30; do
  uv run --frozen python benchmarks/timing.py result-to-solution \
    --instance tsp --formulation one-hot --size "$size" \
    | tee "benchmark_results/v2-one-hot-result-to-solution-timing-${size}.csv"
done

for size in 10 20 30; do
  uv run --frozen python benchmarks/timing.py end-to-end \
    --instance tsp --formulation one-hot --size "$size" \
    | tee "benchmark_results/v2-one-hot-end-to-end-timing-${size}.csv"
done

for special_constraints in indicator sos1 indicator-sos1; do
  for size in 10 20 30; do
    uv run --frozen python benchmarks/timing.py end-to-end \
      --instance one-hot-preparation --formulation one-hot \
      --special-constraints "$special_constraints" --size "$size" \
      | tee "benchmark_results/v2-${special_constraints}-end-to-end-timing-${size}.csv"
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

uv run --frozen --with memray python benchmarks/memory.py instance-to-model \
  --instance one-hot-preparation --formulation one-hot \
  --special-constraints indicator-sos1 --size 20

uv run --frozen --with memray python benchmarks/memory.py end-to-end \
  --instance one-hot-preparation --formulation one-hot \
  --special-constraints indicator-sos1 --size 20
```
