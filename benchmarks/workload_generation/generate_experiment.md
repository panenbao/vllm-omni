# generate_experiment.py 使用说明

阶段 1 工具：基于分布参数生成请求时间戳与各字段 token 数，输出 CSV。

## 两种驱动方式

### 方式 A：CLI（单客户端 / 单组分布 / 分段速率）

```bash
# 纯文本：正态输入（均值 1000，std 200），指数输出（均值 300）
python generate_experiment.py -c language --input-len 1000 --output-len 300 \
    --rate 10 --duration 60

# 多模态：1 张图，每张 ~700 tokens，2 分钟，20 req/s
python generate_experiment.py -c multimodal \
    --image-count 1 --image-tokens 700 \
    --rate 20 --duration 120

# 分段速率：0–30s 是 5 req/s，30–60s 是 25 req/s
python generate_experiment.py -c multimodal --windows "0,30" --rate "5,25" \
    --image-count 1 --image-tokens 700 --duration 60

# 概率化 image 数量：20% 无图，60% 1 图，20% 2 图
python generate_experiment.py -c multimodal --image-tokens 700 \
    --image-count-probs 0.2,0.6,0.2 --rate 20 --duration 60
```

### 方式 B：JSON 配置（多客户端 / 多 window / 每 window 独立分布）

```bash
python generate_experiment.py --config configs/scenario_example.json
```

配置 schema：

```jsonc
{
  "category": "multimodal",       // language | multimodal | reason
  "duration": 60,
  "seed": 42,
  "output": "workload/foo.csv",   // 可省，缺省按规则自动生成
  "rate_schedule": {              // 时刻 -> 聚合速率（ServeGen rate_fn）
    "0": 5.0,
    "30": 25.0
  },
  "clients": [
    {
      "client_id": 0,
      "windows": {
        "0":  {"rate": 5.0,  "cv": 1.5, "pat": "Gamma", "pat_shape": 0.5, "pat_scale": 1.0},
        "30": {"rate": 25.0, "cv": 2.0, "pat": "Gamma", "pat_shape": 0.5, "pat_scale": 1.0}
      },
      "fields": {
        "0": {
          "text_tokens":   {"dist": "normal",      "loc": 500,  "scale": 100, "max": 1500},
          "output_tokens": {"dist": "exponential", "loc": 0,    "scale": 100, "max": 1000},
          "image_tokens":  {"dist": "normal",      "loc": 700,  "scale": 100, "max": 1500},
          "image_count":   {"dist": "categorical", "probs": [0.2, 0.6, 0.2]},
          "audio_tokens":  {"dist": "fixed", "loc": 0, "max": 1},
          "audio_count":   {"dist": "fixed", "loc": 0, "max": 1},
          "video_tokens":  {"dist": "fixed", "loc": 0, "max": 1},
          "video_count":   {"dist": "fixed", "loc": 0, "max": 1}
        },
        "30": { /* 新分布 */ }
      }
    }
  ]
}
```

## CLI 参数表

### 顶层

| 参数 | 默认 | 说明 |
|------|------|------|
| `-c, --category` | `language` | `language` / `multimodal` / `reason` |
| `-d, --duration` | `60` | 持续秒数 |
| `-s, --seed` | `42` | 随机种子 |
| `-o, --output` | 自动 | CSV 路径。默认 `workload/{category}_{duration}_{rate}_experiment.csv` |
| `--config` | — | JSON 配置文件；提供后 CLI 字段除 `--output` / `--dry-run` 都被覆盖 |
| `--dry-run` | 关 | 只打印解析后的 spec，不实际生成 |

### 到达模式

| 参数 | 默认 | 说明 |
|------|------|------|
| `-r, --rate` | `10.0` | 速率(req/s)。多 window 时逗号分隔，如 `5,25` |
| `--windows` | `0` | 多 window 起点，如 `0,30,60` |
| `--cv` | `1.5` | 到达间隔变异系数 |
| `--pat` | `Gamma` | `Gamma` / `Weibull` |
| `--pat-shape` | `0.5` | 到达分布形状 |
| `--pat-scale` | `1.0` | 到达分布尺度 |

### 输入/输出 token

每组都有 `--{prefix}-len / -dist / -scale / -shape / -max` 五个参数。

| 前缀 | 主名 / 别名 | 默认分布 | 含义 |
|------|-------------|----------|------|
| `input` | `--input-len`，`--text-len` | normal | 文本输入 token 数 |
| `output` | `--output-len` | exponential | 输出 token 数 |
| `image` | `--image-len`，`--image-tokens` | normal | 每张图 token 数 |
| `audio` | `--audio-len`，`--audio-tokens` | normal | 每段音频 token 数 |
| `video` | `--video-len`，`--video-tokens` | normal | 每个视频 token 数 |

### 多模态 item 计数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--image-count` | `1` | 固定每请求图数 |
| `--audio-count` | `0` | 固定每请求音频数 |
| `--video-count` | `0` | 固定每请求视频数 |
| `--image-count-probs` | — | 概率向量，如 `0.2,0.6,0.2` 表示 0/1/2 张图的概率 |
| `--audio-count-probs` | — | 同上 |
| `--video-count-probs` | — | 同上 |

## 分布参数对照

| `dist` | `loc` | `scale` | `shape` | `probs` |
|--------|-------|---------|---------|---------|
| `normal` | 均值 μ | 标准差 σ | — | — |
| `exponential` | — | 1/λ | — | — |
| `pareto` | — | 尺度 | 形状 a | — |
| `fixed` | 取值 | — | — | — |
| `uniform` | 中心值 | 半宽 | — | — |
| `categorical` | — | — | — | 0..len-1 的概率向量 |

`max` 是 PDF 截尾的硬上限。默认 `max = max(loc*3, loc + scale*3)`，对长尾分布可能偏小，超长尾建议显式设。

## 输出 CSV

每行一次请求。多模态列 (`image_tokens` / `audio_tokens` / `video_tokens`) 是 **JSON 数组**形式（如 `"[712, 689]"`），下游可直接 `json.loads`。

```csv
request_id,timestamp,text_tokens,output_tokens,image_tokens,audio_tokens,video_tokens
0,0.055,955,52,[630],"[247]","[2142]"
1,0.080,999,68,[736],[],[]
```
