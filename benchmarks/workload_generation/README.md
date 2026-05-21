# workload_generation

构造**多模态推理服务的实验负载**：从分布参数 / 真实迹线生成请求时间戳与各模态 token 数，再反推为 vllm-omni `random-mm` 风格的 `(H, W, T)` / `duration` 输入，供 phase 1（无前缀缓存）压测用。


## 目录结构

```
workload_generation/
├── README.md                   # 本文档
├── generate_experiment.md      # generate_experiment.py 使用说明（详细参数表）
├── build_inputs.md             # build_inputs.py 使用说明 + 反推公式
├── generate_experiment.py      # 阶段 1：生成 CSV（含 token 数与到达时间）
├── token_shape.py              # 纯函数：token ↔ (H, W, T) / duration
├── build_inputs.py             # 阶段 2：CSV → JSONL（带可发请求的形状信息）
├── configs/
│   ├── qwen3_omni.json         # Qwen3-Omni-30B-A3B 的 vision/audio 参数
│   └── scenario_example.json   # 多 window 非平稳负载示例
└── workload/                   # 生成的 CSV / JSONL
```

## 流程总览

```
┌──────────────────────────┐    ┌──────────────────────────┐    ┌────────────────────────┐
│ generate_experiment.py   │ →  │ workload/*.csv           │ →  │ build_inputs.py        │
│  - ServeGen + PDF        │    │  request_id, timestamp,  │    │  + configs/qwen3_omni  │
│  - 多 window / 多客户端  │    │  text_tokens,            │    │                        │
│  - CLI 或 JSON 配置      │    │  output_tokens,          │    │  → workload/*.jsonl    │
└──────────────────────────┘    │  [image_tokens, ...]     │    │  每行：{ts, mm_items   │
                                └──────────────────────────┘    │   含 (H, W, T) /       │
                                                                │   duration_s }         │
                                                                └────────────────────────┘
                                                                          ↓
                                                          下游：注入 vllm-omni（后续做）
```

## 一分钟上手

```bash
# 1. 生成 120 秒、20 req/s 的多模态负载（1 张图，~700 token/图）
python generate_experiment.py -c multimodal \
    --image-count 1 --image-tokens 700 \
    --rate 20 --duration 120 \
    --output workload/mm_120_20.csv

# 2. 反推 (H, W, T) / duration，输出每请求一行的 JSONL
python build_inputs.py \
    --csv workload/mm_120_20.csv \
    --model-config configs/qwen3_omni.json \
    --output workload/mm_120_20.jsonl \
    --verify-tokens
```

`--verify-tokens` 会打印 actual 与 target 的偏差统计（一般 image |err| < 30，video |err| < 100，audio 完全无误差）。

## 阶段 1：`generate_experiment.py`

两种驱动方式：

- **CLI 简单模式**：单客户端，分布只一组，速率可分段。
- **JSON 配置高级模式**：多客户端 + 多 window 不同分布。

```bash
# CLI：分段速率，0–30s 是 10 req/s，30–60s 是 40 req/s
python generate_experiment.py -c multimodal \
    --windows "0,30" --rate "10,40" --duration 60 \
    --image-count 1 --image-tokens 700

# JSON：完整声明每个 window 的 PDF 与到达模式
python generate_experiment.py --config configs/scenario_example.json
```

CSV 输出列：

- 通用：`request_id, timestamp, output_tokens`
- LANGUAGE / REASON：`input_tokens`（REASON 多一列 `reason_ratio`）
- MULTIMODAL：`text_tokens, image_tokens, audio_tokens, video_tokens`（后三列是 JSON 数组，长度 = 该请求该模态的 item 数）

详见 [generate_experiment.md](generate_experiment.md)。

## 阶段 2：`build_inputs.py`

读 CSV，按 `configs/qwen3_omni.json` 反推每个 mm item 的形状。输出 JSONL，每行一个请求：

```jsonc
{
  "request_id": 0,
  "timestamp": 0.055,
  "text_tokens": 955,
  "output_tokens": 52,
  "mm_items": [
    {"modality": "image", "h": 800, "w": 800,  "t": 1,   "target_tokens": 630,  "actual_tokens": 625},
    {"modality": "video", "h": 672, "w": 672,  "t": 150, "target_tokens": 2142, "actual_tokens": 2205},
    {"modality": "audio", "duration_s": 9.88,            "target_tokens": 247,  "actual_tokens": 247}
  ]
}
```

`(h, w, t)` 与 vLLM `RandomMultiModalDataset` 的 bucket key `(height, width, num_frames)` 一致；`t=1` 表示图，`t>1` 表示视频；audio 用 `duration_s`。

详见 [build_inputs.md](build_inputs.md)。

## 关于精度

反推 token 数有量化误差，因为：

- 图像必须按 `patch_size × spatial_merge_size = 32` 像素对齐（Qwen3-Omni）。
- 视频必须按 `gen_fps / encoder_fps × temporal_patch_size = 30` 帧对齐。

因此 target=700 的图可能落到 832×832（=676 tokens, 误差 -24），target=2142 的视频可能落到 150 帧 × 672×672（=2205 tokens, 误差 +63）。这个误差在 phase 1 压测里完全可以接受——如果不能接受，按"先采样形状再算 token 数"的正向流程（即 omni 论文那种 random-mm）会更准确，但需要重写 PDF 那一层。

## 切换模型

修改或新增 `configs/<model>.json`，关键字段：

| 字段 | Qwen3-Omni | 说明 |
|------|------------|------|
| `patch_size` | 16 | 视觉 patch 像素数 |
| `spatial_merge_size` | 2 | 空间合并因子 |
| `temporal_patch_size` | 2 | 时间合并因子（帧成对） |
| `encoder_fps` | 2.0 | 处理器把视频重采样到的 fps |
| `gen_fps` | 30.0 | 合成视频 MP4 的 fps（vLLM 写死 30） |
| `audio_tokens_per_second` | 25.0 | Whisper-style encoder 输出率 |

Qwen2.5-VL/Omni 的差异：`patch_size=14`，`encoder_fps` 仍是 2。
