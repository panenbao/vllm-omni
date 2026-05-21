# build_inputs.py 使用说明

阶段 2 工具：读 CSV（含每请求各模态的目标 token 数）→ 反推为 vllm-omni `random-mm` 风格的 `(H, W, T)` / `duration_s` 形状信息，输出 JSONL。

## 用法

```bash
python build_inputs.py \
    --csv workload/mm_120_20.csv \
    --model-config configs/qwen3_omni.json \
    --output workload/mm_120_20.jsonl \
    --verify-tokens
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--csv` | 必填 | 输入 CSV 路径 |
| `--model-config` | `configs/qwen3_omni.json` | 模型 token 配置 |
| `-o, --output` | 同名 `.jsonl` | 输出 JSONL 路径 |
| `--video-num-frames` | 自适应 | 固定所有视频的合成帧数（按 `gen_fps`） |
| `--image-aspect` | `1.0` | 图片 W/H 比 |
| `--video-aspect` | `1.0` | 视频 W/H 比 |
| `--verify-tokens` | 关 | 打印 actual − target 的偏差统计 |

## 输出 schema

```jsonc
{
  "request_id": 0,
  "timestamp": 0.055,            // 秒，相对工作负载起点
  "text_tokens": 955,            // multimodal：文本部分 token 数
  // "input_tokens": 1000,       // language/reason 用这个
  "output_tokens": 52,
  // "reason_ratio": 0.5,        // 仅 reason
  "mm_items": [                  // 可省（无多模态项时不写）
    {
      "modality": "image",
      "h": 800, "w": 800, "t": 1,
      "target_tokens": 630,      // CSV 里的目标值
      "actual_tokens": 625       // 按 (h,w,t) 实际产生的 token 数
    },
    {
      "modality": "video",
      "h": 672, "w": 672, "t": 150,
      "fps": 30.0,               // 与 vllm-omni 合成 MP4 的 fps 一致
      "duration_s": 5.0,         // = t / fps，便于人工检查
      "target_tokens": 2142,
      "actual_tokens": 2205
    },
    {
      "modality": "audio",
      "duration_s": 9.88,
      "num_channels": 1,         // 1=mono, 2=stereo, 5=5.1；缺省取 cfg.audio_default_channels
      "target_tokens": 247,
      "actual_tokens": 247
    }
  ]
}
```

`(h, w, t)` 与 vLLM `RandomMultiModalDataset` 的 bucket key `(height, width, num_frames)` 完全兼容；`t=1` → 图，`t>1` → 视频。video 的 `t` 是 **vllm-omni 合成 MP4 时按 `fps` (默认 30) 写入的帧数**，`duration_s = t / fps` 给出可视秒数。`audio` 不走 `t`，直接给 `duration_s` 和 `num_channels`（合成时 `int(duration_s * sample_rate)` 个采样点 × 通道数）。

> **关于"图片张数"**：CSV 一行的 `image_tokens` 列表长度是 N 时，输出会出现 N 个 `modality: "image"` 条目，下游 replay 时把同一 `request_id` 的所有 image 条目放进同一条 chat message 的 `content` 数组（OpenAI Vision API 支持多张 `image_url`）。同理适用于 audio/video。

> **关于音频通道**：CSV 中的 `audio_channels` 列由 [generate_experiment.py](generate_experiment.py) 在 CSV 写入阶段按 `--audio-channels` / `--audio-channels-probs` 采样，长度与 `audio_tokens` 一一对应。如果该列缺失，本脚本回退到 `cfg.audio_default_channels`。

## 反推公式（Qwen3-Omni）

记 `p = patch_size`、`m = spatial_merge_size`、`tp = temporal_patch_size`。Qwen3-Omni-30B 取值 `p=16, m=2, tp=2`。

### 图像

```
tokens = (H / p) * (W / p) / m^2
```

`H, W` 必须是 `p * m = 32` 的整数倍。

反推（取正方形）：

```
side_units = sqrt(tokens)            # 单位是 (p*m) 块
H = W = round(side_units) * (p * m)
```

### 视频

`encoder_fps`（处理器重采样到的 fps）和 `gen_fps`（合成 MP4 的 fps，vLLM 写死 30）共同决定一帧合成视频贡献多少个 token 槽：

```
slices_per_gen_frame = encoder_fps / (gen_fps * tp)
                     = 2 / (30 * 2)  = 1/30   (Qwen3-Omni)
```

正向：

```
tokens = floor(T * slices_per_gen_frame) * (H/p) * (W/p) / m^2
```

反推（给定目标 token 数与目标帧数 T）：

```
slices = floor(T * slices_per_gen_frame)
tokens_per_slice = tokens / slices
(H, W) ← image_tokens_to_hw(tokens_per_slice)
```

如果不指定 T，默认按"每 slice 约 512 token"挑选最小的 T。

### 音频

Qwen2-Audio / Qwen3-Omni 的 audio encoder（Whisper-style + 一次 conv 下采样）：

```
mel_frames = duration_s * sample_rate / hop_length = duration_s * 100
feat_lens  = (mel_frames - 1) // 2 + 1
tokens     = (feat_lens - 2) // 2 + 1
```

线性近似 `tokens ≈ duration_s * 25`（误差小于 ±1）。直接反推：

```
duration_s = tokens / 25
```

## 精度

`build_inputs.py --verify-tokens` 会打印偏差统计。我跑 50 条三模态混合的样本结果：

| 模态 | mean(err) | max\|err\| | 备注 |
|------|-----------|-----------|------|
| image | +0.6 | 27 | 非完美平方时按 32px 向下取整 |
| video | +1.6 | 101 | 帧数 / 块尺寸量化叠加 |
| audio | 0 | 0 | 25 tok/s 整除关系 |

如果对精度敏感（比如要做某个特定 token 数的回归测试），可：

- 把目标 token 值直接选成 `n²`（图）或 `30k · n²`（视频），完美匹配。
- 用 `--image-aspect` / `--video-aspect` 偏离正方形以换取更细粒度。

## 接入 vllm-omni（后续）

这一步还没做。两条候选路径：

- **路径 A**：在 vllm-omni 加一个 `OmniTraceDataset`，从 JSONL 读 `mm_items`，复用 `OmniRandomMultiModalDataset.generate_synthetic_*`，对接到 `bench serve --dataset-name omni-trace`。可以复用 vllm-omni 已有的 percentile / 报表。
- **路径 B**：写一个独立的 `replay_workload.py`，按 `timestamp` 异步发请求到 vllm-omni 的 OpenAI 端点，自己测指标。

建议先做 A。
