1. 生成 ServeGen 式 trace，有两种方式
	1. 直接使用 ServeGen 提供的 examples 里来生成自带的突发负载
	2. 使用 benchmark/workload_generation/generate_experiment.py 来生成自定的负载，可参照generate_experiment. md 的说明
2. 生成可被 Benchmark 接受的数据集
	1. 使用 benchmark/workload_generation/build_input.py 来生成数据集，可参照build_inputs.md 的说明
3. 使用 benchmark 进行测试，示例代码如下，重点注意 `dataset-name` 与 `dataset-path` 即可。
```Bash
vllm bench serve \
  --omni \
  --dataset-name servegen \
  --dataset-path /home/codes/ServeGen/workload/MULTIMODAL_20_1_workload.jsonl \
  --port 8004 \
  --model /data/models/Qwen3-Omni-30B-A3B-Instruct \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --num-prompts 100 \
  --max-concurrency 100 \
  --ignore-eos \
  --random-output-len 100 \
  --extra_body '{"modalities": ["text"]}'
```