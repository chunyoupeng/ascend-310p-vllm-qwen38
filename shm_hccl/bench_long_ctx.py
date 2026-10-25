import json
import time
import urllib.request

API_URL = 'http://127.0.0.1:8000/v1/chat/completions'
TOKENIZE_URL = 'http://127.0.0.1:8000/tokenize'
MODEL = 'Qwen/Qwen3.8-27B'

# Construct a realistic varied technical text block and measure its token count
unit_text = (
    'In distributed deep learning systems on Ascend NPU architectures, optimizing collective communication '
    'and memory hierarchy utilization is critical for low-latency autoregressive decoding. '
    'By leveraging shared memory ring buffers, cache-line aligned atomic synchronization, and SIMD vectorization, '
    'tensor parallel workers can minimize synchronization overhead across PCIe root complexes.\n'
)

req = urllib.request.Request(
    TOKENIZE_URL,
    data=json.dumps({'model': MODEL, 'prompt': unit_text}).encode('utf-8'),
    headers={'Content-Type': 'application/json'},
)
with urllib.request.urlopen(req, timeout=30) as resp:
    unit_tokens = len(json.loads(resp.read().decode('utf-8'))['tokens'])

print(f'Unit text tokens: {unit_tokens}', flush=True)
print(f'{"Target":<8} | {"PromptTok":<10} | {"Prefill(s)":<11} | {"Prefill(tok/s)":<15} | {"DecodeTok":<10} | {"Decode(s)":<10} | {"Decode(tok/s)":<14} | {"TPOT(ms/tok)":<12}', flush=True)
print('-' * 110, flush=True)

targets = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

for target in targets:
    repeats = max(1, (target - 60) // unit_tokens)
    context_body = unit_text * repeats
    prompt = f'{context_body}\n\n请基于以上技术背景，详细总结分布式推理性能优化的五个关键要点（编号1到5列出）：'
    payload = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 100,
        'temperature': 0.0,
        'stream': True,
        'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'enable_thinking': False},
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    t_start = time.perf_counter()
    t_first = None
    t_last = None
    prompt_tokens = 0
    completion_tokens = 0
    chunks = 0
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw_line in resp:
            line = raw_line.decode('utf-8').strip()
            if not line.startswith('data: '):
                continue
            data_str = line[6:]
            if data_str == '[DONE]':
                break
            obj = json.loads(data_str)
            if obj.get('usage'):
                prompt_tokens = obj['usage'].get('prompt_tokens', prompt_tokens)
                completion_tokens = obj['usage'].get('completion_tokens', completion_tokens)
            choices = obj.get('choices') or []
            if choices:
                delta = choices[0].get('delta') or {}
                content = delta.get('content') or delta.get('reasoning') or ''
                if content:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    chunks += 1

    ttft = (t_first - t_start) if t_first else 0.0
    decode_time = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0.001
    prefill_tps = prompt_tokens / ttft if ttft > 0 else 0.0
    dec_tokens = max(1, completion_tokens - 1)
    decode_tps = dec_tokens / decode_time if decode_time > 0 else 0.0
    tpot_ms = (decode_time * 1000.0) / dec_tokens if dec_tokens > 0 else 0.0
    label = f'{target // 1000}K'
    print(f'{label:<8} | {prompt_tokens:<10} | {ttft:<11.2f} | {prefill_tps:<15.1f} | {completion_tokens:<10} | {decode_time:<10.2f} | {decode_tps:<14.2f} | {tpot_ms:<12.2f}', flush=True)
