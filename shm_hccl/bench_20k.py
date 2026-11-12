import json, random, sys, time, urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen3.8-27B"
OUT_TOKENS = 250

SENTENCES = [
    "在分布式推理系统中，跨PCIe根复合体的集合通信延迟往往由主机侧同步屏障、DMA页锁定状态以及NUMA跨片一致性流量共同决定。",
    "对于混合线性注意力（Gated DeltaNet）与全注意力（GQA）的64层架构，长序列解码阶段的KV缓存读取只发生在16层全注意力层。",
    "多步投机解码（MTP）通过单次主干前向同时验证多个候选词元，将算术强度从单步1词元提升至4至5词元，从而跨越访存带宽瓶颈。",
    "在高并发支付清算链路中，状态机流转必须保证幂等性与单调性，严禁在数据库事务内部嵌套外部RPC调用以防连接池耗尽。",
    "Ascend 310P处理器采用DaVinci V200架构，配备LPDDR4X显存与分形NZ矩阵布局，适合通过INT8量化充分发挥Cube单元吞吐。",
    "When capturing an ACL graph for speculative decoding, tensor addresses and query start locations must remain static across replays.",
    "Lock-free ring buffers aligned to 128-byte Kunpeng-920 L3 cache lines prevent false sharing between tensor-parallel worker threads.",
    "Prefix caching hashes fixed-size KV blocks with SHA-256 so multi-turn agent sessions reuse historical context without recomputing prefill.",
]

def post(path, payload, timeout=900):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)

def ntok(text):
    with post("/tokenize", {"model": MODEL, "prompt": text}) as r:
        return json.load(r)["count"]

def build_doc(target_tokens, seed):
    rng = random.Random(seed)
    blocks = []
    # Unique header block so uncached seeds never share even block 0 (512 tokens)
    blocks.append(f"[Session-{seed}-{rng.randint(100000, 999999)}] 系统技术架构与性能分析档案：\n")
    est_toks = 25
    while est_toks < target_tokens * 1.15:
        s = rng.choice(SENTENCES)
        tag = rng.randint(10, 9999)
        line = f"第{tag}节：{s}\n"
        blocks.append(line)
        est_toks += 42
    full = "".join(blocks)
    # Binary search character slice to hit target_tokens within +-16 tokens
    lo, hi = int(len(full) * 0.4), len(full)
    best = full[: (lo + hi) // 2]
    for _ in range(8):
        mid = (lo + hi) // 2
        cand = full[:mid]
        c = ntok(cand)
        best = cand
        if abs(c - target_tokens) <= 16:
            break
        if c < target_tokens:
            lo = mid
        else:
            hi = mid
    return best

QUESTIONS = [
    "\n\n请基于以上技术材料，分五点深入总结长上下文大模型推理优化的核心原理，并推导吞吐量与延迟的定量关系：\n答：",
    "\n\n请结合上述文档内容，详细阐述分布式系统与AI推理引擎在内存管理、并发控制和延迟优化上的共通工程模式，并举例说明：\n答：",
]

def get_metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        txt = r.read().decode("utf-8")
    out = {}
    for line in txt.splitlines():
        if line.startswith("vllm:spec_decode_num_"):
            k, v = line.split("} ")
            out[k + "}"] = float(v)
    return out

def run_chat(prompt, temp, max_tokens=OUT_TOKENS):
    m0 = get_metrics()
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_k": 20 if temp > 0 else -1,
        "top_p": 0.95 if temp > 0 else 1.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    tf = tl = None
    chunks = ptok = otok = 0
    with post("/v1/completions", payload) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line[6:] == "[DONE]":
                continue
            obj = json.loads(line[6:])
            if obj.get("usage"):
                ptok = obj["usage"]["prompt_tokens"]
                otok = obj["usage"]["completion_tokens"]
            ch = obj.get("choices") or []
            if ch and ch[0].get("text"):
                now = time.perf_counter()
                if tf is None:
                    tf = now
                tl = now
                chunks += 1
    m1 = get_metrics()
    drafts = 0.0
    accepted = 0.0
    for k in m1:
        if "spec_decode_num_drafts_total" in k:
            drafts += m1[k] - m0.get(k, 0.0)
        elif "spec_decode_num_accepted_tokens_total" in k:
            accepted += m1[k] - m0.get(k, 0.0)
    ttft = tf - t0
    dec_s = tl - tf if (tl and tf and tl > tf) else 1e-6
    tps = (otok - 1) / dec_s
    tpot_ms = dec_s * 1000.0 / max(1, otok - 1)
    ms_step = dec_s * 1000.0 / max(1.0, drafts) if drafts > 0 else 0.0
    tok_step = 1.0 + (accepted / drafts) if drafts > 0 else (otok / max(1, chunks))
    return {
        "ptok": ptok, "otok": otok, "ttft": ttft,
        "pf_tps": ptok / ttft, "tps": tps, "tpot_ms": tpot_ms,
        "ms_step": ms_step, "tok_step": tok_step, "drafts": int(drafts),
    }

def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    # Warmup
    _ = run_chat("请简要介绍动态规划：\n答：", 0.0, max_tokens=32)

    print(f"=== 20K BENCHMARK [{tag}] (out={OUT_TOKENS}) ===", flush=True)
    hdr = f"{'Case':<26} {'Prompt':>7} {'TTFT(s)':>8} {'Prefill':>9} {'Decode(t/s)':>12} {'TPOT(ms)':>9} {'ms/step':>8} {'tok/step':>9}"
    print(hdr, flush=True)

    summary = {}
    # 1. Uncached 20K (Greedy & Default t=0.3)
    for label, temp, base_seed in [
        ("20K-Uncached-Greedy(t=0)", 0.0, 202611),
        ("20K-Uncached-Default(0.3)", 0.3, 202621),
    ]:
        vals = []
        for tr in range(2):
            doc = build_doc(20400, seed=base_seed + tr * 17 + int(time.time() * 10) % 997)
            r = run_chat(doc + QUESTIONS[tr % 2], temp)
            vals.append(r)
            print(f"{label+'#'+str(tr+1):<26} {r['ptok']:>7} {r['ttft']:>8.2f} {r['pf_tps']:>9.1f} {r['tps']:>12.2f} {r['tpot_ms']:>9.2f} {r['ms_step']:>8.1f} {r['tok_step']:>9.2f}", flush=True)
        summary[label] = vals

    # 2. Cached 20K prefix (populate prefix once, then measure 2 cached turns for t=0 and t=0.3)
    shared_doc = build_doc(20400, seed=888800 + int(time.time()) % 10000)
    _ = run_chat(shared_doc + "\n\n预热问题：请用一句话概括主题。\n答：", 0.0, max_tokens=16)
    for label, temp in [
        ("20K-Cached-Greedy(t=0)", 0.0),
        ("20K-Cached-Default(0.3)", 0.3),
    ]:
        vals = []
        for tr in range(2):
            q = f"\n\n[追问-{temp}-{tr}]" + QUESTIONS[tr % 2]
            r = run_chat(shared_doc + q, temp)
            vals.append(r)
            print(f"{label+'#'+str(tr+1):<26} {r['ptok']:>7} {r['ttft']:>8.2f} {r['pf_tps']:>9.1f} {r['tps']:>12.2f} {r['tpot_ms']:>9.2f} {r['ms_step']:>8.1f} {r['tok_step']:>9.2f}", flush=True)
        summary[label] = vals

    print("\n--- AVERAGES ---", flush=True)
    for k, vals in summary.items():
        avg_ttft = sum(x["ttft"] for x in vals) / len(vals)
        avg_tps = sum(x["tps"] for x in vals) / len(vals)
        avg_tpot = sum(x["tpot_ms"] for x in vals) / len(vals)
        avg_step = sum(x["ms_step"] for x in vals) / len(vals)
        avg_tok = sum(x["tok_step"] for x in vals) / len(vals)
        print(f"{k:<26} | TTFT={avg_ttft:6.2f}s | Decode={avg_tps:6.2f} tok/s | TPOT={avg_tpot:5.2f} ms | Step={avg_step:5.1f} ms | Acc={avg_tok:4.2f} tok/step", flush=True)

if __name__ == "__main__":
    main()
