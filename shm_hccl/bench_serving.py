"""End-to-end TTFT / decode-TPS sweep against the live vLLM endpoint.

Prompts are built from a per-request seeded random word stream, so no two
requests share a prefix block and --enable-prefix-caching never short-circuits
the prefill we are trying to measure.  Length is dialled in against /tokenize
before the request goes out, and the reported figure is the prompt_tokens the
server itself counted.

Usage: python3 bench_serving.py single|concurrent
"""
import json
import random
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "Qwen/Qwen3.8-27B"
OUT_TOKENS = 200

WORDS = """time year people way day man thing woman life child world school state
family student group country problem hand part place case week company system
program question work government number night point home water room mother area
money story fact month lot right study book eye job word business issue side
kind head house service friend father power hour game line end member law car
city community name president team minute idea kid body information back parent
face others level office door health person art war history party result change
morning reason research girl guy moment air teacher force education""".split()


def post(path, payload, timeout=1800):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def ntok(text):
    with post("/tokenize", {"model": MODEL, "prompt": text}) as r:
        return json.load(r)["count"]


def make_prompt(target, seed):
    """Random words, count tuned by secant steps on the measured token count."""
    rng = random.Random(seed)
    nwords = int(target / 1.35)
    for _ in range(6):
        text = " ".join(rng.choices(WORDS, k=nwords))
        got = ntok(text)
        if abs(got - target) <= max(8, target * 0.01):
            return text
        nwords = max(1, int(nwords * target / got))
        rng = random.Random(seed)
    return text


def run_one(prompt):
    """Returns (ttft_s, decode_tps, prompt_tokens, out_tokens)."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": OUT_TOKENS,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    n = 0
    ptok = None
    otok = None
    with post("/v1/completions", payload) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            chunk = json.loads(body)
            if chunk.get("usage"):
                ptok = chunk["usage"]["prompt_tokens"]
                otok = chunk["usage"]["completion_tokens"]
            ch = chunk.get("choices") or []
            if ch and ch[0].get("text"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                n += 1
    ttft = t_first - t0
    # Speculative decoding delivers a whole accepted burst in one SSE chunk, so
    # the chunk count is ~2.3x below the token count.  Rate must come from the
    # server's own completion_tokens, never from len(chunks).
    otok = otok or n
    tps = (otok - 1) / (t_last - t_first) if t_last > t_first else 0.0
    return ttft, tps, ptok, otok, n


# Appended so the model answers a real question instead of continuing word
# soup; the filler sets the context length, this sets the output's coherence,
# which is what the MTP draft head's acceptance rate actually depends on.
TASK = "\n\n以上是背景材料。请用中文解释什么是动态规划，并给出一个具体的例子。\n\n答："


def single():
    print(f"{'ctx(req)':>9} {'ctx(实际)':>10} {'trial':>6} {'TTFT s':>9} {'TPS':>8} {'tok/chunk':>10} {'out':>5}", flush=True)
    for target in (4096, 32768, 65536, 102400):
        for trial in (1,):
            prompt = make_prompt(target, seed=hash((target, trial, time.time())) & 0xFFFF) + TASK
            ttft, tps, ptok, otok, nch = run_one(prompt)
            print(f"{target:>9} {ptok:>10} {trial:>6} {ttft:>9.2f} {tps:>8.2f} {otok / nch:>10.2f} {otok:>5}", flush=True)


def concurrent():
    import threading
    target = 4096
    print(f"\n并发聚合吞吐 @ ctx≈{target}", flush=True)
    print(f"{'conc':>5} {'TTFT p50 s':>11} {'单路 TPS':>10} {'聚合 TPS':>10}", flush=True)
    for conc in (1, 4, 8, 16):
        prompts = [make_prompt(target, seed=1000 * conc + i) + TASK for i in range(conc)]
        res = [None] * conc
        def work(i):
            res[i] = run_one(prompts[i])
        th = [threading.Thread(target=work, args=(i,)) for i in range(conc)]
        t0 = time.perf_counter()
        for t in th:
            t.start()
        for t in th:
            t.join()
        wall = time.perf_counter() - t0
        ttfts = sorted(r[0] for r in res)
        per = sum(r[1] for r in res) / conc
        total_out = sum(r[3] for r in res)
        print(f"{conc:>5} {ttfts[conc // 2]:>11.2f} {per:>10.2f} {total_out / wall:>10.2f}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "single"
    if which in ("single", "both"):
        single()
    if which in ("concurrent", "both"):
        concurrent()
