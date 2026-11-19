
import os
import re
import json
import csv
import time
import math
import torch
import tiktoken

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from model import GPT, GPTConfig
from analyze_generation import calculate_metrics


# ========== 1. 配置 ==========

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("请配置 DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com/v1",
    timeout=60
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = ROOT / "outputs/gpt2-medium/ckpt_final_1.pt"

OUTPUT_DIR = ROOT / "experiment_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "log.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "summary.csv"

MAX_NEW_TOKENS = 300
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.95

NUM_SAMPLES = 3

AVG_THRESHOLD = 7
RELEVANCE_THRESHOLD = 7
COMPLETENESS_THRESHOLD = 6

SCORE_KEYS = [
    "relevance",
    "coherence",
    "completeness",
    "language",
    "diversity"
]

TEST_TOPICS = [
    "a little girl found a magic key",
    "a brave dog saved his friend",
    "a rabbit lost its way home",
    "a boy planted a magic seed",
    "a little bird learned to fly",
    "a cat helped a lonely kitten",
    "a bear shared his food",
    "a girl discovered a hidden garden",
    "a puppy helped an old man",
    "a monkey learned to be patient",
    "a little fish escaped a big net",
    "a boy found a lost treasure",
    "a butterfly helped a flower grow",
    "a squirrel prepared for winter",
    "a dragon became friends with a child",
    "a girl learned to ride a bicycle",
    "a mouse helped a lion",
    "a boy rescued a kitten from a tree",
    "a little elephant overcame its fear",
    "two friends built a treehouse"
]

STATS = {
    "total": 0,
    "local_accepted": 0,
    "fallback": 0,
    "failed": 0,
    "judge_errors": 0,
    "api_calls": 0,
    "local_time": 0.0,
    "fallback_time": 0.0,
    "final_scores": []
}


# ========== 2. 加载模型 ==========

def load_model():

    ckpt = torch.load(
        CKPT_PATH,
        map_location="cpu",
        weights_only=False
    )

    model = GPT(GPTConfig(**ckpt["config"]["model"]))

    state = ckpt["model"]

    if any(k.startswith("module.") for k in state):
        state = {
            k.replace("module.", "", 1): v
            for k, v in state.items()
        }

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    print(f"GPT-2 loaded, step={ckpt.get('step')}")
    print("Device:", DEVICE)

    return model


# ========== 3. 本地故事生成 ==========

@torch.no_grad()
def generate_story_local(model, enc, request):

    prompt = f"Once upon a time, {request}. "
    tokens = enc.encode(prompt)

    idx = torch.tensor(
        tokens,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    start = time.perf_counter()

    out = model.generate(
        idx,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        eos_token_id=enc.eot_token
    )

    elapsed = time.perf_counter() - start

    new_tokens = out[0, len(tokens):].tolist()

    if enc.eot_token in new_tokens:
        new_tokens = new_tokens[:new_tokens.index(enc.eot_token)]

    return enc.decode(new_tokens).strip(), elapsed


# ========== 4. 故事质量评价 ==========

def evaluate_story(request, story):

    if not story.strip():
        raise ValueError("Empty story")

    metrics = calculate_metrics(story)

    prompt = f"""
Evaluate this children's story from 0 to 10
on five dimensions:

1. relevance: Does it follow the requested topic?
2. coherence: Are events logically connected?
3. completeness: Does it have a beginning, middle and ending?
4. language: Is the English natural and grammatical?
5. diversity: Is the expression sufficiently varied?

Scoring:
9-10 excellent, 7-8 good, 5-6 fair,
3-4 poor, 0-2 very poor.

Evaluate the actual story, not keyword matching.
Apply the same standards to every story.
Do not penalize a short story solely for its length.

User request:
{request}

Story:
{story}

Objective metrics:
{json.dumps(metrics)}

Return JSON only:
{{
"relevance": 0,
"coherence": 0,
"completeness": 0,
"language": 0,
"diversity": 0,
"reason": "brief explanation"
}}
"""

    STATS["api_calls"] += 1

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content
    match = re.search(r"\{[\s\S]*\}", content)

    if not match:
        raise ValueError("Invalid evaluation JSON")

    result = json.loads(match.group())

    for key in SCORE_KEYS:
        score = result.get(key)

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"Invalid score: {key}")

        if not math.isfinite(float(score)) or not 0 <= score <= 10:
            raise ValueError(f"Score out of range: {key}")

    average = sum(float(result[k]) for k in SCORE_KEYS) / 5

    result["average"] = round(average, 2)

    result["passed"] = (
        average >= AVG_THRESHOLD
        and result["relevance"] >= RELEVANCE_THRESHOLD
        and result["completeness"] >= COMPLETENESS_THRESHOLD
    )

    result["objective_metrics"] = metrics

    return result


# ========== 5. DeepSeek 兜底生成 ==========

def generate_story_deepseek(request):

    start = time.perf_counter()

    STATS["api_calls"] += 1

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional children's story writer. "
                    "Write a complete, coherent English story. "
                    "Follow the requested topic closely. "
                    "Include a clear beginning, middle and ending. "
                    "Use natural language and avoid repetition. "
                    "Write approximately 250-400 words."
                )
            },
            {
                "role": "user",
                "content": f"Write a story about: {request}"
            }
        ],
        temperature=0.7
    )

    return (
        response.choices[0].message.content.strip(),
        time.perf_counter() - start
    )


# ========== 6. Agent 路由 ==========

def hybrid_agent(request, model, enc, topic_id, sample_id):

    STATS["total"] += 1

    print(f"\n[{topic_id} | Sample {sample_id}]")
    print("Request:", request)

    # 本地生成
    try:
        local_story, local_time = generate_story_local(
            model, enc, request
        )
        STATS["local_time"] += local_time

    except Exception as e:
        print("Local generation failed:", e)
        local_story = ""
        local_time = 0

    print("\nLocal Story:\n", local_story)
    print(f"Local time: {local_time:.2f}s")

    # 本地质量评价
    local_score = None

    try:
        local_score = evaluate_story(request, local_story)

        print("\nLocal scores:")
        for key in SCORE_KEYS:
            print(f"{key}: {local_score[key]}/10")

        print("Average:", local_score["average"])
        print("Passed:", local_score["passed"])

    except Exception as e:
        STATS["judge_errors"] += 1
        print("Evaluation failed:", e)

    # 路由决策
    if local_score and local_score["passed"]:

        final_story = local_story
        final_score = local_score
        source = "local"
        fallback_time = 0

        STATS["local_accepted"] += 1

    else:

        print("\nLocal rejected. Using DeepSeek...")

        STATS["fallback"] += 1

        try:
            final_story, fallback_time = generate_story_deepseek(
                request
            )

            STATS["fallback_time"] += fallback_time

            print("\nDeepSeek Story:\n", final_story)

            # 评价最终故事
            final_score = evaluate_story(request, final_story)

        except Exception as e:
            print("DeepSeek failed:", e)

            final_story = ""
            final_score = None
            fallback_time = 0

            STATS["failed"] += 1
            source = "failed"

        else:
            source = "deepseek"

    if final_score is not None:
        STATS["final_scores"].append(final_score)

    print("\nFinal Source:", source)

    # 保存记录
    record = {
        "topic_id": topic_id,
        "sample_id": sample_id,
        "request": request,
        "local_story": local_story,
        "local_time": round(local_time, 3),
        "local_scores": local_score,
        "final_story": final_story,
        "final_source": source,
        "fallback_time": round(fallback_time, 3),
        "final_scores": final_score,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


# ========== 7. 批量实验与统计 ==========

def run_experiment(model, enc):

    records = []

    if LOG_FILE.exists():
        LOG_FILE.unlink()

    for i, request in enumerate(TEST_TOPICS, 1):

        for sample in range(1, NUM_SAMPLES + 1):

            try:
                record = hybrid_agent(
                    request,
                    model,
                    enc,
                    f"T{i:02d}",
                    sample
                )
                records.append(record)

            except Exception as e:
                print("Task error:", e)

    return records


def save_summary():

    total = STATS["total"]

    if total == 0:
        print("No tasks completed.")
        return

    accepted = STATS["local_accepted"]
    fallback = STATS["fallback"]

    scores = STATS["final_scores"]

    summary = {
        "total_tasks": total,
        "local_accepted": accepted,
        "deepseek_fallback": fallback,
        "failed": STATS["failed"],
        "local_acceptance_rate": accepted / total,
        "fallback_rate": fallback / total,
        "avg_local_time": STATS["local_time"] / total,
        "avg_fallback_time": (
            STATS["fallback_time"] / fallback if fallback else 0
        ),
        "avg_final_quality": (
            sum(s["average"] for s in scores) / len(scores)
            if scores else 0
        ),
        "judge_errors": STATS["judge_errors"],
        "api_calls": STATS["api_calls"]
    }

    for key in SCORE_KEYS:
        values = [s[key] for s in scores]
        summary[key] = (
            sum(values) / len(values) if values else 0
        )

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary.keys())
        )
        writer.writeheader()
        writer.writerow(summary)

    print("\n========== Experiment Summary ==========")

    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    print("\nResults saved to:", OUTPUT_DIR)


# ========== 8. 主程序 ==========

if __name__ == "__main__":

    model = load_model()
    enc = tiktoken.get_encoding("gpt2")

    print("\nTopics:", len(TEST_TOPICS))
    print("Samples per topic:", NUM_SAMPLES)
    print("Total tasks:", len(TEST_TOPICS) * NUM_SAMPLES)

    run_experiment(model, enc)
    save_summary()

    print("\nExperiment completed!")
