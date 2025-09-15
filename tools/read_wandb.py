import re, json, pandas as pd, pathlib

# === 把这里改成你的 .wandb 文件路径 ===
wandb_file = "/home/fangqiang.d/MVGamba/results/mvgamba_man/wandb/latest-run/run-okxbbf4h.wandb"

bin_data = pathlib.Path(wandb_file).read_bytes()

# 方式：把二进制先按 latin-1 解码成“准文本”，以后再从中提取 JSON 片段
text = bin_data.decode("latin-1", errors="ignore")

# 正则策略：
# 1) 抓包含 "_step" 的 JSON 对象
# 2) 或抓包含 "Loss/" 的 JSON 对象
# 说明：JSON 对象在 .wandb 里通常是完整的一段 {...}，用非贪婪匹配尽量避免跨段
patterns = [
    r"\{[^{}]*\"_step\"[^{}]*\}",
    r"\{[^{}]*\"Loss\/[^\"]+\"[^{}]*\}"
]

candidates = []
for pat in patterns:
    for m in re.finditer(pat, text):
        candidates.append(m.group(0))

# 去重（按内容）
candidates = list(dict.fromkeys(candidates))

rows = []
for s in candidates:
    # 某些片段里可能混入奇怪字符，尝试简单清理
    # 去掉控制字符
    cleaned = "".join(ch for ch in s if ch == '\t' or ch == '\n' or 32 <= ord(ch) <= 126)
    # 尝试 json 解析
    try:
        obj = json.loads(cleaned)
        # 过滤：必须至少有 _step 或者某个 Loss 字段
        if ("_step" in obj) or any(k.startswith("Loss/") for k in obj.keys()):
            rows.append(obj)
    except json.JSONDecodeError:
        continue

# 如果有 _step，按 step 排序；否则按出现顺序
def step_of(o):
    v = o.get("_step", None)
    try:
        return int(v)
    except Exception:
        return 10**12  # 无 step 的排后

rows.sort(key=step_of)

# 输出 jsonl
out_jsonl = str(pathlib.Path(wandb_file).with_suffix("")) + ".recovered.jsonl"
with open(out_jsonl, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# 也做一个 CSV，常见列优先
priority_cols = ["_step", "_runtime", "_timestamp",
                 "Loss/train_total", "Loss/train_cd", "Loss/train_rcs", "Loss/train_vrel",
                 "Loss/eval_total", "Loss/eval_cd", "Loss/eval_rcs", "Loss/eval_vrel",
                 "LR/lr"]
# 收集所有出现过的列
all_cols = set()
for r in rows: all_cols.update(r.keys())
# 列顺序：优先列 + 其它列
ordered_cols = priority_cols + sorted([c for c in all_cols if c not in priority_cols])

df = pd.DataFrame(rows)[ordered_cols]
out_csv = str(pathlib.Path(wandb_file).with_suffix("")) + ".history.csv"
df.to_csv(out_csv, index=False)

print(f"[OK] 提取 {len(rows)} 条记录")
print(f"JSONL: {out_jsonl}")
print(f"CSV  : {out_csv}")
try:
    # 打印几行看看
    print(df.head(10).to_string(index=False))
except Exception:
    pass
