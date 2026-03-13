# Self-Healing Agent Framework

一個具備「反思」能力的自癒型 AI Agent。當工具執行失敗時，不是直接報錯停止，而是自動分析錯誤、批判自己的策略、提出改進方案，再重試。

---

## 核心架構

```
工具失敗 ❌
    │
    ├─► 記憶庫有此錯誤？ ──YES──► 直接套用過去成功策略（跳過 API 調用）
    │
    └─► NO
         │
         ├─► 第 1 次失敗 → Haiku 輕量反思
         ├─► 第 2 次失敗 → Opus 標準反思
         └─► 第 3 次失敗 → Opus + Adaptive Thinking 深度反思
                  │
                  ├─► confidence < 35？→ 放棄，不再重試
                  └─► 注入結構化反思 → Claude 調整策略 → 重試
                           │
                           └─► 成功 → 記錄策略到記憶庫
```

### 五個核心模組

| 模組 | 功能 |
|------|------|
| `agent/core.py` | 主循環：agentic loop + 反思注入 |
| `agent/reflection.py` | Reflector：結構化反思 + Confidence Gate + 漸進深度 |
| `agent/memory.py` | FailureMemory：持久化成功策略，避免重複反思 |
| `agent/logger.py` | RunLogger：每次執行產生 JSON log，支援統計報告 |
| `agent/sandbox.py` | Python 沙盒：AST 靜態分析 + resource 限制 |

---

## 快速開始

### 1. 安裝

```bash
git clone https://github.com/allen-0777/self_healing_agent
cd self_healing_agent

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 設定 API Key

```bash
cp .env.example .env
# 編輯 .env，填入 ANTHROPIC_API_KEY=sk-ant-...
```

### 3. 建立測試資料

```bash
python examples/setup_demo.py
```

### 4. 執行 Demo

```bash
# Demo 1：檔名錯誤自動修復
python main.py --demo file

# Demo 2：Python 程式碼自動修復（含沙盒）
python main.py --demo code

# Demo 3：記憶庫 hit 演示（跑兩次，第二次跳過反思）
python main.py --demo memory

# 全部執行
python main.py
```

---

## Demo 說明

### Demo 1 — 檔名錯誤修復

Agent 被要求讀取 `sales.csv`，但目錄裡只有 `revenue_2024.csv`：

```
[TOOL] read_file({"path": ".../sales.csv"})
  ❌ FAILED: FileNotFoundError: No such file or directory
  🔍 Reflecting (depth=1)...

  [REFLECTION]
  Root Cause: The file 'sales.csv' does not exist at the specified path.
  Self-Criticism: I assumed the filename without verifying what files exist.
  New Strategy:
    1. Call list_directory to see what files are actually present
    2. Read the correct file based on the listing

[TOOL] list_directory({"path": "..."})
  ✅ Success

[TOOL] read_file({"path": ".../revenue_2024.csv"})
  ✅ Success
```

### Demo 2 — Python 沙盒 + 程式碼修復

程式碼中若有 `import os` 等危險操作，沙盒會在執行前攔截：

```
[TOOL] execute_python({"code": "import os; ..."})
  ❌ FAILED: SandboxViolation: Blocked: import of 'os' is not allowed
  🔍 Reflecting...
  → Claude 重寫程式碼，不使用 os 模組
```

### Demo 3 — 記憶庫

第一次執行會觸發反思；第二次執行時直接命中記憶庫：

```
  💾 Memory hit! Reusing past strategy (seen 1x)
  → 跳過反思 API 調用，直接套用策略
```

---

## 執行報告

每次執行完自動印出摘要：

```
────────────────────────────────────────────────────────────
  📊  AGENT RUN REPORT  [a3f2c1b0]
────────────────────────────────────────────────────────────
  Task:             Read the file '.../sales.csv', calculate...
  Model:            claude-opus-4-6
  Status:           ✅ success
────────────────────────────────────────────────────────────
  Total tool calls: 4
    ✅ succeeded:    3
    ❌ failed:       1
  Reflections:      1
  Memory hits:      0
  Self-heal rate:   100.0%
────────────────────────────────────────────────────────────
  Tool call timeline:
     1. ❌ read_file [depth=1]  (234ms)
     2. ✅ list_directory  (89ms)
     3. ✅ read_file  (91ms)
     4. ✅ write_file  (76ms)
────────────────────────────────────────────────────────────
```

查看所有歷史執行的統計：

```bash
python main.py --report
```

---

## 在自己的程式中使用

```python
import anthropic
from agent import SelfHealingAgent
from agent.tools import build_default_registry

client = anthropic.Anthropic()

agent = SelfHealingAgent(
    client=client,
    model="claude-opus-4-6",
    tools=build_default_registry(),
    max_retries=3,
    enable_memory=True,   # 記憶庫
    enable_sandbox=True,  # Python 沙盒
    print_report=True,    # 執行後印報告
)

result = agent.run("讀取 data.csv 並計算平均值，存到 result.txt")
print(result.final_answer)
print(f"反思次數: {result.total_reflections}")
print(f"記憶命中: {result.memory_hits}")
```

### 自定義工具

```python
from agent.tools import ToolRegistry

registry = ToolRegistry()

registry.register(
    name="get_stock_price",
    func=lambda symbol: f"{symbol}: $123.45",
    description="Get current stock price for a symbol.",
    input_schema={
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol"}
        },
        "required": ["symbol"],
    },
)

agent = SelfHealingAgent(client=client, model="claude-opus-4-6", tools=registry)
```

---

## 沙盒安全性

`execute_python` 工具在執行前會進行 AST 靜態分析，阻擋危險模組：

**封鎖的模組：** `os`, `sys`, `subprocess`, `shutil`, `socket`, `http`, `urllib`, `requests`, `pickle`, `ctypes` 等

**允許的模組：** `math`, `statistics`, `json`, `csv`, `re`, `datetime`, `numpy`, `pandas`, `scipy`, `matplotlib` 等

**資源限制：** CPU timeout 10 秒、輸出上限 64KB

---

## 專案結構

```
self_healing_agent/
├── agent/
│   ├── core.py          # SelfHealingAgent 主循環
│   ├── reflection.py    # Reflector（結構化反思 + Confidence Gate）
│   ├── memory.py        # FailureMemory（持久化策略記憶庫）
│   ├── logger.py        # RunLogger（結構化 log + 報告）
│   ├── sandbox.py       # Python 執行沙盒
│   └── tools.py         # ToolRegistry + 內建工具
├── examples/
│   └── setup_demo.py    # 建立測試資料
├── logs/                # 執行 log（自動產生）
├── main.py              # Demo 入口
├── requirements.txt
└── .env.example
```

---

## 已知限制與待優化項目

- `retry_counts` 目前以 `tool_use_id` 為 key，每次 Claude 重試會產生新 ID，max_retries 有效性待修正
- 主循環缺少全局 `max_turns` 上限
- Reflection 解析使用手動字串處理，可改用 SDK structured output 提升穩定性
- Memory 儲存的策略僅記錄最終成功的 input，尚未保存完整反思內容

---

## 授權

MIT
