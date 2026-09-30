# finlab-sentinel

![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![Windows](https://img.shields.io/badge/OS-Windows-0078D6?logo=windows&logoColor=white)
![Linux](https://img.shields.io/badge/OS-Linux-FCC624?logo=linux&logoColor=black)
![macOS](https://img.shields.io/badge/OS-macOS-000000?logo=apple&logoColor=white)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/iapcal/finlab-sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/iapcal/finlab-sentinel/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/codecov/c/github/iapcal/finlab-sentinel)](https://codecov.io/gh/iapcal/finlab-sentinel)

**finlab-sentinel** 是 [finlab](https://github.com/finlab-python/finlab) 套件的防禦層，用於監控 `data.get` API 的資料變化，防止未預期的資料異動影響回測或選股結果。

## 功能特色

- **自動比對**: 每次 `data.get` 時自動比對歷史資料
- **滾動備份**: 保留 7 天（可配置）的備份資料
- **智慧檢測**:
  - 數值容差比對（可配置 rtol/atol）
  - dtype 變更檢測
  - NA 類型差異檢測（pd.NA vs np.nan vs None）
- **彈性政策**:
  - `append_only`: 只允許新增，不允許刪除或修改歷史
  - `threshold`: 允許小幅度變更（如 10% 以內）
  - 黑名單配置：指定可修改歷史的資料集
- **可配置行為**:
  - 拋出例外（預設）
  - 警告並使用快取
  - 警告並使用新資料
- **Preprocess Hook**: 比對前預處理（如四捨五入），支援萬用字元模式
- **通知機制**: 支援自訂 callback（如 LINE、email 通知）
- **CLI 工具**: 管理備份、查看差異、接受新資料
- **時間旅行**: 回到歷史時間點取得當時的備份資料
- **永久 Patch**: accept 新資料時自動保存舊 baseline 快照，不受備份清理影響，隨時可查閱、匯出或還原（撤銷 accept）

## 安裝

```bash
pip install finlab-sentinel
```

或使用 uv：

```bash
uv add finlab-sentinel
```

## 快速開始

```python
import finlab_sentinel

# 啟用 sentinel
finlab_sentinel.enable()

# 正常使用 finlab
from finlab import data
close = data.get('price:收盤價')  # 自動備份並比對

# 如果資料異常，會根據配置拋出例外或警告
```

## 配置

建立 `sentinel.toml` 檔案：

```toml
[storage]
path = "~/.finlab-sentinel/"
retention_days = 7

[comparison]
rtol = 1e-5
change_threshold = 0.10

[comparison.policies]
default_mode = "append_only"
history_modifiable = ["fundamental_features:某些財報資料"]
# 允許 NA→有值 的轉換（例如：預估資料後來補上）
allow_na_to_value = ["price:收盤價"]

[anomaly]
behavior = "raise"  # raise | warn_return_cached | warn_return_new
save_reports = true

# 可選：設定通知 callback
# callback = "myproject.notifications:send_line"
```

## CLI 使用

```bash
# 列出所有備份
sentinel list

# 清理過期備份
sentinel cleanup --days 14

# 查看資料差異
sentinel diff "price:收盤價"

# 接受新資料作為基準（會印出保存舊 baseline 的 patch id）
sentinel accept "price:收盤價" --reason "確認資料修正"

# 一定要能撤銷：無法建立 patch 時不接受、不做任何變更；--json 輸出單行 JSON
sentinel accept "price:收盤價" --require-patch --json

# 匯出備份
sentinel export "price:收盤價" -o ./backup.parquet

# 管理永久 patch（accept 時自動產生）
sentinel patch list
sentinel patch show <patch_id>
sentinel patch export <patch_id> -o ./old_data.parquet
sentinel patch delete <patch_id>
sentinel patch restore <patch_id>   # 撤銷 accept：還原 patch 保存的舊 baseline
```

## 處理資料異常

當檢測到資料異常時：

```python
from finlab_sentinel import DataAnomalyError

try:
    close = data.get('price:收盤價')
except DataAnomalyError as e:
    print(f"資料異常: {e.report.summary}")
    # 檢查報告詳情
    print(f"變動比例: {e.report.comparison_result.change_ratio:.1%}")

    # 如果確認要接受新資料
    from finlab_sentinel.core.interceptor import accept_current_data
    accept_current_data('price:收盤價', reason="確認資料修正")
```

需要知道 accept 產生的 patch（例如之後要撤銷）時，改用 `accept_dataset`：

```python
import finlab_sentinel as fs

result = fs.accept_dataset("price:收盤價", reason="確認資料修正", require_patch=True)
result.patch_id    # 保存舊 baseline 的 patch；fs.restore_patch(result.patch_id) 即撤銷
result.to_dict()   # 可序列化成 JSON 的結果摘要
```

`accept_dataset` 失敗時拋出 `AcceptError`（找不到 baseline、取不到目前資料、`require_patch=True` 卻無法建立 patch、或 accept 途中 baseline 被其他程序更新），baseline 保持不變。`require_patch=True` 時，只要 baseline 有任何改變（content hash 或原始資料，即使 preprocess hook 讓 hash 相同、或 `[accept] create_patch = false`）都會建立 patch；資料完全相同時不需要 patch，`patch_id` 為 `None`。

## Preprocess Hook

Preprocess hook 讓你可以在比對前先對資料做預處理，例如四捨五入、排序欄位等。這在處理預期的浮點數精度差異時特別有用。

**注意**: 預處理只用於比對，回傳給使用者的永遠是原始資料。

```python
import finlab_sentinel

# 註冊特定 dataset 的 preprocess hook
finlab_sentinel.register_preprocess_hook(
    "price:收盤價",
    lambda df: df.round(2)  # 四捨五入到小數第二位
)

# 支援萬用字元模式
finlab_sentinel.register_preprocess_hook(
    "price:*",  # 符合所有 price: 開頭的 dataset
    lambda df: df.round(2)
)

# 也支援 ? 萬用字元（符合單一字元）
finlab_sentinel.register_preprocess_hook(
    "price:?",
    lambda df: df.round(2)
)

finlab_sentinel.enable()

# 使用 finlab
from finlab import data
close = data.get('price:收盤價')  # 比對時會先 round(2)，但回傳原始資料
```

### 進階用法

```python
import finlab_sentinel

# 自訂預處理函式
def normalize_for_comparison(df):
    """標準化 DataFrame 以忽略預期的差異"""
    df = df.copy()
    # 四捨五入數值欄位
    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    # 排序欄位（忽略欄位順序差異）
    df = df[sorted(df.columns)]
    return df

finlab_sentinel.register_preprocess_hook("fundamental_features:*", normalize_for_comparison)

# 取消註冊
finlab_sentinel.unregister_preprocess_hook("price:收盤價")

# 清除所有 hooks
finlab_sentinel.clear_preprocess_hooks()
```

### 優先順序

當多個 pattern 都符合時，精確匹配優先於萬用字元匹配：

```python
finlab_sentinel.register_preprocess_hook("price:*", lambda df: df.round(1))
finlab_sentinel.register_preprocess_hook("price:收盤價", lambda df: df.round(2))

# "price:收盤價" 會使用 round(2)（精確匹配）
# "price:開盤價" 會使用 round(1)（萬用字元匹配）
```

## 時間旅行 (Time Travel)

時間旅行功能讓你可以回到過去的某個時間點，取得當時備份的資料。這對於重現歷史回測結果、調查資料異動，或驗證策略在特定時間點的表現非常有用。

### 基本用法

```python
import finlab_sentinel as fs
from datetime import datetime

# 啟用 sentinel
fs.enable()

# 設定時間旅行到指定時間點
fs.set_time_travel(datetime(2024, 1, 5, 14, 30))

# 現在 data.get() 會回傳該時間點的備份資料
from finlab import data
close = data.get("price:收盤價")  # 回傳 2024-01-05 14:30 的備份資料

# 結束時間旅行，回到正常模式
fs.exit_time_travel()
```

### 查詢狀態

```python
# 查詢目前時間旅行狀態
status = fs.get_time_travel_status()
print(status)
# {'enabled': True, 'target_time': '2024-01-05T14:30:00'}
```

### 錯誤處理

```python
from finlab_sentinel import NoHistoricalDataError

try:
    fs.set_time_travel(datetime(2020, 1, 1))  # 很久以前
    close = data.get("price:收盤價")
except NoHistoricalDataError as e:
    print(f"找不到該時間點的備份資料: {e}")
finally:
    fs.exit_time_travel()
```

### 使用情境

- **重現歷史回測**: 確保使用與當時相同的資料進行回測
- **調查資料異動**: 比對不同時間點的資料差異
- **驗證策略表現**: 在特定歷史時間點驗證選股策略

## 永久 Patch

當你 accept 新資料作為 baseline 時，sentinel 會自動建立一個**永久 patch**，保存 accept 前的舊 baseline 完整快照與 diff 摘要。Patch 存放在 `<storage.path>/patches/`，**不受滾動備份的 retention 清理影響**，之後可以隨時查閱、匯出舊資料，或還原成 baseline（見下方「還原 patch」）。

Patch ID 的格式為 `<backup_key>__<時間>`；同一秒內建立多個 patch 時，後建立者會加上 `_2`、`_3` 等後綴。

### CLI 用法

```bash
# 列出所有 patch（可用 --dataset 過濾）
sentinel patch list

# 查看 patch 詳情（含 diff 摘要）
sentinel patch show "price__收盤價__2026-07-11T10-30-00"

# 匯出 accept 前的舊資料
sentinel patch export "price__收盤價__2026-07-11T10-30-00" -o ./old_data.parquet

# 刪除 patch（需確認，或加 --force）
sentinel patch delete "price__收盤價__2026-07-11T10-30-00"

# 還原 patch，撤銷當初的 accept（需確認，或加 --yes）
sentinel patch restore "price__收盤價__2026-07-11T10-30-00"
```

### Python API

```python
import finlab_sentinel as fs

# 列出 patch
patches = fs.list_patches("price:收盤價")
for p in patches:
    print(p.patch_id, p.created_at, p.diff_summary["summary_text"])

# 取回 accept 前的舊 baseline DataFrame
old_close = fs.load_patch_data(patches[0].patch_id)
```

### 還原 patch（撤銷 accept）

`sentinel patch restore` 以 patch 保存的舊 baseline 取代該資料源目前的 baseline，回到 accept 之前的狀態：資料與 content hash 都與 accept 前完全相同，因此下次 `data.get` 的比對行為也與 accept 前一致（例如當初觸發異常的新資料會再次觸發異常）。

```bash
# 先預覽：列出目前 baseline 與 patch 資料的 hash、時間、大小，不寫入任何東西
sentinel patch restore "price__收盤價__2026-07-11T10-30-00" --dry-run

# 還原（需確認，或加 --yes / -y（同 --force / -f）略過；--reason 記錄原因）
sentinel patch restore "price__收盤價__2026-07-11T10-30-00" --reason "撤銷誤判的 accept"

# 給程式呼叫：--json 輸出單行 JSON（需搭配 --yes 或 --dry-run，不會詢問）
sentinel patch restore "price__收盤價__2026-07-11T10-30-00" --yes --json
```

確認時只會取代預覽中看到的那個 baseline：若預覽之後 baseline 又被更新，還原會中止（回傳 1），不做任何變更。

```python
import finlab_sentinel as fs

result = fs.restore_patch("price__收盤價__2026-07-11T10-30-00", reason="撤銷誤判的 accept")
result.changed       # baseline 是否被取代
result.new_patch_id  # 被取代的 baseline 另存成的新 patch；還原它即可撤銷這次還原
result.to_dict()     # 可序列化成 JSON 的結果摘要

# 只預覽（不寫入）：preview.previous 是目前的 baseline，preview.patch 是要還原的 patch
preview = fs.restore_patch("price__收盤價__2026-07-11T10-30-00", dry_run=True)
# 確認後還原；預覽之後 baseline 若有變動則拋出 PatchRestoreError、不做任何變更
fs.restore_patch("price__收盤價__2026-07-11T10-30-00", expected_latest=preview.latest)
```

- **還原本身也可以再還原**：被取代的 baseline 會先另存成一個新 patch（reason 預設為 `restore of <patch_id>`，`patch show` 會顯示 `Restored From`），之後 restore 這個新 patch 就能回到還原前。
- **來源 patch 會保留**：還原不會消耗或刪除 patch，同一個 patch 可以重複使用；若 baseline 已與 patch 相同（hash 與資料皆一致）則不寫入任何東西，重複執行是安全的。
- **content hash 原樣還原、不重新計算**：baseline 的 hash 是 `data.get` 攔截時對 preprocess hook 處理後的資料計算的，還原時直接沿用 patch 記錄的值，所以在沒有註冊 hook 的 CLI 中還原也正確。
- **Retention**：還原後的 baseline 是還原當下建立的一般備份，retention 清理對待它的方式與剛存入的 baseline 相同；accept 與還原之前的備份仍留在歷史中（時間旅行查得到），照一般規則過期。
- **不會讓資料源失去 baseline**：新檔案完整寫入並 flush 到磁碟後才更新索引；任何一步失敗，還原會中止並拋出 `PatchRestoreError`（CLI 回傳 1），原本的 baseline 保持不變，也不留下半成品。
- **不要在啟用 sentinel 的程序執行中還原（或 accept）同一個資料源**：寫入 baseline 時一律檢查最新的 baseline 仍是當初讀到的那一個（compare-and-swap），所以兩邊都不會覆蓋對方——還原途中 baseline 被改動時還原會中止；`data.get` 比對到一半時 baseline 被還原，它不會存檔覆蓋（只記錄警告），下次 `data.get` 才以還原後的 baseline 比對。但那個執行中的程序已經拿到、並依舊 baseline 驗證過的資料不會因此改變，所以請等它結束再還原。
- 資料源目前沒有 baseline 時，patch 資料直接成為 baseline（沒有東西需要另存）；patch 不存在時拋出 `PatchNotFoundError`（CLI 回傳 1），不做任何變更。

### 關閉自動建立

在 `sentinel.toml` 中設定：

```toml
[accept]
create_patch = false
```

注意：若 accept 時新資料與現有 baseline 完全相同（hash 一致），不會產生 patch；patch 建立失敗（如磁碟已滿）不會阻擋 accept 本身，只會記錄錯誤日誌（`accept_dataset` 的 `patch_error` 會說明原因；要讓 accept 在這種情況下失敗，請用 `--require-patch` / `require_patch=True`）。無法計算 diff 時（例如 index 重複），patch 仍會建立，diff 摘要記為 `diff unavailable`。

## 自訂通知

```python
def send_line_notification(report):
    """當檢測到異常時發送 LINE 通知"""
    import requests
    requests.post(
        "https://notify-api.line.me/api/notify",
        headers={"Authorization": f"Bearer {LINE_TOKEN}"},
        data={"message": f"finlab 資料異常: {report.summary}"}
    )

# 在 sentinel.toml 中設定
# [anomaly]
# callback = "myproject.notifications:send_line_notification"
```

## 開發

```bash
# Clone 專案
git clone https://github.com/yourusername/finlab-sentinel
cd finlab-sentinel

# 使用 uv 安裝開發依賴
uv sync --dev

# 執行測試
uv run pytest

# 執行 lint
uv run ruff check src/ tests/
uv run mypy src/
```

## License

MIT License
