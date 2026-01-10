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

# 接受新資料作為基準
sentinel accept "price:收盤價" --reason "確認資料修正"

# 匯出備份
sentinel export "price:收盤價" -o ./backup.parquet
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
