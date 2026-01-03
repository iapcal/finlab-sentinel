# finlab-sentinel

[![CI](https://github.com/yourusername/finlab-sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/finlab-sentinel/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/finlab-sentinel.svg)](https://badge.fury.io/py/finlab-sentinel)
[![Python Versions](https://img.shields.io/pypi/pyversions/finlab-sentinel.svg)](https://pypi.org/project/finlab-sentinel/)

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
- **通知機制**: 支援自訂 callback（如 LINE、email 通知）
- **CLI 工具**: 管理備份、查看差異、接受新資料

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
