# Telegram Trading Bot (观察型信号系统)

这是一个“只提示、不自动下单”的加密货币合约观察系统：

- 数据源：Binance USD-M Futures（公开行情接口）
- 输出：Telegram 消息提醒 + 本地模拟盘账本（JSON）+ 周期性导出（CSV/XLSX）
- 用途：学习量化、做策略迭代、用小资金人工跟单验证

重要说明：本项目不会自动下单，仅生成观察信号与风控建议；内容不构成任何投资建议。

## 功能概览

- `--scan-market`：扫描市场一次，输出满足阈值的候选信号（可选 Telegram 推送）
- `--run-live`：常驻运行循环扫描 + 同步模拟盘 + 定时导出表格（断网/异常时会落盘快照并停止）
- `--paper-record`：将“达到通知条件的候选信号”写入模拟盘账本
- `--paper-sync`：用最新行情推进模拟盘（触发入场/止盈/止损/移动止损/到期平仓等），并更新未平仓浮动盈亏
- `--backtest`：用最近 N 天 Binance 历史 K 线回测观察策略，输出摘要并可导出交易明细 CSV

## 环境要求

- Python 3.11+（Windows/macOS/Linux 均可）
- 网络可访问 Binance 行情接口与 Telegram Bot API

## 快速开始（Windows / PowerShell）

1. 安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

2. 配置环境变量

把 `.env.example` 复制成 `.env`，并填写：

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
LOG_LEVEL=INFO
```

3. 测试 Telegram 是否可用

```powershell
python main.py --telegram-ping
```

## 常用命令

提示：所有参数都可以用 `--help` 查看：

```powershell
python main.py --help
```

### 1) 扫描市场一次（只提示，不常驻）

```powershell
python main.py --scan-market --score-threshold 90 --notify
```

常用可选参数（按需加）：

- `--fixed-symbols BTCUSDT,ETHUSDT,SOLUSDT`：固定观察池
- `--top-volume-limit 10`：无条件加入 24h 成交额前 N 个合约
- `--dynamic-alt-limit 3`：每日从活跃币中筛选最多 N 个“放量/波动”山寨机会
- `--symbol-cooldown-minutes 60`：同币种同方向冷却，减少重复信号
- `--paper-record --paper-path paper_trades.json`：把达到阈值的信号写入模拟盘账本
- `--paper-entry-mode immediate|confirm`：模拟盘入场方式（默认 `immediate`）

通知去重与增强确认（推荐开启 `--paper-record`）：

- 同一币种同方向“相同信号”（触发价/止损几乎一致）只会发送一次完整开仓通知
- 后续再次扫到相同信号，会改为发送“增强确认”提醒（不再刷屏）
- 当 TP1 触发或平仓（止损/时间止损/到期平仓/移动止损等）时，会在常驻运行的同步阶段推送事件通知

### 2) 常驻运行（推荐：扫描 + 模拟盘 + 定时导出）

下面这条会每 5 分钟扫描一次，并把信号写入模拟盘；模拟盘会每 24 小时导出一次表格：

```powershell
python main.py --run-live `
  --loop-minutes 5 `
  --market-summary-every-scans 12 `
  --score-threshold 90 `
  --paper-record `
  --paper-path paper_trades.json `
  --paper-export paper_trades.csv `
  --paper-export-every-hours 24 `
  --paper-snapshot-dir paper_snapshots `
  --notify
```

行为说明：

- 每隔 `--market-summary-every-scans` 轮扫描会发送一条“市场总结”（默认 12；当 `--loop-minutes=5` 时约等于每小时一次），包含每个币的：
  - 结构判断：上升/下跌/横盘（基于 1h EMA20/EMA50 结构）
  - 6h/24h 涨跌幅
  - 24h 估算成交额（由 K 线成交量 * 收盘价近似）
  - 当前 1h 量能相对近 20h 均值倍数
- 程序遇到网络异常（例如断网、请求失败等）时：
  1. 第一次网络异常：暂停本轮（不退出），等待下一轮自动重试（例如 VPN 可能稍后自动重连）
  2. 如果连续两轮仍网络异常：会把当前模拟盘账本导出“最终快照”到 `paper_snapshots/`，并在开启 `--notify` 时发送告警，然后停止运行（退出码 2）

停止运行：按 `Ctrl+C`。

### 3) 手动同步/导出模拟盘

同步模拟盘（推进 pending/open 的交易状态、更新浮动盈亏等）：

```powershell
python main.py --paper-sync --paper-path paper_trades.json --notify
```

只看模拟盘摘要：

```powershell
python main.py --paper-summary --paper-path paper_trades.json
```

### 4) 回测最近 N 天并导出交易明细

```powershell
python main.py --backtest --backtest-days 7 --score-threshold 90 --hold-hours 2 --export-trades backtest_trades.csv
```

交易明细 CSV 包含清晰表头与可读时间（包含 UTC 与本地时间两列）。

### 5) 方向准确率评估（入场后 N 分钟方向对错）

用于回答“方向对了但止盈止损不合理”的问题：对每一笔回测交易，取入场后 N 分钟的价格，判断方向是否正确，并统计胜率。

```powershell
python main.py --direction-eval `
  --direction-trades-csv backtest_trades.csv `
  --direction-horizon-minutes 60 `
  --direction-min-move-pct 0
```

说明：

- 价格取值：使用 Binance 5m K 线，找到 `close_time >= 入场时间 + N分钟` 的第一根K线收盘价作为评估点
- 判定：做多在评估点涨则算对，做空在评估点跌则算对；`--direction-min-move-pct` 可把微小波动视为平局

## 输出文件说明（默认）

这些文件是运行时生成的，仓库已通过 `.gitignore` 忽略它们：

- `paper_trades.json`：模拟盘账本（信号、入场/出场、浮动盈亏等）
- `paper_trades.csv`：模拟盘导出表（便于每周复盘）
- `paper_snapshots/`：断网/异常/停止时的最终快照导出目录
- `backtest_trades*.csv`：回测交易明细

## 策略与风控（简述）

系统核心是一套“观察型入场评价体系”，并带有基础风控/过滤，例如：

- 同币种同方向冷却（减少重复开仓建议）
- BTC 环境冲突过滤（BTC 与山寨信号冲突时降权/过滤）
- 周度趋势过滤（大趋势不一致时过滤逆势单）
- 资金费率与手续费在回测与模拟盘中会计入收益
- 交易推进包含：止损/分批止盈/移动止损/时间止损/到期平仓等

## 开发与测试

```powershell
python -m pytest -q
```
