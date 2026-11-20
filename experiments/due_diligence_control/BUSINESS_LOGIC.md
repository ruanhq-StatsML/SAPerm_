# 业务逻辑：尽调注意力的双边控制

核心主张：**尽调不是“把资料读完”，而是在三类稀缺资源下，决定「谁、何时、读什么、能不能上 IC」。**  
FinanceGPT 只打分；规则系统管容量与完整性。

## 1. 稀缺资源（和招聘控制同构）

| 招聘双边控制 | 尽调对应 |
| --- | --- |
| 候选人注意力 / 疲劳 | 分析师注意力 / alert fatigue |
| 面试官评审产能 | Associate / Partner 评审 + IC 档期 |
| 投放 / 触达流量 | 数据室 coverage（读哪些文件） |

三个计数必须分开：

1. **Touches** — 今天打扰分析师几次（digest / Slack）
2. **Work items** — 一次 touch 里塞几份文件
3. **Coverage** — 某笔 deal 又推了多少份材料

混在一起就会出现：邮件少但每人塞 50 份、或同一份 10-K 反复刷覆盖率。

## 2. 一天的业务闭环

```
材料进室 → 模型打风险分 → 规则分配注意力
    → 读完确认风险？ → 完整性够不够上 IC？
         ├─ 够 → IC ready
         ├─ 不够 → 拒绝「尽调完成」宣称（abstain）
         └─ 缺证据 → 升级要材料 / 去脱敏 → 当天再读关键 → 重开
```

操作规则：

**Score → Allocate → Integrity-gate → Escalate → Same-day micro-pass → Dual-look → Reopen → Deadline-pull → Last-mile → Clearance IC**

## 3. 逐步论证

1. **模型只打分，不抢方向盘** — Mute / 频率 / `backlog_hard` / pacing 永远优先。Flood 把分析师打穿（dense confirmed/1k ≈ 52 vs controlled ≈ 200）。
2. **分配：疲劳 − 拥堵 − 冗余** — 边际价值递减；重复推同一 sector/deal 是在烧 IC 前注意力。
3. **Soft-halt ≠ 冻结算** — 停 filler，保留 critical lane。Paused ⇒ budget=0 会在要材料时反而不许看关键。
4. **Integrity gate** — 材料不齐却上会是假 IC；宁可 abstain。Controlled 全 regime false-IC=0。
5. **Escalate** — 只拒绝不追材料 = 把问题留给合伙人；升级把完整性失败转成数据采集工单。
6. **Same-day micro-pass** — 材料刚回来当天就读；IC 日历上晚一天 = 少一天确认窗口。
7. **Dual-look** — 同日 second reviewer，不重新抢 desk budget。
8. **Delayed abstain + Reopen** — Abstain 是 hold 状态，不是坟场。
9. **Dense 才 critical-first；Sparse 别硬抄** — 政策按 regime 切换。
10. **Fast escalate on gap（v12）** — 可得性/脱敏缺口活跃时每天可升级，别卡 3 天冷却。
11. **Last-mile（v14）** — 收官窗口把「已可得但未进 live / 未确认」的 critical 强制放出并做合伙人级确认；sparse_missing 的残差常是确认缺口不是可得性。
12. **Clearance IC（v15）** — 证据清干净后把 allow_ic 的 deal 推进 IC，并校准 abstain 存量；别让已确认风险烂在 incomplete 状态。
13. **Deadline-pull（v16/v17）** — `days_to_ic ≤ 6` 时提前拉 stranded critical；v17 只在确有未确认工作量时开火，attempts 只计 look，避免空转。
14. **Hostile 压测** — 极薄流 + 高缺失/脱敏下 controlled ≫ flood；控制栈的价值在压力态放大。
15. **IC-ready 停 filler（v18）** — 会上已定的 deal 别再往 live 灌噪声；coverage↓、confirmed/1k↑。

## 4. 判决

可替代「筛查 + 注意力分配 + 证据追讨」的前半段尽调；  
**不可替代**「主源确认 + 合伙人对 IC 负责」。

度量完整性用 `recall_all` + false-IC，不用「读了多少页」。  
收官效率另看 deadline_pull confirms/attempts 与 residual abstain。

## 5. 落地话术

1. 模型是红旗雷达，不是自动签字。
2. 规则是 desk 产能与 IC 诚实性的闸门。
3. Soft-halt = 停垃圾阅读，不停关键通道。
4. Abstain / escalate / reopen / clearance = 标准运营状态机。
5. Alt-data 也要不来 → 继续拒绝 complete diligence 宣称，不补全叙事。
6. IC 临近 → deadline-pull；收官 → last-mile；证据齐 → clearance IC。
