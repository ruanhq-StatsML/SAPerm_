# Bilateral exposure control (Phase 1: rule-based)

招聘双边市场的**直接策略层**：先管住过度曝光，再谈预测与联合优化。

三类资源分开记账（不要混成一个“曝光”）：

| 资源 | 含义 |
| --- | --- |
| 候选人营销触达次数 | 邮件 / Push / 站内主动触达次数 |
| 触达内岗位槽位 | 一封邮件可带多个岗位 |
| 同一岗位曝光次数 | 同一 posting 的 impression |

主动营销受规则约束；**不限制**候选人自主搜索与申请。

## Phase 1 已实现

### 候选人侧

Frequency Cap · Cooldown · Cross-channel Dedup · Company-level Cap · Pipeline-aware · Negative Feedback · 动态频控表（按求职状态给每日岗位槽位）。

### 公司侧

Application Cap · Interview Cap · **Backlog Control** · Family Dedup · **Pacing**。

积压例子（需求 10 人、待审 120、日审 30、日自然申请 20）会触发 `backlog_hard`，营销流量归零，直到审核追上。

### 分批曝光

四阶段释放：probe → qualify → interview → dynamic fill。用申请量/质量区分：

- **Supply Shortage**：人不够 → 继续放量
- **Matching Failure**：量够质差 → 停量、改匹配，而不是加曝光

### 双边分配

\[
Score_{ij}=\Delta V_{ij}-C_i^{\mathrm{fatigue}}-C_j^{\mathrm{congestion}}-C_{ij}^{\mathrm{redundancy}}
\]

在候选人频控与公司预算的**硬约束**下贪心分配（退订、积压硬停优先于高分）。

## Run

```bash
python3 -m experiments.bilateral_exposure_control
python3 -m experiments.bilateral_exposure_control --days 30 --candidates 120 --seeds 2026 2027 2028 2029
python3 -m pytest experiments/bilateral_exposure_control/test_smoke.py -q
```

`results/FACTORIAL.md` 是 2×2 集群随机化仿真看板：

| Arm | 候选人动态频控 | 公司动态容量 |
| --- | --- | --- |
| control | 否 | 否 |
| t1_candidate | 是 | 否 |
| t2_company | 否 | 是 |
| t3_both | 是 | 是 |

关注：每千次营销曝光的增量面试、每百个期初开放名额·日的填补、退订率、审核等待、净价值。

## 落地优先级（机会）

| 阶段 | 内容 | 本包状态 |
| --- | --- | --- |
| 1 | 频控、去重、积压、分批曝光、双边硬约束贪心 | **已落地（本目录）** |
| 2 | LightGBM 疲劳/容量；DeepHit 作 DP 状态 | stub：接口位在 README 下节 |
| 3 | Causal Transformer 边际触达收益 | 未做 |
| 4 | Min-Cost Flow / DP 大规模分配 | 未做（Phase 1 用贪心） |

与仓库内 FSDS Thompson 激励的衔接：公司侧积压份额可作成 `λ · w_j` 惩罚项（同 `fsds_tabred` / `fsds-review-agent` 的均匀发力逻辑），在 Phase 2 把规则 `pace_factor` 换成预测分位容量。

审核 agent：`fsds-review-agent` 的 `review_board` 可直接吃本仿真的 arm 份额/阻塞原因，做策略板审核。

## Phase 2 钩子（尚未接线）

```text
B_i* = max { a : P(Fatigue | X_i, H_i, a) <= delta }   # 随机频控实验估计
C_j,t = [ Mhat_j,t - Q_j,t - Ahat_organic_j,t ]_+     # 保守分位而非均值
```

DeepHit 的 7 日填补概率进入状态，**不要**直接当排序分；边际收益仍要覆盖拥堵与疲劳成本。
