# ModaSense 专题：RFPerm / CFPerm / MMD占比 / ρ_PO / localization

数据：ModaSense/emotion（n=500；标签两两冲突概率=1）。  
Power 表：`power_mm_modasense_full`（B=80, N_perm=100, X=[M_text,M_image]）。  
MMD：B=50，z-scored 三模态块（batch0 vs batch1）。  
Localization：三模态 CF LOCO + carrier subset（单次窗，见 `case_modasense_business`）。

定义：

- **RFPerm power**：该特征集上的拒绝率  
- **CFPerm power**：命名模态块 LOCO 拒绝率  
- **MMD 占比**：\(\pi_m=\mathrm{MMD}_m/\sum_{m'\in\{t,i,a\}}\mathrm{MMD}_{m'}\)（只描述分布移动份额，不是因果权重）  
- **ρ_PO**：\(\mathrm{PO}([M_t,M_i])/\mathrm{PO}(M_t)\)

---

## 1. 总表（只报已跑通的数）

| scenario | π_text / π_image / π_audio | MMD top | RF text → MM | CF text / image | ρ_PO | 综合标签 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| null | 0.35 / 0.34 / 0.31 | flat | 0.01 → 0.01 | 0.03 / 0.06 | 2.14 | Type-I 附近 |
| text_sentiment | **0.91** / 0.03 / 0.06 | text | **1.00 → 1.00** | **1.00** / 0.54 | 1.95 | 检测+定位在 text |
| image_brightness | 0.03 / **0.95** / 0.02 | image | 0.00 → 0.05 | 0.00 / 0.01 | **0.62** | 高 MMD + 吸收；power 近零 |
| audio_energy | 0.02 / 0.02 / **0.95** | audio | ≈0 → 0（smoke） | audio CF≈0（单次窗） | ~1.06（smoke） | MMD 有、power/定位弱 |

MMD 原值：null ≈0.016/0.016/0.014；text 0.326/0.011/0.020；image 0.016/**0.570**/0.014；audio 0.012/0.014/**0.542**。

---

## 2. 分情景论证

### 2.1 `text_sentiment_shift` — 四方一致指向 text

| 证据 | 数 | 含义 |
| --- | ---: | --- |
| MMD 占比 | π_text=**0.91** | 分布移动几乎全在文本块 |
| RFPerm | text=**1.00**，MM=**1.00** | 单用文本即可饱和检测；拼图像不改变 RF |
| DRPerm | 0.28 → **0.64** | 拼图像后 DR 上升（ρ_PO>1 同源） |
| CFPerm | text=**1.00**，image=0.54 | 主定位在 text；image 有次级共现但弱于 text |
| ρ_PO | **1.95** | PO 变大：图像没有把情感 covariate「吸收掉」 |
| Localization | top=`text_meta`（imp=0.40, p=0.012, reject） | 块级点名 text |
| Subset | `vader_neg`, `compound`, `pos`, `polar_gap`, `neu`（+少量 POS） | 驱动维是 VADER 极性，不是画面 |

**发现：** 当 shift 落在文本情感轴时，MMD 占比、RFPerm、CFPerm、carrier subset **同一模态、同一特征族**。ρ_PO>1 说明此时 multimodal concat 的角色是「检测已饱和」，不是 nuisance 吸收。

### 2.2 `image_brightness_shift` — MMD/ρ_PO 与 power 分裂

| 证据 | 数 | 含义 |
| --- | ---: | --- |
| MMD 占比 | π_image=**0.95** | 分布移动几乎全在图像 |
| RFPerm | 0.00 → 0.05 | 相对 α 几乎无检测力 |
| CFPerm | text=0.00，image=0.0125 | 块级 LOCO **不拒绝**（近 null） |
| ρ_PO | **0.62** | PO 下降：图像进 nuisance 后吸收亮度 covariate |
| Carrier subset | `img_brightness`, `hue_std`, `contrast`, `sat_mean` | 窗差的载体确实是视觉维 |
| 单次 CF LOCO | image imp≈5e-5, p=0.38, **不拒绝** | 与 B=80 的 CF power≈0.01 一致 |

**发现（重要）：** 高 MMD + ρ_PO≪1 **可以与近零 RF/CF power 并存**。  
图像通道在变，且拼进 \(X\) 能降低 PO-risk，但 **不足以** 在当前 Y 上形成稳定拒绝。  
科学含义：MMD/ρ_PO 描述的是 **covariate 通道**；RFPerm/CFPerm 描述的是 **可检测 / 可定位的拒绝事件**。二者不可互换。  
业务含义：亮度漂移应走「视觉 nuisance / 抽检」，不能因为 MMD 大就当成 RF 报警。

### 2.3 `audio_energy_shift` — 仅 MMD 清楚

| 证据 | 数 | 含义 |
| --- | ---: | --- |
| MMD 占比 | π_audio=**0.95** | 分布移动在音频 |
| RFPerm（smoke） | 各模态 ≈0 | 无检测 |
| CF（单次三模态窗） | audio imp=0, p=1；top 误落在 text 且 p>0.9 | **定位失败 / 欠功效** |
| ρ_PO（smoke） | all/text≈1.06 | 无明显吸收 |

**发现：** MMD 已标明音频通道；在现有 B/Y 设定下，RF/CF/ρ_PO **尚未形成与 text 窗同级的证据链**。专题结论应写清：音频轴目前是「MMD-only」发现，power/localization 需更大 B 或更贴 Y 的标签，**不能外推成已定位**。

### 2.4 `null_shift` — 校准

MMD 三模态同阶（~0.015）；RF≈0.01；CF≈0.03–0.06。ρ_PO=2.14 在 null 上波动大、不宜解读为吸收。作 Type-I / 基线即可。

---

## 3. Localization 与 modality 占比如何合读

| 层级 | text 窗 | image 窗 | audio 窗 |
| --- | --- | --- | --- |
| MMD 占比 | text 91% | image 95% | audio 95% |
| CFPerm power（B=80, text+image） | text 1.00 > image 0.54 | 两者≈0 | （无 audio 列） |
| 单次块 LOCO | text reject | 全不 reject | 全不 reject |
| Subset | VADER 极性 | brightness 等载体 | 不可信（信号太弱） |

规则：

1. **先看 MMD 占比** → 哪条模态的特征分布在动  
2. **再看 RFPerm** → 该动是否变成拒绝  
3. **再看 ρ_PO** → concat 是吸收（≪1）还是抬 PO（≳1）  
4. **最后看 CFPerm + subset** → 仅在 CF 有拒绝或 imp 明显时，才声称「定位到哪些 meta」

违反第 4 步（例如 audio 窗硬报 text drivers）就是过读。

---

## 4. 科学发现（可写进报告的句子）

1. **模态路由与 MMD 占比对齐是可重复的**：三色窗上 π_routed≈0.91–0.95，null 上 π 均匀。  
2. **Text 情感窗给出完整证据链**：高 π_text + RF=1 + CF_text=1 + VADER subset + ρ_PO≈1.95（检测饱和、非吸收）。  
3. **Image 亮度窗给出「静默吸收」**：π_image≈0.95 且 ρ_PO=0.62，但 RF/CF≈α —— covariate 通道成立，检测/块定位不成立。  
4. **Audio 窗目前证据不完整**：仅 MMD 强；RF/CF/ρ_PO 未闭合，不能与 text 窗并列宣称 localization。  
5. **方法论**：对 multimodal 模型，应用 `{modality: array}` 同时报 MMD占比、RFPerm、CFPerm、ρ_PO；只有四者闭合才升级为「该模态驱动 shift」的强声明。

---

## 5. 配置与文件

- Full power：`power_mm_modasense_full.json`（B=80）  
- MMD：`mmd_two_datasets.json` → modasense  
- Localization：`case_modasense_business.json`  
- Helper：`mmd_modalities.py`
