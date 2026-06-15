# 项目方向：生成式音乐推荐 Agent

将 RecSys Challenge 2026 检索 pipeline 转变为端到端生成式 Agent——自己对话、自己推荐，闭环在自己手里。

## 核心架构

```
用户 → Chat LLM (对话管理) → SID Generator (对话→SID) → Codebook解码 → 回复
```

两个模块：Chat LLM 现成大模型直接用，SID Generator 需要自己训。不需要 State Extractor。

## SID 构造

用 **RQ-VAE**（不用 Residual KMeans——KMeans 已证实 93% 碰撞率，不可用）。

| 模态 | 维度 | 语义 | 用否 |
|---|---|---|---|
| attributes-qwen3 | 1024d | 流派/情绪/时代 | ✅ |
| lyrics-qwen3 | 1024d | 歌词内容 | ✅ |
| cf-bpr | 128d | 协同行为（谁在听） | ✅ |
| metadata | 1024d | 曲名/艺人名 | ❌ shortcut learning |
| audio/image | 512/768d | — | ❌ 维度低，与 attr 冗余 |

三模态 L2 normalize → 拼接 2176d → RQ-VAE（depth=4, k=256, EMA+commitment loss）→ track→SID 映射表。

**为什么不用 metadata**：metadata 让模型学会"查艺人字典"而非理解音乐语义。用户说 "Coldplay" → 所有 Coldplay 歌的 SID 挤在一起 → 模型随便输出一个 cluster 内的。捷径存在，模型就不学内容了。显式指称（"放 Yellow by Coldplay"）靠训练中记忆映射解决——55 万条样本足够。

## 训练数据

**"去掉 Gemini 轨迹" = 丢掉 `role=="music"` 的文本，不是丢掉训练信号。**

| 保留 | 丢掉 |
|---|---|
| `role=="user"` 文本 | `role=="music"` 的 Gemini 推荐文本 |
| ground truth track → SID label | Gemini 的推荐策略 |

逻辑：用合成 query（Gemini 生成的 user 话语），预测真实 label（playlist 中的 ground truth track）。输入可以是合成的，输出必须是真实的。

数据增强：同一 session 前缀截断（t1, t1-2, t1-3, ..., t1-8）→ ~55 万条训练样本。

## Playlist 利用

**用。** Playlist 是真实用户行为数据，和 Gemini 推荐文本有本质区别：

| Gemini 推荐 | User Playlist |
|---|---|
| LLM 生成 → 蒸馏 Gemini | 真实用户行为 → 学习真实偏好 |

两种注入方式：
1. **SID 构造**：cf-bpr 已编码全局协同行为
2. **SID Generator 输入**：playlist embedding attention pooling → user_vector → concat 进 decoder

**不需要重排**。Playlist 信号已经通过 concat 进入 SID Generator，模型一步完成"理解对话 + 理解用户"。重排是 Vision A 的思维。

## 架构定位

**Vision B（采用）：SID Generator 是唯一推荐引擎，砍掉 BM25/dense。**

Vision A（放弃）：SID 只是多路 ensemble 的一个通道 → 退化成了检索特征。

Baseline 保留但不做 ensemble，只做对比基线：生成式 vs 检索式，在哪些 turn 各有优劣。

## 执行路线

| Phase | 内容 |
|---|---|
| 1 | 三模态 Embedding 加载 + RQ-VAE 训练 → codebook + track→SID |
| 2 | 训练数据构造 + SID Generator seq2seq 训练 + constrained decoding |
| 3 | Agent 组装（Chat LLM + SID Generator 链路串联） |
| 4 | Dev 评测 + 对比损失对齐迭代 |

## 缺失模块

| 模块 | 说明 |
|---|---|
| Constrained decoding | SID Generator 只输出合法 SID token（trie/FSM 约束） |
| SID→Track 倒排 | codebook 查表，毫秒级 |
| SID 评估桥接 | SID→track_ids→evaluate.py |
| 训练数据构造脚本 | user 文本提取 + gold track→SID label |

## 工业成本

**离线**：RQ-VAE 训练 ~3-4h (A100, ~$5-10)，SID Generator ~6-12h (~$15-30)

**在线 per turn**：
- SID Generator forward (T5-small 60M)：~50ms GPU / ~1-2s CPU
- Chat LLM (Qwen3-4B int8)：~500ms GPU
- Pooling + 倒排查：<2ms
- 总延迟 ~600ms，GPU 显存 ~4GB，磁盘 ~500MB

**规模化**：10 QPS → 1×T4 (~$100/月)，1000 QPS → 2×A100 (~$3000/月)

**冷启动**：新 track → 算 embedding + 已有 codebook 编码 → 秒级上线；新用户 → SID Generator 纯对话模式 + cf-bpr 全局先验
