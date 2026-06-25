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

### Playlist 注入实现计划（Phase 2 提升项）

**状态**：❌ 未实现。当前 SID Generator 只用对话历史 + user_culture，没有 playlist embedding。

#### 输入构造

对每个 session 的每个 turn，从已接受的 track 提取 playlist signal：

```
turn 1: playlist = []                    → global prior (cf-bpr)
turn 2: playlist = [track_1]             → 1 track embedding
turn 3: playlist = [track_1, track_2]    → 2 track embeddings
...
turn N: playlist = [track_1, ..., track_{N-1}] → N-1 track embeddings
```

每个 track 的 embedding 来源（已在 `exp/sid/multimodal_2176d/embeddings.npy` 中）：
- attributes-qwen3: 1024d
- lyrics-qwen3: 1024d
- cf-bpr: 128d
- 合计 2176d（L2 normalized，可直接复用）

#### 两种注入方案

| 方案 | 方法 | 优点 | 缺点 |
|---|---|---|---|
| **A: Text 注入** | playlist mean pooling → 找最近邻 track → 描述拼进 prompt | 改动最小，不改模型 | 信息损失大 |
| **B: Embedding 注入** | playlist → attention pooling → user_emb token → concat 进 decoder | 信息完整，端到端 | 需改模型架构 |

**推荐先用方案 A 快速验证，方案 B 长期更好。**

#### 方案 A 实现步骤（Text 注入，~2h）

1. **构建 track embedding 索引**：`src/sid/build_track_index.py`
   - 加载 `embeddings.npy` + `track_ids.txt` → faiss IndexFlatIP
   - 加载 metadata（track_name, artist_name, genres）→ 描述文本映射

2. **改造 `build_training_data.py`**：
   - 每个 turn 收集已接受 track 的 embedding → mean pooling
   - 用 faiss 找 top-3 最近邻 track（泛化到未见 track）
   - 拼成：`"User listens to: {track1_desc}; {track2_desc}; {track3_desc}"`
   - 加入 input parts

3. **改造 `sid_inference.py` 的 `build_session_inputs`**：
   - 同上逻辑，推理时实时查

#### 方案 B 实现步骤（Embedding 注入，~1d）

1. **Attention pooling 模块**（`src/sid/playlist_encoder.py`）：
   ```python
   class PlaylistEncoder(nn.Module):
       # 2176d track embeddings → multi-head self-attention → mean pool → 2176d user_vector
       # 空 playlist 时输出可学习的 <empty> embedding
   ```

2. **注入 SID Generator**：
   - 在 decoder input 前 concat 一个 `[USER_EMB]` token，token embedding = projected user_vector
   - 或：在每个 transformer layer 的 cross-attention 中注入（改动更大）

3. **训练改动**：
   - `train_sid_generator.py`：dataset 增加 `user_emb` 字段
   - 冻结 Qwen3 主体，只 train LoRA + playlist encoder
   - 新增 special token `<USER_EMB>`，embedding 由 playlist encoder 输出替换

4. **推理改动**：
   - `sid_inference.py`：每个 turn 实时算 playlist embedding 再生成

#### 数据流总结

```
Session track_ids (已接受) 
  → exp/sid/multimodal_2176d/embeddings.npy 查表 
  → attention pooling (方案B) / mean+检索 (方案A)
  → user_vector / 描述文本 
  → concat 进 SID Generator input
  → 一步生成 SID
```

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

## 5090 训练加速优化过程

最终配置：micro_batch=16, grad_accum=8, ckpt=off, bf16, liger fused CE。3.5h, VRAM 80%。

### 已尝试

| 方案 | 结果 |
|---|---|
| batch=256, ckpt=on | 5h，backward 重算 forward ~30% 冗余 |
| batch=96, ckpt=off | OOM（激活 28 层全存） |
| batch=64, ckpt=off | OOM |
| batch=16, ckpt=off | ✅ 3.5h，vrrm 80% |
| batch=24, ckpt=off | 待测试 |

### 剩余可试

- **flash SDP 显式启用**：`torch.backends.cuda.enable_flash_sdp(True)` 启动 flash attention kernel，省 1-2 GB attention 激活 → 可能让 batch=24 或 32 不 OOM
- **4-bit 基座 (QLoRA)**：省 ~1 GB 模型权重，但有量化开销
- **max_length 压缩**：384 省 ~3-5 GB 但截断 40% 数据

## 工业成本

**离线（实测，非估算）**：
- RQ-VAE 训练：5070 Ti ~1 分钟（免费）
- SID Generator：5090 ~3.5h（早停，3 epoch 内收敛），~$1-3

**在线 per turn**（Qwen3-0.6B LoRA + constrained decoding）：
- 单请求推理：~50-100ms（beam=3, fp16, Flash Attn）
- 显存：~2GB（fp16） 或 ~600MB（4-bit）
- 无需 Chat LLM 单独部署（Qwen3 同一模型输出 SID，外层加对话模板即可）

**规模化**：10 QPS → 1×T4 (~$100/月)，1000 QPS → 1×A100 (~$1000/月)

**冷启动**：新 track → 算 embedding + 已有 codebook 编码 → 秒级上线；新用户 → SID Generator 纯对话模式 + cf-bpr 全局先验

  5090 上运行的命令：

  python src/sid/train_sid_generator.py \
    --train_pt data/sid_train_512.pt \
    --eval_pt data/sid_eval_512.pt \
    --model_id Qwen/Qwen3-0.6B \
    --output_dir out/sid_generator \
    --preset 5090 \
    --epochs 5

    # 1. 克隆
  git clone https://github.com/Goodnamee/music-rec.git
  cd music-rec

  # 2. 安装
  pip install -r requirements.txt
  pip install flash-attn --no-build-isolation  # Linux 5090

  # 3. 预下载 Qwen3 模型
  python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')"

  # 4. 上传数据文件（或服务器上重新生成）
  # 需要上传这些大文件（不在 git 中）：
  #   data/sid_train_512.pt (388MB)
  #   data/sid_eval_512.pt  (23MB)
  #   exp/sid/multimodal_2176d/ (386MB)
  #   exp/sid/rqvae_2176d_d4_k256/ (codebook + model)
  # 或者服务器上重新跑 prepare_multimodal_embedding + build_rqvae_sid + pre_tokenize

  # 5. 训练
  python src/sid/train_sid_generator.py \
    --train_pt data/sid_train_512.pt \
    --eval_pt data/sid_eval_512.pt \
    --output_dir out/sid_generator \
    --preset 5090 --epochs 5

  GitHub 上只包含代码（总共 1200 行，不包括数据文件）。数据文件需要用 SCP/SFTP 单独传到服务器，或者在服务器上从头生成。

    训练完成后运行：
  python src/sid/sid_inference.py \
    --model_dir out/sid_generator \
    --sid_to_tracks exp/sid/rqvae_2176d_d4_k256/sid_to_tracks.json \
    --track_to_sid exp/sid/rqvae_2176d_d4_k256/track_to_sid.json \
    --out exp/inference/devset/sid_generator.json

  python src/evaluate.py \
    --inference exp/inference/devset/sid_generator.json \
    --scores exp/scores/devset/sid_generator.json \
    --ground_truth exp/ground_truth/devset.json


      # 1. 克隆
  git clone https://github.com/Goodnamee/music-rec.git && cd music-rec

  # 2. 装环境
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  pip install -r requirements.txt
  pip install flash-attn --no-build-isolation

  # 3. 预下载 Qwen3（首次）
  python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')"

  # 4. 上传数据文件（不在 git 中的大文件）
  #   data/sid_train_512.pt, data/sid_eval_512.pt
  #   exp/sid/multimodal_2176d/, exp/sid/rqvae_2176d_d4_k256/

  # 5. 配 git token（一次，后续自动）
  git config credential.helper store
  # 替换 YOUR_TOKEN 为 GitHub Classic Token（repo scope）
  git push https://Goodnamee:YOUR_TOKEN@github.com/Goodnamee/music-rec.git master

  # 6. 启动！关 SSH 不管
nohup bash scripts/train_eval_5090.sh > train.log 2>&1 &


  nvidia-smi
  # PyTorch with CUDA 12.8
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  # 然后装其他依赖
  pip install -r requirements.txt
  # Flash Attention 2（5090 支持）
  pip install flash-attn --no-build-isolation
  5XLcIV00JGyB

  解压：tar -xzf sid_data.tar.gz

## 推理评估记录 (2026-06-21)

训练设置：d4_3tok SIDs, depth=3, codebook=256, Qwen3-0.6B LoRA, ~161K 训练样本
推理：beam=20, constrained decoding, batch=1, max_new_tokens=8

| Checkpoint | Train Loss | Eval Loss | nDCG@1 | nDCG@10 | nDCG@20 | Hit@20 | Catalog Div |
|-----------|-----------|-----------|--------|---------|---------|--------|-------------|
| 500 | ~0.002 | 22.06 | 0.00013 | 0.00021 | 0.00027 | 0.06% | 0.042 |
| 2000 | ~0.0009 | 23.00 | 0 | 0.00019 | 0.00022 | 0.06% | 0.046 |

**结论：两个 checkpoint 都是随机水平。train loss 降到几乎 0，eval loss 始终 22+，严重过拟合。模型没学到从对话到 SID 的映射。**

## 诊断过程 (2026-06-22)

### 诊断脚本验证

用训练样本直接测试模型预测 vs 标签：

```
Label:   <a_4> <b_139> <c_62>
Predict: ighing sigh sigh sigh sigh sigh sigh
```

无约束解码时模型输出普通文本，完全没学会 SID。

### 关键发现：SID token embedding norm 过小

| Token 类型 | LM Head norm | Embedding norm |
|-----------|-------------|----------------|
| SID token (`<a_4>`) | 0.3750 | 0.3750 |
| SID token (`<b_139>`) | 0.3535 | 0.3535 |
| 普通文本 (`igh`) | 1.0859 | 1.0859 |
| 普通文本 (`the`) | 1.1562 | 1.1562 |

SID token 的向量长度比普通词小约 **3 倍**。

### 根因分析

`resize_token_embeddings` 默认 `mean_resizing=True`，新 token embedding 从旧分布（μ≈0, σ≈0.02）采样初始化，norm ≈ 0.02。训练 2500 步只涨到 0.35，远不到普通 token 的 1.0。

```
logit = hidden · lm_head_weight
SID logit = |hidden| × 0.35  vs  普通 logit = |hidden| × 1.0
```

方向完美对齐时 SID 分数也自动低 3 倍。softmax 后 SID token 排名跌到 88253 / 153717。

**约束解码（Constrained Decoding）确保输出格式合法，但不能让模型选对——选对靠训练时 SID embedding 学到语义，而 SID embedding 因 norm 太小根本没学到。**

### 对比 baseline

| 方法 | nDCG@1 |
|------|--------|
| BM25 | 0.014 |
| Random | 0.000125 |
| Popularity | 0.000125 |
| **SID Generator** | 0.000125 |

模型 = 随机水平。

## 解决方案

### ✅ 已修复：SID Embedding Norm 对齐

在 `train_sid_generator.py` 第 255 行之后添加：

```python
# scale new SID token embeddings to match existing token norms
with torch.no_grad():
    embed_w = model.get_input_embeddings().weight
    old_norm_mean = embed_w[:original_vocab_size].norm(dim=1).mean().item()
    for i in range(original_vocab_size, len(tokenizer)):
        cur_norm = embed_w[i].norm().item()
        if cur_norm > 0:
            embed_w[i] *= (old_norm_mean / cur_norm)
```

效果：SID token embedding 初始 norm ≈ 1.0，和普通词同一量级。

### 验证结果：Norm 修复未解决根本问题 (2026-06-25)

| Version | Eval Loss | nDCG@1 | nDCG@20 | SID Norm |
|---------|-----------|--------|---------|----------|
| 旧 no-fix ckpt-500 | 22.06 | 0.00013 | 0.00027 | 0.35 |
| 旧 no-fix ckpt-2000 | 23.00 | 0 | 0.00022 | 0.35 |
| norm-fix ckpt-500 | **37.76** | **0** | 0.00003 | **0.94** |
| norm-fix ckpt-2500 | 37.76 | **0** | 0.00011 | 0.94 |
| Random | — | 0.000125 | — | — |
| BM25 | — | 0.014 | — | — |

eval_loss 翻倍（22→37.76）—— norm=1.0 让模型自信地猜错，而不是不确定地猜。

## 实验记录：排查 SID Generator 无法学到 SID 映射的根因 (2026-06-22 ~ 06-25)

### 问题症状
训练收敛（train_loss ~0.0007），但推理+约束解码后 nDCG@1 = 0.000125，等于随机 baseline。

### 诊断实验设计框架

五个可测试的假设，从浅到深：

| # | 假设 | 测试方法 |
|---|------|---------|
| 1 | 架构/梯度有 bug，模型根本记不住 SID | 100 条过拟合 → 看 train loss 能否到 0 + 预测正确率 |
| 2 | 训练 softmax 分母 153717 vs 推理 256，分布不匹配 | 训练时约束 softmax 到 SID token 集合 |
| 3 | LoRA r=16 容量不够 | r=64/128 对比 |
| 4 | lm_head SID 行被排除了，间接更新不够 | optimizer 加入 lm_head SID 行 |
| 5 | SID 码本身不承载语义 | 检查相同 SID 前缀的 track 是否共享 artist/genre |

---

### 实验 1：过拟合测试（假设 1）—— 2026-06-25

**目的**：验证 LoRA + sid_embed + gradient hook 架构是否能正常工作。如果连 100 条都记不住，说明代码有 bug。

**方法**：
1. 从 `data/sid_train_512.pt` 取前 100 条，保存为 `data/sid_overfit_100.pt`
2. 用 `--preset test --epochs 50 --lr 1e-3` 训练（提高 lr、多 epoch 强制过拟合）
3. 训完用约束解码（beam=1）对训练集 100 条做推理
4. 比对 pred SID 是否等于 label SID

**命令**：
```bash
# 数据准备 (diagnose_tests.py)
torch.save(subset, "data/sid_overfit_100.pt")

# 训练
python src/sid/train_sid_generator.py \
  --train_pt data/sid_overfit_100.pt \
  --eval_pt data/sid_overfit_100.pt \
  --model_path ./Qwen3-0.6B \
  --output_dir out/sid_overfit \
  --preset test --epochs 50 --lr 1e-3

# 验证（手动加载模型 + 逐条生成比对）
```

**结果**：
```
train_loss: 2.066 → 0.049 (epoch 50)
Accuracy: 78/100 = 78%

Sample predictions:
[0] pred="<a_4> <b_139> <c_62>" | label="<a_4> <b_139> <c_62>" | OK
[1] pred="<a_125> <b_3> <c_100>" | label="<a_125> <b_3> <c_100>" | OK
```

**结论**：✅ 架构没有 bug。LoRA + sid_embed + 梯度路由都能正常工作。100 条中能记住 78 条（22 条难样本可能容量不够或 SID 冲突）。
**问题是泛化**，不是架构错误。→ 继续测假设 2。

---

### 实验 2：约束 softmax 训练（假设 2）—— 2026-06-25

**目的**：训练时 softmax 分母 153717 tokens，推理时 constrained decoding 只允许 ~770 个合法 token。两者分布不匹配可能导致训练学到"压制非 SID token"而不是"区分 SID token"。让训练 loss 只计算合法 token。

**方法**：
1. 修改 `LigerTrainer.compute_loss`（`train_sid_generator.py` 第 46-114 行）
2. 在训练时的 label 位置（labels != -100），隐藏状态只投影到 `valid_ids` 集合：
   - 所有 SID token (a_0~a_255, b_0~b_255, c_0~c_255 = 768 个实际使用的)
   - 空格 token（ID=220）
   - EOS token
   - 共 ~770 个合法 token
3. 在 constrained 路径：h_selected @ lm_head[valid_ids].T → logits [N, 770] → CE loss
4. Eval 和 Liger fuse CE 路径不受影响

**代码变更**（`train_sid_generator.py`）：
```python
# LigerTrainer.__init__ 增加:
self._sids_valid_ids = valid_ids   # set of legal token ids
self._sids_valid_list = sorted(valid_ids)
self._sids_idx_map = {tid: i for i, tid in enumerate(self._sids_valid_list)}

# compute_loss constrained 路径:
h_selected = hidden_states[label_mask]           # [N, D]
valid_w = lm_head_w[self._sids_valid_list]       # [K, D]
logits = h_selected @ valid_w.T                   # [N, K]
# map original label id → constrained index
loss = F.cross_entropy(logits, mapped_labels, ignore_index=-100)
```

**测试**：
```bash
# 同样 100 条过拟合，对比约束 vs 无约束
python src/sid/train_sid_generator.py \
  --train_pt data/sid_overfit_100.pt \
  --eval_pt data/sid_overfit_100.pt \
  --model_path ./Qwen3-0.6B \
  --output_dir out/sid_overfit_cs \
  --preset test --epochs 50 --lr 1e-3
```

**结果**：
```
              Train Loss    100条准确率
无约束 softmax   0.049        78%
约束 softmax     0.029        78%

Loss 更低（-41%），但准确率天花板相同。
约束版决策更精准（logit 分布更集中），22 条"难样本"两者都记不住。
```

**结论**：约束 softmax 降低 loss 但未突破准确率天花板。需要在全量数据上验证泛化效果。→ 服务器重训测试。

---

### 实验 3：SID 语义检查（假设 5）—— 2026-06-25

**目的**：验证 RQ-VAE 产生的 SID 码是否真正按"音乐氛围"聚类。如果 SID 是随机的，模型永远学不到。

**方法**：
1. 加载 track metadata（HF `talkpl-ai/TalkPlayData-Challenge-Track-Metadata`）
2. 对每个 SID 前缀（`<a_X>`），收集其下 track 的 artist
3. 随机采样 10000 对 SID：同前缀 vs 不同前缀
4. 比较 artist 重合度（Jaccard overlap）
5. 如果同前缀 artist 重合 > 不同前缀 × 1.5，则 SID 有语义

**命令**：
```python
# diagnose_tests.py Test 5 部分
# 关键逻辑：
for _ in range(10000):
    s1, s2 = random.choice(sids)
    p1, p2 = s1.split()[0], s2.split()[0]   # <a_X> prefix
    # get artists for tracks under each SID
    artists1 = {track_artist[tid] for tid in sid_to_tracks[s1][:3]}
    artists2 = {track_artist[tid] for tid in sid_to_tracks[s2][:3]}
    overlap = len(artists1 & artists2) / len(artists1 | artists2)
    # group by same/diff prefix
```

**结果**：
```
Same prefix (<a_X>) artist overlap: 0.0000 (42 pairs)
Diff prefix artist overlap:         0.0007 (9942 pairs)
Depth-2 (<a_X> <b_Y>) same:         0.0000 (0 matching pairs)

Metadata 问题: genres 字段所有 47071 条均为空
```

**结论**：SID 码**不按 artist 聚类**（0% 重合）。RQ-VAE 按 multimodal embedding
（attributes-qwen3 + lyrics-qwen3 + cf-bpr，共 2176d）聚类，artist 不是明确维度。
SID 空间 256³ = 16M，46K tracks 极其稀疏（每 SID ~0.003 条 track）。
对话中用户表达的"音乐偏好"（如 "intense dramatic"）能否映射到 SID 空间，
取决于 RQ-VAE 编码是否保留了这些维度——这无法从 metadata 直接验证。

---

### 实验 0：SID Embedding Norm 修复 —— 2026-06-22

**目的**：修复 `resize_token_embeddings(mean_resizing=True)` 导致 SID token 向量长度
比普通文本 token 小 ~3 倍的问题。

**发现过程**：
```python
# diagnose.py 诊断脚本
# 加载 checkpoint-500 → 测一条训练样本
# 预测: "ighing sigh sigh sigh..."  ← 普通文本！
# 正确: "<a_4> <b_139> <c_62>"

# 检查 SID token logit 排名：
correct token <a_4>: rank 88253/153717, prob=0.000000
top-5: ['igh', 'aying', 'LOT', 'ear', 'arc']  ← 全是普通词

# 检查 norm：
SID token lm_head norm: 0.35
普通词 lm_head norm:    1.09
```

**根因**：`mean_resizing=True` 对每维独立采样，丢失了旧词内部的维度间相关性。
舊词靠少数大值维度撑起 norm，独立采样均匀撒到 1024 维，norm 从 ~1.0 跌到 ~0.3。

**修复**（`train_sid_generator.py` 第 256-267 行）：
```python
with torch.no_grad():
    embed_w = model.get_input_embeddings().weight
    old_norm_mean = embed_w[:original_vocab_size].norm(dim=1).mean().item()
    for i in range(original_vocab_size, len(tokenizer)):
        cur_norm = embed_w[i].norm().item()
        if cur_norm > 0:
            embed_w[i] *= (old_norm_mean / cur_norm)
```

**全量训练验证**：
```
旧版 no-fix: SID norm 0.35, eval_loss 22.06, nDCG@1=0.00013
新版 norm-fix: SID norm 0.94, eval_loss 37.76, nDCG@1=0
```
Norm 修复生效，但 eval_loss 翻倍（模型自信猜错），nDCG 未改善。
→ 问题不在 norm，需要继续排查。

---

### 辅助发现与修复

**A. 训练数据对齐 bug**（`build_training_data.py` 第 91-99 行）
当 track 无 SID 映射时，`all_user_texts` 已添加但 `prev_rec_sids` 未添加，
导致 `zip()` 将错误的历史 SID 与后续 turn 的用户文本配对。
**修复**：跳过时不添加 user_text，保持上下文对齐。
commit: `ad050c1` Fix training data alignment bug

**B. 推理增量保存**（`sid_inference.py` 第 293-325 行）
原始实现只在最后一次性写 JSON，电脑睡眠/崩溃丢全部进度。
**修复**：每 50 条增量保存，启动时检测已有结果 → 断点续传。
commit: `e9340b8`

---

### 当前状态总结

| 已排除 | 证据 |
|--------|------|
| 架构 bug | 过拟合 100 条可达 78% 准确率 |
| Norm 问题 | 修复后 norm=0.94，但仍然预测错误 |
| SID 语义 | RQ-VAE 按 multimodal embedding 聚类，不按 artist |

| 已验证，效果未定 | 证据 |
|------------------|------|
| 约束 softmax | 过拟合 loss 更低，全量数据泛化效果待测 |

| 未测试 | 内容 |
|--------|------|
| 假设 3 | LoRA r=16 容量 → r=64/128 |
| 假设 4 | lm_head SID 行未被优化器直接更新 |

### 下一步

1. 服务器全量训练（含 norm fix + 约束 softmax + 数据对齐修复）
2. 下载推理+评估，对比旧版
3. 如果仍然随机 → 测假设 3（增大 LoRA rank）+ 假设 4（lm_head 参与训练）