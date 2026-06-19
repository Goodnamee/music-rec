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