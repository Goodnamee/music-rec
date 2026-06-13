# Label Semantics
说明：
点击/播放是用户主动行为，虽然有噪声，但可以作为弱正反馈。
Music-CRS 中的 music target 更接近 benchmark target / teacher policy label。
因此本项目不声称学习真实满意度。
为了增强用户偏好建模，引入 CF-BPR user/item embeddings 和历史 track 信息。
最终评估仍按 benchmark 的 nDCG@20 / Hit@20，同时报告限制。
== 相当于自回归？根据已有序列预测下一个序列 ==

这个项目的 RQ-VAE 是一个 商品语义 ID 构造器。它先把商品文本转成连续 embedding，再用多级残差向量量化把 embedding 离散化成 SID。后续 LLM 推荐模型不直接预测商品编号，而是像生成文本一样生成 <a_*> <b_*> <c_*> 这些语义 token。优点是把推荐问题转成 LLM 更容易处理的生成问题；缺点是 SID 质量高度依赖 embedding、RQ-VAE 训练和 collision 控制。

# target
● 已整理并保存。以下是完整总结：

  ---
  项目方向：生成式音乐推荐 Agent

  核心问题：竞赛数据是 Gemini 生成的模拟对话，用这些对话训练模型就是在模仿 Gemini。目标是把推荐系统改成一个完整 Agent——自己对话、自己推荐，闭环在自己手里。

  Agent 架构

  用户 → Chat LLM (对话管理) → SID Generator (对话→语义token) → Codebook解码 → 个性化重排 → 回复

  核心模块只有两个：
  - Chat LLM：管理对话节奏，现成大模型直接用
  - SID Generator：对话文本 → SID tokens，需要自己训

  不需要 State Extractor，SID Generator 隐式完成了偏好提取。

  SID 构造

  用 attributes-qwen3 + lyrics-qwen3 + cf-bpr 三模拼接 (2176d)，训 RQ-VAE。attributes+lyrics 给内容语义，cf-bpr 给行为语义，互补且不冗余。

  训练数据

  从竞赛数据只取 role=="user" 的文本 + ground truth track 映射成 SID label。role=="music" 的 Gemini 推荐轨迹丢掉不用。

  两阶段对齐

  RQ-VAE 和 seq2seq 独立训练有 gap。在 seq2seq decoder hidden state 上加 InfoNCE 对比损失（anchor=gold SID, negatives=batch 内其他 SID），不重训 RQ-VAE，工程简单且有效。

  Playlist 信号

  三步：① cf-bpr 已在 SID 里；② playlist embedding pooling 可选做 user profile 向量塞进 seq2seq 输入；③ SID 召回后 playlist embedding 重排（最保守的先做）。

  执行路线

  ┌─────────┬───────────────────────────────────────────────────┬─────────┐
  │  阶段   │                       内容                        │  时间   │
  ├─────────┼───────────────────────────────────────────────────┼─────────┤
  │ Phase 1 │ 三模态 SID 构建 + RQ-VAE 训练                     │ ~1 天   │
  ├─────────┼───────────────────────────────────────────────────┼─────────┤
  │ Phase 2 │ SID Generator seq2seq 训练 + constrained decoding │ ~2-3 天 │
  ├─────────┼───────────────────────────────────────────────────┼─────────┤
  │ Phase 3 │ Agent 组装（Chat LLM + SID 链路串联）             │ ~1 天   │
  ├─────────┼───────────────────────────────────────────────────┼─────────┤
  │ Phase 4 │ Dev 评测 + 对齐迭代                               │ 持续    │
  └─────────┴───────────────────────────────────────────────────┴─────────┘

  ---