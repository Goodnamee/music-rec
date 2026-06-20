关键教训：把 2176d 直接 KMeans 量化 = 坍塌。加 MLP 压到 64d + EMA = 几乎是免费的（1 分钟），准确率从 16% 跳到 99.8%。

样本的构造

constraint SID


SID训练损失是SID accuracy，有多少<SID>命中。不是，语言模型监督学习

nohup bash scripts/train_eval_5090.sh > train.log 2>&1 &
tail -f train.log
flash SDP
minionerec在数据、代码方面有什么加速计、优化的地方吗 
# 或者直接全部杀掉重来
pkill -f train_sid_generator
  
flash attn 并非瓶颈 5090没有预编译

工程加速：
liger-kernel 没用不适配qwen？ply_liger_kernel_to_qwen2 不兼容 Qwen3 的 forward 路径——patch 返回了（节省很多显存）
PeftModel 的层级多了一层。
hidden_states 是 3D [batch, seq, hidden]，liger 要 2D [batch*seq, hidden]。flatten 一下：
用之前节省的显存把ckpt关闭
flash SDP（取消）
cosine scheduler
20*5

新增sid参数冻结了，根本没训练.
必须一起传入，然后掩码，计算量没变
留梯度 hook 让 autograd 算梯度，但优化器跳过 embedding，新 LoRA + 一个独立的 256 行参数。（自定义优化器）
瓶颈在liger 的 lm_head 梯度计算。没用。还是5s/it