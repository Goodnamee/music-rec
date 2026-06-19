关键教训：把 2176d 直接 KMeans 量化 = 坍塌。加 MLP 压到 64d + EMA = 几乎是免费的（1 分钟），准确率从 16% 跳到 99.8%。

样本的构造

constraint SID


SID训练损失是SID accuracy，有多少<SID>命中。不是，语言模型监督学习

nohup bash scripts/train_eval_5090.sh > train.log 2>&1 &
tail -f train.log

# 或者直接全部杀掉重来
pkill -f train_sid_generator
  
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

wget "https://ghproxy.net/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch26cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" -O flash_attn.whl && pip install flash_attn.whl

工程加速：
liger-kernel