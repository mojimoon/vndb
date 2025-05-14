# VNDB PONet

适用于 [Visual Novel Database](https://vndb.org/) (VNDB) 的基于偏序网络 (Partial Order Network, PONet) 的排名算法。

目前正在建设中。

## 背景

基于偏序网络的排名算法最初由 [@eyecandy](https://bgm.tv/user/eyecandy) 提出，其核心思想是 **基于同一用户对不同作品的评分来判断作品之间的相对优劣**，而不再关注作品本身的评分分布。简而言之，PONet 认为如果在有很多人同时给作品 A 和作品 B 评分的情况下，大多数人 (>50%) 认为 A 比 B 更好，那么我们就可以认为 A 比 B 更好。

- 2022 年，@eyecanday 以 Bangumi 动画区进行了实验，并将结果发布在 [Bangumi 讨论区](https://bgm.tv/group/topic/371075)。
- 2023 年，我用 [Bangumi15M](https://www.kaggle.com/datasets/klion23/bangumi15m) 数据集重新计算了动画区的 PONet 排名，仓库在 [mojimoon/bangumi-anime-ranking](https://github.com/mojimoon/bangumi-anime-ranking)，结果也发布在 [Bangumi 讨论区](https://bgm.tv/group/topic/382497)。
- 然而，由于从 Bangumi API 获取大规模数据集不便且存在访问限制，且计算耗时较长，因此以上排名一直没有得到更新。所幸，VNDB 提供了每日更新的 [database dump](https://vndb.org/d14)，使得定期更新排名成为可能，因此我决定将 PONet 算法应用于 VNDB 数据库。

## 算法简述

1. 构建偏序网络。对每一对作品 A 和 B，如果有 n 名用户同时对 A 和 B 进行了评分，其中 x 人认为 A 比 B 更好，y 人认为 A 比 B 差，则：

- 定义 A 对 B 的「合计积分（Total Score）」为 (x-y) 分；
- 定义 A 对 B 的「比例积分（Percentage Score）」为 (x-y)/n 分，这个分数接近于科学排名中的「倾向性概率」；
- 定义 A 对 B 的「简易积分（Simple Score）」为 sgn(x-y) 分，其中 sgn 表示符号函数（即正数取值为 1，负数取值为 -1，0 取值为 0）。

2. 将偏序网络转化为全序网络。针对每一个作品 A，将它与其它每一部作品 B1, B2, ... 进行比较，将 A 相对 B1, B2, ... 的积分进行平均，得到 A 的最终得分。

此外还有两处 hyperparameter：

- `min_vote = 30`：仅当作品 A 的评分人数 `>= min_vote` 时，才会进入这个排名系统。
- `min_common_vote = 3`：仅当 A 和 B 的共同评分人数 `>= min_common_vote` 时，才会计算 A 对 B 的积分。否则，A 对 B 的积分为 0 分，且在计算平均值时直接忽略。

此处选择 `min_vote = 30` 是因为在 VNDB 的评分制度下，>= 30 票的作品不使用贝叶斯平均（而是使用简单平均），且只有 >= 30 票的作品才能进入 Top 50。

## 使用方法

1. 下载 database dump，解压缩到 `db` 目录下。

```bash
curl -L -o db.tar.zst https://dl.vndb.org/dump/vndb-db-latest.tar.zst
mkdir -p db
tar -I zstd -xf db.tar.zst -C db/
rm db.tar.zst
```

2. 安装依赖。

```bash
pip install -r requirements.txt
```

3. 运行 `main.py`。

```bash
python main.py
```
