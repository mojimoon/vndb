# VNDB PONet

适用于 [Visual Novel Database](https://vndb.org/) (VNDB) 的基于偏序网络 (Partial Order Network, PONet) 的排名算法。

目前正在建设中。

## 背景

基于偏序网络的排名算法最初由 [@eyecandy](https://bgm.tv/user/eyecandy) 提出，其核心思想是**基于同一用户对不同作品的评分来判断作品之间的相对优劣**，而不再关注作品本身的评分分布。

借此，PONet 可以通过简单高效的算法给出作品的相对排名，无需利用贝叶斯平均等复杂的统计方法，作为 [科学排名](https://chii.ai/rank) 的补充。

简而言之，PONet 认为如果在有很多人同时给作品 A 和作品 B 评分的情况下，大多数人 (>50%) 认为 A 比 B 更好，那么我们就可以认为 A 比 B 更好。

- 2022 年，@eyecanday 以 Bangumi 动画区进行了实验，并将结果发布在 [Bangumi 讨论区](https://bgm.tv/group/topic/371075)。
- 2023 年，我用 [Bangumi15M](https://www.kaggle.com/datasets/klion23/bangumi15m) 数据集重新计算了动画区的 PONet 排名，仓库在 [mojimoon/bangumi-anime-ranking](https://github.com/mojimoon/bangumi-anime-ranking/tree/main/ponet)，结果也发布在 [Bangumi 讨论区](https://bgm.tv/group/topic/382497)。
- 然而，由于从 Bangumi API 获取大规模数据集不便且存在访问限制，上述排名长期没有得到更新。
- 所幸，VNDB 提供了每日更新的 [database dump](https://vndb.org/d14)，使得定期更新排名成为可能，因此我决定将 PONet 算法应用于 VNDB 数据库。

## 算法简述

1. 构建偏序网络。对每一对作品 A 和 B，如果有 n 名用户同时对 A 和 B 进行了评分，其中 x 人认为 A 比 B 更好，y 人认为 A 比 B 差，则：

- 定义 A 对 B 的「合计积分（Total Score）」为 (x-y) 分；
- 定义 A 对 B 的「比例积分（Percentage Score）」为 (x-y)/n 分，这个分数接近于科学排名中的「倾向性概率」；
- 定义 A 对 B 的「简易积分（Simple Score）」为 sgn(x-y) 分，其中 sgn 表示符号函数（即正数取值为 1，负数取值为 -1，0 取值为 0）。

2. 将偏序网络转化为全序网络。针对每一个作品 A，将它与其它每一部作品 B1, B2, ... 进行比较，将 A 相对 B1, B2, ... 的积分进行平均，得到 A 的最终得分。

此外还有两处 hyperparameter：

- `min_vote = 30`：仅当作品 A 的评分人数 `>= min_vote` 时，才会进入这个排名系统。
- `min_common_vote = 5`：仅当 A 和 B 的共同评分人数 `>= min_common_vote` 时，才会计算 A 对 B 的积分，否则忽略这对作品。

此处选择 `min_vote = 30` 是由于 VNDB 只有 >= 30 票的作品才能进入 Top 50。

### 要不要试试科学排名？以及更多

实际上，构建偏序网络过程中记录的数据 `(A, B, x, y, n)` 也可以用于科学排名（详见 [科学排名原博客](https://ikely.me/2016/02/05/%E4%BD%BF%E7%94%A8-rankit-%E6%9E%84%E5%BB%BA%E6%9B%B4%E7%A7%91%E5%AD%A6%E7%9A%84%E6%8E%92%E5%90%8D/) 和 [rankit 项目](https://github.com/wattlebird/ranking)，因此，在这个项目中也进行了科学排名的实现。

此外，同样在构建偏序网络过程中，可以计算出 sample percentile（样本百分位数），其不关心用户 C 具体的评分分布，而是将其转化为一个 0-100% 的百分位数，表示某个具体分数 t 在 C 的所有评分中所处的百分位数。具体来说，

- 假设用户 C 总共给出了 $n$ 个评分 $\{x_1, x_2, \ldots, x_n\} (x_1 < x_2 < \ldots < x_n)$。
- 将整个标准化的取值范围划分为 $n+1$ 个区间 $(-\infty, x_1), (x_1, x_2), \ldots, (x_{n-1}, x_n), (x_n, +\infty)$。假设随机变量落在每个区间的概率相等，都是 $\frac{1}{n+1}$。因此，计算 $P(x \leq x_k)$ 即可得出 $x_k$ 的百分位数。
- 对于每一个具体的评分 $x_k$，如有多个相同评分，将其视为一个整体，其百分位数取值为该区间的中点。

得出计算公式：

$$\text{sp}(x_k) = \frac{(|\{x_i | x_i < x_k\}| + 0.5 \cdot |\{x_i (i \neq k) | x_i = x_k\}| + 1)}{n + 1}$$

其中 $|\{x_i | x_i < x_k\}|$ 表示小于 $x_k$ 的评分数量。这个项目中也尝试了使用 sample percentile 来进行排名。

## 使用方法

本项目的代码分为两个部分，数据使用 [Supabase](https://supabase.com/) 存储。

### 数据处理

见 [run.sh](run.sh)。使用前需要先添加执行权限：

```bash
chmod +x run.sh
```

1. 下载 database dump，解压缩到 `db` 目录下。

```bash
curl -L -o db.tar.zst https://dl.vndb.org/dump/vndb-db-latest.tar.zst
mkdir -p db
tar -I zstd -xf db.tar.zst -C db/
rm db.tar.zst
```

注意：在 Windows PowerShell 中，`curl` 是 `Invoke-WebRequest` 的别名，因此需要使用 `curl.exe` 来调用 curl。

```powershell
curl.exe -L -o db.tar.zst https://dl.vndb.org/dump/vndb-db-latest.tar.zst
```

此外，`tar.exe` 无法解压缩 `.tar.zst` 文件，请使用 [7-Zip](https://www.7-zip.org/) 等工具进行解压缩。

2. 复制 `dev/.env.example` 为 `dev/.env`，并填入 Supabase 环境变量。可以在官网 [这一页](https://supabase.com/docs/guides/getting-started/quickstarts/nextjs) 一键获取。

3. 安装依赖、运行脚本。

```bash
cd dev
pip install -r requirements.txt
python main.py
```

### 前端展示

框架：React TypeScript + Next.js 全栈框架。推荐使用 [pnpm](https://pnpm.io/) 进行包管理。

同样地，复制 `www/.env.example` 为 `www/.env.local`，并填入 Supabase 环境变量。

```bash
cd www
pnpm install
pnpm run dev
```
