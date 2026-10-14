# VNDB SciRanking

[English](README.md) | 中文

适用于 [Visual Novel Database](https://vndb.org/)（VNDB）的基于偏序网络（Partial Order Network, PONet）的排名算法 + [科学排名](https://ikely.me/2016/02/05/%E4%BD%BF%E7%94%A8-rankit-%E6%9E%84%E5%BB%BA%E6%9B%B4%E7%A7%91%E5%AD%A6%E7%9A%84%E6%8E%92%E5%90%8D/)。数据每天从 VNDB 数据库转储自动更新，网站运行在 Cloudflare Workers + D1 上。

## 背景

基于偏序网络的排名算法最初由 [@eyecandy](https://bgm.tv/user/eyecandy) 提出，其核心思想是**基于同一用户对不同作品的评分来判断作品之间的相对优劣**，而不再关注作品本身的评分分布。

借此，PONet 可以通过简单高效的算法给出作品的相对排名，无需利用贝叶斯平均等复杂的统计方法，作为 [科学排名](https://chii.ai/rank) 的补充。简而言之，如果很多人同时给作品 A 和作品 B 评了分，其中大多数人（>50%）认为 A 比 B 更好，那么就认为 A 比 B 更好。

- 2022 年，@eyecandy 以 Bangumi 动画区进行了实验，并将结果发布在 [Bangumi 讨论区](https://bgm.tv/group/topic/371075)。
- 2023 年，我用 [Bangumi15M](https://www.kaggle.com/datasets/klion23/bangumi15m) 数据集重新计算了动画区的 PONet 排名（[mojimoon/bangumi-anime-ranking](https://github.com/mojimoon/bangumi-anime-ranking/tree/main/ponet)，[讨论帖](https://bgm.tv/group/topic/382497)）。
- 由于从 Bangumi API 获取大规模数据集不便，上述排名长期没有更新。所幸 VNDB 提供每日更新的 [数据库转储](https://vndb.org/d14)，使定期更新排名成为可能，因此本项目将 PONet 算法应用于 VNDB。

## 算法简述

1. **构建偏序网络。** 对每一对作品 A 和 B，设有 n 名用户同时评分，其中 x 人认为 A 更好、y 人认为 B 更好，则 A 对 B 的：
   - **合计积分（Total Score）** 为 x − y；
   - **比例积分（Percentage Score）** 为 (x − y) / n，接近科学排名中的「倾向性概率」；
   - **简易积分（Simple Score）** 为 sgn(x − y)，即 1、0 或 −1。
2. **转化为全序。** 对每部作品，把它对其他所有作品的积分取平均，得到最终得分。

两个参数：

- `min_vote = 30`：评分人数 ≥ 30 的作品才进入排名（与 VNDB Top 50 的门槛相同）。
- `min_common_vote = 5`：两部作品的共同评分人数 ≥ 5 时才计入这对作品。

### 科学排名以及更多

实际上，构建偏序网络过程中记录的五元组 `(A, B, x, y, n)` 可以作为科学排名的倾向性概率使用（详见 [科学排名原博客](https://ikely.me/2016/02/05/%E4%BD%BF%E7%94%A8-rankit-%E6%9E%84%E5%BB%BA%E6%9B%B4%E7%A7%91%E5%AD%A6%E7%9A%84%E6%8E%92%E5%90%8D/) 和 [rankit 项目](https://github.com/wattlebird/ranking)。因此，这个项目也将科学排名囊括在内。

此外，上述过程同时可以计算出 sample percentile（样本百分位数），其不关心用户 C 具体的评分分布，而是将其转化为一个 0-100% 的百分位数，表示某个具体分数 t 在 C 的所有评分中所处的百分位数。具体来说，

- 假设用户 C 总共给出了 $n$ 个评分 $\{x_1, x_2, \ldots, x_n\} (x_1 < x_2 < \ldots < x_n)$。
- 将整个标准化的取值范围划分为 $n+1$ 个区间 $(-\infty, x_1), (x_1, x_2), \ldots, (x_{n-1}, x_n), (x_n, +\infty)$。假设随机变量落在每个区间的概率相等，都是 $\frac{1}{n+1}$。因此，计算 $P(x \leq x_k)$ 即可得出 $x_k$ 的百分位数。
- 对于每一个具体的评分 $x_k$，如有多个相同评分，将其视为一个整体，其百分位数取值为该区间的中点。

得出计算公式：

$$\text{sp}(x_k) = \frac{(|\{x_i | x_i < x_k\}| + 0.5 \cdot |\{x_i (i \neq k) | x_i = x_k\}| + 1)}{n + 1}$$

其中 $|\{x_i | x_i < x_k\}|$ 表示小于 $x_k$ 的评分数量。这个项目中也尝试了使用 sample percentile 来进行排名。

共有 41 种排名方法。方法的取舍与综合排名的实验记录见 [docs/ranking-experiments.md](docs/ranking-experiments.md)（英文）。

## 功能

- **排行**：41 种方法任选，可并排对比最多 3 种方法；自选显示列、导出 CSV；按开发商、年份、语言、长度、票数、VNDB 评分筛选。另有「年度最佳」「与 VNDB 的分歧」「排名变动」「历史」（任选一天查看当天的排名及之后的变化）。
- **作品页**：概览、可筛选评分者的评分分析、各方法排名、排名历史与 VNDB 评分 / 票数走势、与任意作品的正面交锋、相似作品、短评（公开列表备注）、TF-IDF 标签分析。
- **开发商**：开发商排行（作品数、最佳 / 中位排名、平均评分）、开发商页面与开发商之间的对比。
- **用户**：用户排行、口味分析、全部评分、基于相似用户的推荐、相似用户、短评、标签偏好。在已排名作品上至少有 5 个评分的用户才有页面。
- **对比**：两部作品（正面交锋、评分分布、共同评分者的评分矩阵、两部都写了短评的用户）或两位用户（原始分 / 百分位散点图、分歧、共同喜欢的作品）。
- 中文、英文、日文界面；浅色与深色主题。

## 使用方法

使用仓库自带的示例数据（非真实数据）在本地预览网站（需要 Node.js 22+）：

```bash
cd web
npm install
npm run db:seed:local
npm run dev                          # http://localhost:5173
```

用真实转储运行数据管线（需要 Python 3.11+ 和数 GB 内存）：

```bash
curl -L -o db.tar.zst https://dl.vndb.org/dump/vndb-db-latest.tar.zst
mkdir -p db && tar -I zstd -xf db.tar.zst -C db && rm db.tar.zst

cd pipeline
pip install -r requirements.txt
python -m vndb_rank --dump ../db --out out
npx --prefix ../web wrangler d1 execute vndb --local --file out/snapshot.sql
```

自行部署并每日自动更新，见 [docs/deployment.md](docs/deployment.md)（英文）。

## 文档（英文）

- [docs/deployment.md](docs/deployment.md)：Cloudflare 配置、GitHub Actions、本地开发
- [docs/architecture.md](docs/architecture.md)：目录结构、数据流、数据库设计、免费额度估算
- [docs/ranking-experiments.md](docs/ranking-experiments.md)：排名方法与综合排名的实验

## 许可

数据来自 [VNDB 数据库转储](https://vndb.org/d14)，依 ODbL 授权。
