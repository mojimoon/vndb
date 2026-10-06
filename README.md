# VNDB Ranking+

适用于 [Visual Novel Database](https://vndb.org/) (VNDB) 的基于偏序网络 (Partial Order Network, PONet) 的排名算法 + [科学排名](https://ikely.me/2016/02/05/%E4%BD%BF%E7%94%A8-rankit-%E6%9E%84%E5%BB%BA%E6%9B%B4%E7%A7%91%E5%AD%A6%E7%9A%84%E6%8E%92%E5%90%8D/)。

数据每天从 VNDB 数据库转储自动更新，网站运行在 Cloudflare Workers + D1 上。

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

实际上，构建偏序网络过程中记录的五元组 `(A, B, x, y, n)` 可以作为科学排名的倾向性概率使用（详见 [科学排名原博客](https://ikely.me/2016/02/05/%E4%BD%BF%E7%94%A8-rankit-%E6%9E%84%E5%BB%BA%E6%9B%B4%E7%A7%91%E5%AD%A6%E7%9A%84%E6%8E%92%E5%90%8D/) 和 [rankit 项目](https://github.com/wattlebird/ranking)。因此，这个项目也将科学排名囊括在内。

此外，上述过程同时可以计算出 sample percentile（样本百分位数），其不关心用户 C 具体的评分分布，而是将其转化为一个 0-100% 的百分位数，表示某个具体分数 t 在 C 的所有评分中所处的百分位数。具体来说，

- 假设用户 C 总共给出了 $n$ 个评分 $\{x_1, x_2, \ldots, x_n\} (x_1 < x_2 < \ldots < x_n)$。
- 将整个标准化的取值范围划分为 $n+1$ 个区间 $(-\infty, x_1), (x_1, x_2), \ldots, (x_{n-1}, x_n), (x_n, +\infty)$。假设随机变量落在每个区间的概率相等，都是 $\frac{1}{n+1}$。因此，计算 $P(x \leq x_k)$ 即可得出 $x_k$ 的百分位数。
- 对于每一个具体的评分 $x_k$，如有多个相同评分，将其视为一个整体，其百分位数取值为该区间的中点。

得出计算公式：

$$\text{sp}(x_k) = \frac{(|\{x_i | x_i < x_k\}| + 0.5 \cdot |\{x_i (i \neq k) | x_i = x_k\}| + 1)}{n + 1}$$

其中 $|\{x_i | x_i < x_k\}|$ 表示小于 $x_k$ 的评分数量。这个项目中也尝试了使用 sample percentile 来进行排名。

## 架构

```
VNDB dump ──► pipeline/ (Python) ──► snapshot.sql ──► Cloudflare D1 ──► web/worker (Hono API) ──► web/src (React SPA)
             每日 12:07 UTC，GitHub Actions                              边缘缓存，按快照失效
```

| 目录 | 内容 |
| --- | --- |
| `pipeline/` | 数据管线：读取转储、构建偏序网络、计算 57 种排名、导出 D1 快照 SQL |
| `web/` | Cloudflare Worker（API + 静态资源）与 React 前端；`web/migrations/` 为 D1 表结构 |
| `.github/workflows/` | `refresh.yml` 每日更新数据，`deploy.yml` 部署网站，`ci.yml` 测试，`dump-probe.yml` 检查转储表结构 |
| `research/` | 旧版脚本、评论情感分类实验（DistilBERT / Transformer）和 playground |
| `docs/architecture.md` | 数据库设计与 D1 免费额度估算（英文） |

### 为什么从 Supabase 换到 D1 之后放得下

旧方案把逐用户评分（`ulist`，数百万行）整个上传到数据库。新方案中原始评分和 N² 的作品对矩阵只在离线管线里计算，数据库只保存网站需要读取的结果，并且按「一次读取一行」打包：

| 表 | 行数（2026-10 转储） | 内容 |
| --- | --- | --- |
| `vn` | 7,945 | 每部作品一行：57 种方法的排名、评分分析、正面交锋、相似作品、排名历史（JSON 列） |
| `producer` | 2,410 | 开发商 |
| `pair_block` | 7,948 | 全部 489 万对可比较作品的胜负数据，每对 8 字节二进制，任意两部作品都能精确对比 |
| `user_block` | 2,048 | 55,778 位用户的评分、相似用户和推荐，按 `uid % 2048` 分片 |
| `user_name` | 64 | 用户名 → uid 索引 |
| `meta` | 4 | 快照编号、方法列表、统计数据、方法一致性矩阵 |

每日更新共写入约 2 万行（D1 免费额度为每天 10 万行），数据库预计约 100 MB（免费额度单库 500 MB）。导入脚本先建好新表、最后一次性替换旧表，读者不会看到新旧混合的数据。详见 [docs/architecture.md](docs/architecture.md)。

### 功能

- **排行**：57 种方法任选，可并排对比最多 3 种方法、自选显示列、导出 CSV；按语言、长度、年份、票数、开发商筛选；「年度最佳」「与 VNDB 的分歧」「排名变动」分页。
- **作品页**：概览、评分分析（分布、每年评分、列表状态、评分者心中的位置、评分者偏好）、各方法排名与排名历史、与任意作品的正面交锋、「喜欢它的人也喜欢」。
- **用户页**：评分统计与口味分析（与 VNDB 评分的相关性、比大家更喜欢/更不喜欢的作品）、全部评分、基于相似用户的推荐、相似用户。只收录在进入排名的作品上至少有 5 个评分的用户。
- **对比**：任意两部作品或两位用户并排对比。

## 使用方法

### 1. 一次性配置 Cloudflare

需要 Node.js 22+。

```bash
cd web
npm install
npx wrangler login
npx wrangler d1 create vndb          # 把输出的 database_id 填入 web/wrangler.jsonc
npm run deploy                       # 部署 Worker 和前端；表结构由每日导入的快照创建
```

自定义域名：在 `web/wrangler.jsonc` 中取消注释 `routes` 并填入域名（域名需已托管在 Cloudflare），或在 Cloudflare 控制台的 Worker → Settings → Domains & Routes 中添加。注意 Workers 的边缘缓存（Cache API）只在自定义域名上生效，`*.workers.dev` 上每次请求都会读数据库。

### 2. 配置自动更新（GitHub Actions）

在 Cloudflare 创建 API Token：控制台右上角头像 → **My Profile → API Tokens → Create Token**，选择 **Edit Cloudflare Workers** 模板，再点 **+ Add more** 增加一条权限 **Account · D1 · Edit**；Account Resources 选择你的账户，Zone Resources 选择你的域名（或 All zones）。

然后在 GitHub 仓库 **Settings → Secrets and variables → Actions** 中添加：

| Secret | 值 |
| --- | --- |
| `CLOUDFLARE_API_TOKEN` | 上面创建的 Token |
| `CLOUDFLARE_ACCOUNT_ID` | 控制台 Workers & Pages 页面右侧的 Account ID |

之后：

- `refresh.yml` 每天 12:07 UTC（VNDB 约在 08:00 UTC 发布转储）下载转储、重新计算并导入 D1；也可以在 Actions 页面手动运行。定时任务只在默认分支上触发。
- `deploy.yml` 在 `main` 分支的 `web/` 有改动时自动部署。

### 3. 本地开发

前端（使用仓库自带的**合成**示例数据，不是真实 VNDB 数据）：

```bash
cd web
npm install
npm run db:seed:local
npm run dev                          # http://localhost:5173
```

数据管线（需要 Python 3.11+；完整计算约需数 GB 内存）：

```bash
curl -L -o db.tar.zst https://dl.vndb.org/dump/vndb-db-latest.tar.zst
mkdir -p db && tar -I zstd -xf db.tar.zst -C db && rm db.tar.zst

cd pipeline
pip install -r requirements.txt
python -m vndb_rank --dump ../db --out out          # 约 10 分钟；--skip-rankit / --skip-users 可跳过耗时步骤
npx --prefix ../web wrangler d1 execute vndb --local --file out/snapshot.sql   # 导入本地 D1
python -m pytest                                     # 测试（使用合成数据）
```

## 与旧版的差异

重写时修复了旧版 `research/legacy_main.py` 中的几个问题，因此结果与旧版不完全相同：

- **作品对计数丢失约一半**：旧版按评分顺序给作品编号，但按 vid 顺序遍历用户列表，约一半的比较落在矩阵下三角，导出时只读取上三角而被丢弃。
- rankit 的三个 Markov 变体会原地修改共享的输入表，导致之后的 OD、Difference 等方法使用了被改动过的比分。
- Elo 的评分被截断为整数，且按人数放大的更新在热门作品对上会发散；现改为每对作品一场比赛、以偏好比例为结果的标准 Elo。
- 熵加权方法的归一化项符号错误；无随机种子的「VI」方法替换为 Bradley–Terry（MM 算法）。
- 「正面交锋」各类别现在分别在所有作品对中选取，而不是只在共同评分最多的 10 部中选取。
- 排除了被 VNDB 标记为忽略评分（`ign_votes`）的用户。这些账号贡献了约 220 万对作品比较（总数从 711 万降到 489 万），对排名影响明显。

## 许可

数据来自 [VNDB 数据库转储](https://vndb.org/d14)，依 ODbL 授权。
