# VNDB SciRanking

English | [中文](README.zh.md)

Rankings for the [Visual Novel Database](https://vndb.org/) (VNDB) built from partial order networks (PONet) and [scientific ranking](https://github.com/wattlebird/ranking). The data is refreshed daily from the VNDB database dump, and the site runs on Cloudflare Workers + D1.

## Background

Partial-order-network ranking was proposed by [@eyecandy](https://bgm.tv/user/eyecandy). Its core idea is to **judge titles against each other through the users who rated both**, instead of looking at each title's own score distribution.

This gives a relative ranking from simple, fast computations, with no Bayesian averaging or other heavy statistics, and complements [scientific ranking](https://chii.ai/rank). Put simply: if many people rated both A and B, and most of them (>50%) rated A higher, then A is better than B.

- In 2022, @eyecandy tried it on Bangumi's anime ratings and posted the results on the [Bangumi forum](https://bgm.tv/group/topic/371075).
- In 2023, I recomputed the anime PONet ranking from the [Bangumi15M](https://www.kaggle.com/datasets/klion23/bangumi15m) dataset ([mojimoon/bangumi-anime-ranking](https://github.com/mojimoon/bangumi-anime-ranking/tree/main/ponet), [discussion](https://bgm.tv/group/topic/382497)).
- Large datasets are hard to get from the Bangumi API, so that ranking was never updated. VNDB publishes a daily [database dump](https://vndb.org/d14), which makes regular updates possible, so this project applies the method to VNDB.

## Algorithm

1. **Build the partial order network.** For each pair of titles A and B with n common voters, of whom x rated A higher and y rated B higher, A scores against B:
   - **Total score**: x − y;
   - **Percentage score**: (x − y) / n, close to the preference probability of scientific ranking;
   - **Simple score**: sgn(x − y), i.e. 1, 0 or −1.
2. **Turn it into a total order.** Each title's final score is the average of its scores against every other title.

Two parameters:

- `min_vote = 30`: only titles with at least 30 votes are ranked (VNDB's own top 50 uses the same threshold).
- `min_common_vote = 5`: a pair only counts when at least 5 users voted on both.

### Scientific ranking and beyond

The tuples `(A, B, x, y, n)` recorded while building the network are exactly the game results that scientific ranking needs (see the [rankit write-up](https://ikely.me/2016/02/05/%E4%BD%BF%E7%94%A8-rankit-%E6%9E%84%E5%BB%BA%E6%9B%B4%E7%A7%91%E5%AD%A6%E7%9A%84%E6%8E%92%E5%90%8D/) and [rankit](https://github.com/wattlebird/ranking)), so the project also includes Massey, Colley, Markov, offence–defence and other rating systems, plus Bradley–Terry and Elo. Borda counts merge them; the default **grand ranking** is a Borda count over the 9 methods that agree best between two random halves of the voters.

The project also ranks by **sample percentiles**, which ignore how generous a user is and replace each vote with its position among that user's votes. If user C gave n votes $x_1 < x_2 < \ldots < x_n$, the n + 1 intervals $(-\infty, x_1), (x_1, x_2), \ldots, (x_n, +\infty)$ are taken to be equally likely, and tied votes share the midpoint:

$$\text{sp}(x_k) = \frac{|\{x_i \mid x_i < x_k\}| + 0.5 \cdot |\{x_i \ (i \neq k) \mid x_i = x_k\}| + 1}{n + 1}$$

41 methods are available in total. [docs/ranking-experiments.md](docs/ranking-experiments.md) covers how they were chosen and how the grand ranking was built.

## Features

- **Ranking**: any of the 41 methods, side by side with up to 3 others; custom columns, CSV export; filters by developer, year, language, length, votes and VNDB rating. Tabs for best by year, disagreements with VNDB, movers, and history (the ranking on any recorded day and the change since).
- **Title pages**: overview, rating analysis with voter filters, rank by every method, rank history and VNDB rating / votes over time, head-to-head against any title, similar titles, reviews (public list notes) and a TF-IDF tag analysis.
- **Developers**: developer leaderboard (titles, best / median rank, mean rating), developer pages and developer comparisons.
- **Users**: user leaderboards, taste analysis, all votes, recommendations from similar users, similar users, reviews and tag preferences. Users with at least 5 votes on ranked titles get a page.
- **Compare**: two titles (head-to-head, distributions, joint vote matrix, reviews by people who reviewed both) or two users (raw score / percentile scatter, disagreements, shared favourites).
- Chinese, English and Japanese UI; light and dark themes.

## Usage

Preview the site locally with the bundled sample data (not real data, Node.js 22+):

```bash
cd web
npm install
npm run db:seed:local
npm run dev                          # http://localhost:5173
```

Run the pipeline on a real dump (Python 3.11+, a few GB of RAM):

```bash
curl -L -o db.tar.zst https://dl.vndb.org/dump/vndb-db-latest.tar.zst
mkdir -p db && tar -I zstd -xf db.tar.zst -C db && rm db.tar.zst

cd pipeline
pip install -r requirements.txt
python -m vndb_rank --dump ../db --out out
npx --prefix ../web wrangler d1 execute vndb --local --file out/snapshot.sql
```

To host your own copy with daily updates, see [docs/deployment.md](docs/deployment.md).

## Documentation

- [docs/deployment.md](docs/deployment.md): Cloudflare setup, GitHub Actions, local development
- [docs/architecture.md](docs/architecture.md): repository layout, data flow, database schema, free-tier budget
- [docs/ranking-experiments.md](docs/ranking-experiments.md): ranking methods and the experiments behind the grand ranking

## License

Data from the [VNDB database dump](https://vndb.org/d14), under the ODbL.
