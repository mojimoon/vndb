import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

export type Lang = "zh" | "en";

const STRINGS = {
  zh: {
    "nav.ranking": "排行",
    "nav.methods": "方法",
    "nav.stats": "统计",
    "site.tagline": "基于偏序网络与科学排名的 VNDB 视觉小说排行",
    "footer.data": "数据来自 VNDB 数据库转储",
    "footer.snapshot": "快照",
    "footer.source": "源代码",
    "theme.light": "浅色",
    "theme.dark": "深色",
    "theme.system": "跟随系统",
    "common.loading": "加载中…",
    "common.error": "加载失败",
    "common.retry": "重试",
    "common.nodata": "数据库中还没有数据。请先运行数据管线并导入快照。",
    "common.notfound": "找不到该作品（可能票数不足，未进入排名）。",
    "rank.method": "排名方法",
    "rank.featured": "推荐",
    "rank.search": "搜索标题 / 别名 / v12345",
    "rank.olang": "原语言",
    "rank.all": "全部",
    "rank.yearFrom": "起始年份",
    "rank.yearTo": "结束年份",
    "rank.minVotes": "最少票数",
    "rank.developer": "开发商",
    "rank.clear": "清除筛选",
    "rank.results": "{n} 部作品",
    "rank.col.rank": "排名",
    "rank.col.delta": "对比 VNDB",
    "rank.col.title": "标题",
    "rank.col.year": "年份",
    "rank.col.votes": "票数",
    "rank.col.rating": "VNDB 评分",
    "rank.col.score": "得分",
    "rank.deltaHint": "VNDB 排名 − 本方法排名，正数表示本方法排得更靠前",
    "rank.empty": "没有符合条件的作品。",
    "rank.prev": "上一页",
    "rank.next": "下一页",
    "rank.page": "第 {p} / {n} 页",
    "vn.onVndb": "在 VNDB 查看",
    "vn.developer": "开发商",
    "vn.released": "发售",
    "vn.votes": "票数",
    "vn.rating": "VNDB 评分",
    "vn.average": "平均分",
    "vn.ranks": "各方法排名",
    "vn.allMethods": "展开全部 {n} 种方法",
    "vn.lessMethods": "收起",
    "vn.h2h": "正面交锋",
    "vn.h2hHint": "同时给两部作品打分的用户中：多少人认为本作更好（蓝）、打平（灰）、更差（红）。",
    "vn.h2h.popular": "共同评分最多",
    "vn.h2h.ahead": "常被认为更好",
    "vn.h2h.behind": "常被认为更差",
    "vn.h2h.contested": "势均力敌",
    "vn.h2h.tied": "常打平手",
    "vn.h2h.none": "没有足够的共同评分用户。",
    "vn.h2h.row": "{w} 胜 · {t} 平 · {l} 负（共 {n} 人）",
    "vn.relations": "关联作品",
    "vn.showCover": "显示封面",
    "vn.nsfwCover": "封面可能含有成人内容",
    "vn.unranked": "未排名",
    "methods.title": "排名方法",
    "methods.params": "参数",
    "methods.minVote": "进入排名所需最少票数",
    "methods.minCommon": "计入一对作品所需最少共同评分人数",
    "methods.agreement": "方法一致性（Kendall τ）",
    "methods.agreementHint": "两种方法给出的排名越一致，τ 越接近 1。悬停查看数值。",
    "methods.group.merged": "Borda 合并",
    "methods.group.po": "偏序网络 (PONet)",
    "methods.group.rankit": "科学排名 (rankit)",
    "methods.group.ref": "参考",
    "methods.rankitHint": "把每对作品视作一场比赛：行是求解方法，列是用作「比分」的量。点击查看对应排名。",
    "stats.title": "统计",
    "stats.ranked": "进入排名的作品",
    "stats.pairs": "可比较的作品对",
    "stats.votes": "总评分数",
    "stats.users": "参与排名的用户",
    "stats.mean": "平均分",
    "stats.dist": "评分分布",
    "stats.byYear": "每年评分数与平均分",
    "stats.meanByYear": "每年平均分",
    "stats.table": "表格",
    "stats.chart": "图表",
    "stats.year": "年份",
    "stats.count": "数量",
    "stats.score": "分数",
    "nav.users": "用户",
    "nav.compare": "对比",
    "rank.tab.table": "总榜",
    "rank.tab.years": "年度最佳",
    "rank.tab.disputes": "分歧",
    "rank.tab.movers": "排名变动",
    "rank.columns": "显示列",
    "rank.compareWith": "对比方法",
    "rank.addMethod": "＋ 添加方法",
    "rank.pageSize": "每页",
    "rank.export": "导出 CSV",
    "rank.col.trend": "7 日",
    "rank.col.dev": "开发商",
    "rank.col.lang": "语言",
    "rank.col.length": "长度",
    "rank.years.hint": "按发售年份分组，每年按当前方法取前 {n} 名。",
    "rank.disputes.under": "被 VNDB 低估",
    "rank.disputes.over": "被 VNDB 高估",
    "rank.disputes.hint": "在当前方法前 {n} 名中，与 VNDB 排名差距最大的作品。",
    "rank.movers.up": "上升最多",
    "rank.movers.down": "下降最多",
    "rank.movers.none": "排名历史仍在积累，每日更新后才会出现变动。",
    "length.1": "很短",
    "length.2": "短",
    "length.3": "中等",
    "length.4": "长",
    "length.5": "很长",
    "vn.tab.overview": "概览",
    "vn.tab.ratings": "评分分析",
    "vn.tab.ranks": "排名",
    "vn.tab.versus": "对决",
    "vn.tab.similar": "相似作品",
    "vn.ratings.dist": "评分分布",
    "vn.ratings.byYear": "每年评分数与平均分",
    "vn.ratings.meanByYear": "每年平均分",
    "vn.ratings.labels": "列表状态",
    "vn.ratings.std": "标准差",
    "vn.ratings.count": "评分人数（排名用）",
    "vn.ratings.sp": "在评分者心中的位置",
    "vn.ratings.spHint": "把每个评分换算成它在该用户全部评分中的百分位：越靠右，说明评分者越把它当作自己的最爱之一。平均 {p}。",
    "vn.ratings.bias": "评分者偏好",
    "vn.ratings.biasHint": "评分减去该评分者自己的平均分后再取平均：{b}。正数表示大家给它的分数高于他们平时的水平。",
    "label.1": "在玩",
    "label.2": "已完成",
    "label.3": "搁置",
    "label.4": "弃坑",
    "label.5": "想玩",
    "label.6": "黑名单",
    "vn.history": "排名变化",
    "vn.historyHint": "每日更新后记录；越往上排名越高。",
    "vn.historyEmpty": "排名历史从第一次每日更新开始积累。",
    "vn.compareAny": "与任意作品对比",
    "vn.pickVn": "搜索作品…",
    "vn.fullCompare": "完整对比 →",
    "vn.belowThreshold": "共同评分的用户不足 {n} 人，没有记录这对作品。",
    "vn.similarHint": "评分模式最相似的作品：喜欢本作的用户也倾向于给这些作品打高分（中心化余弦相似度）。",
    "vn.common": "共同评分 {n} 人",
    "user.search.title": "用户",
    "user.search.hint": "输入 VNDB 用户名、用户编号（u12345）或个人主页链接。只有在已进入排名的作品上至少有 {n} 个评分的用户才有页面。",
    "user.search.placeholder": "用户名 / u12345",
    "user.search.go": "查看",
    "user.notFound": "找不到这位用户，或其公开评分不足。",
    "user.tab.overview": "概览",
    "user.tab.votes": "评分",
    "user.tab.recs": "推荐",
    "user.tab.similar": "相似用户",
    "user.onVndb": "在 VNDB 查看",
    "user.votes": "评分数",
    "user.mean": "平均分",
    "user.corr": "与 VNDB 评分的相关性",
    "user.generosity": "相对 VNDB 评分",
    "user.loved": "比大家更喜欢",
    "user.hated": "比大家更不喜欢",
    "user.dist": "评分分布",
    "user.recsHint": "根据评分最相似的 {k} 位用户推荐尚未评分的作品。预测分 = 你的平均分 + 相似用户对该作品的相对评价。",
    "user.pred": "预测分",
    "user.support": "{n} 位相似用户评过",
    "user.similarHint": "评分偏好最接近的用户（只比较双方都评分过的作品）。",
    "user.sim": "相似度",
    "user.common": "共同作品",
    "user.yourVote": "评分",
    "user.diff": "与 VNDB 之差",
    "user.compare": "对比",
    "user.noRecs": "暂时没有足够的数据给出推荐。",
    "compare.title": "对比",
    "compare.vn": "作品",
    "compare.user": "用户",
    "compare.pickA": "第一项",
    "compare.pickB": "第二项",
    "compare.go": "对比",
    "compare.h2h": "正面交锋",
    "compare.common": "共同作品",
    "compare.pearson": "评分相关性 r",
    "compare.agreeRate": "评分差距 ≤ 1 的比例",
    "compare.scatter": "共同作品评分",
    "compare.disagree": "分歧最大",
    "compare.agree": "看法一致的高分作品",
    "compare.only": "{a} 打了高分、{b} 还没评分",
    "compare.hint": "选择两部作品或两位用户进行对比。",
    "compare.swap": "交换",
    "nav.devs": "开发商",
    "chart.cumulative": "累计占比",
    "common.more": "显示更多",
    "common.all": "全部显示",
    "compare.raw": "原始分",
    "compare.sp": "样本百分位",
    "compare.scale": "比较尺度",
    "compare.agreeRateSp": "百分位差距 ≤ 10% 的比例",
    "compare.spAxis": "百分位映射到 1–10 刻度显示",
    "compare.draw": "打平",
    "compare.prefers": "认为「{x}」更好",
    "compare.matrix": "共同评分者的评分矩阵",
    "compare.matrixHint": "{n} 位同时评过两部作品的用户：每格是「横轴分数 × 纵轴分数」的人数。对角线以下表示更偏爱横轴作品。",
    "compare.sharedHigh": "两人都打了高分",
    "compare.sharedLow": "两人都打了低分",
    "compare.onlyHigh": "{a} 打了高分、{b} 未评分",
    "compare.onlyLow": "{a} 打了低分、{b} 未评分",
    "dev.title": "开发商",
    "dev.hint": "按「{m}」统计每个开发商的已排名作品；排名越小越好。",
    "dev.search": "搜索开发商…",
    "dev.minCount": "最少作品数",
    "dev.count": "{n} 个开发商",
    "dev.vns": "作品数",
    "dev.best": "最佳排名",
    "dev.median": "排名中位数",
    "dev.meanRating": "平均 VNDB 评分",
    "dev.byYear": "每年作品数与平均评分",
    "dev.compareWith": "与其他开发商对比",
    "dev.inRanking": "在排行中查看",
    "dev.beats": "优于 {p} 的开发商（共 {n} 个，至少 3 部作品）",
    "dev.statsHint": "统计基于「{m}」，只包含进入排名的 {n} 部作品中的作品。",
    "dev.titles": "作品（{n}）",
    "lb.title": "用户排行",
    "lb.hint": "平均分与「主流程度」只统计在已排名作品上至少有 {n} 个评分的用户。",
    "lb.mostVotes": "评分最多",
    "lb.mostVotesYear": "{y} 年评分最多",
    "lb.highestMean": "平均分最高",
    "lb.lowestMean": "平均分最低",
    "lb.mainstream": "最主流（与作品平均分最相关）",
    "lb.contrarian": "最另类（与作品平均分最不相关）",
    "lb.rankedVotes": "在已排名作品上有 {n} 个评分",
    "notes.hint": "来自用户公开列表中的备注（至少 20 字），按更新时间从新到旧排列。",
    "notes.userHint": "该用户在已排名作品上的公开列表备注。",
    "notes.none": "暂无备注。",
    "rank.advanced": "高级筛选",
    "rank.devSearch": "按开发商筛选…",
    "rank.filterDev": "只看该开发商",
    "user.pages": "共 {n} 位用户有个人页面",
    "vn.noCommon": "没有用户同时评过这两部作品。",
    "vn.tab.notes": "备注",
    "methods.credits": "致谢",
  },
  en: {
    "nav.ranking": "Ranking",
    "nav.methods": "Methods",
    "nav.stats": "Stats",
    "site.tagline": "Visual novel rankings from VNDB via partial order networks and scientific ranking",
    "footer.data": "Data from the VNDB database dump",
    "footer.snapshot": "Snapshot",
    "footer.source": "Source",
    "theme.light": "Light",
    "theme.dark": "Dark",
    "theme.system": "System",
    "common.loading": "Loading…",
    "common.error": "Failed to load",
    "common.retry": "Retry",
    "common.nodata": "No data in the database yet. Run the pipeline and import a snapshot first.",
    "common.notfound": "Not found (it may not have enough votes to be ranked).",
    "rank.method": "Method",
    "rank.featured": "Featured",
    "rank.search": "Search title / alias / v12345",
    "rank.olang": "Language",
    "rank.all": "All",
    "rank.yearFrom": "From year",
    "rank.yearTo": "To year",
    "rank.minVotes": "Min. votes",
    "rank.developer": "Developer",
    "rank.clear": "Clear filters",
    "rank.results": "{n} titles",
    "rank.col.rank": "Rank",
    "rank.col.delta": "vs VNDB",
    "rank.col.title": "Title",
    "rank.col.year": "Year",
    "rank.col.votes": "Votes",
    "rank.col.rating": "VNDB",
    "rank.col.score": "Score",
    "rank.deltaHint": "VNDB rank − this rank; positive means this method ranks it higher",
    "rank.empty": "Nothing matches these filters.",
    "rank.prev": "Previous",
    "rank.next": "Next",
    "rank.page": "Page {p} of {n}",
    "vn.onVndb": "View on VNDB",
    "vn.developer": "Developer",
    "vn.released": "Released",
    "vn.votes": "Votes",
    "vn.rating": "VNDB rating",
    "vn.average": "Average",
    "vn.ranks": "Rank by method",
    "vn.allMethods": "Show all {n} methods",
    "vn.lessMethods": "Show less",
    "vn.h2h": "Head to head",
    "vn.h2hHint": "Among users who voted on both: how many rated this one higher (blue), the same (gray), or lower (red).",
    "vn.h2h.popular": "Most co-rated",
    "vn.h2h.ahead": "Usually preferred",
    "vn.h2h.behind": "Usually behind",
    "vn.h2h.contested": "Closest calls",
    "vn.h2h.tied": "Most ties",
    "vn.h2h.none": "Not enough users voted on both.",
    "vn.h2h.row": "{w} wins · {t} ties · {l} losses (of {n})",
    "vn.relations": "Related",
    "vn.showCover": "Show cover",
    "vn.nsfwCover": "Cover may contain adult content",
    "vn.unranked": "unranked",
    "methods.title": "Ranking methods",
    "methods.params": "Parameters",
    "methods.minVote": "Minimum votes to be ranked",
    "methods.minCommon": "Minimum common voters for a pair to count",
    "methods.agreement": "Method agreement (Kendall τ)",
    "methods.agreementHint": "The closer τ is to 1, the more two methods agree. Hover for values.",
    "methods.group.merged": "Borda merges",
    "methods.group.po": "Partial order network (PONet)",
    "methods.group.rankit": "Scientific ranking (rankit)",
    "methods.group.ref": "Reference",
    "methods.rankitHint": "Each pair of titles is a game: rows are the rating system, columns the quantity used as the game score. Click a cell to open that ranking.",
    "stats.title": "Statistics",
    "stats.ranked": "Ranked titles",
    "stats.pairs": "Comparable pairs",
    "stats.votes": "Votes",
    "stats.users": "Users in the ranking",
    "stats.mean": "Mean vote",
    "stats.dist": "Vote distribution",
    "stats.byYear": "Votes and mean vote per year",
    "stats.meanByYear": "Mean vote per year",
    "stats.table": "Table",
    "stats.chart": "Chart",
    "stats.year": "Year",
    "stats.count": "Count",
    "stats.score": "Score",
    "nav.users": "Users",
    "nav.compare": "Compare",
    "rank.tab.table": "Table",
    "rank.tab.years": "Best by year",
    "rank.tab.disputes": "Disagreements",
    "rank.tab.movers": "Movers",
    "rank.columns": "Columns",
    "rank.compareWith": "Compare with",
    "rank.addMethod": "+ Add method",
    "rank.pageSize": "Per page",
    "rank.export": "Export CSV",
    "rank.col.trend": "7d",
    "rank.col.dev": "Developer",
    "rank.col.lang": "Language",
    "rank.col.length": "Length",
    "rank.years.hint": "Grouped by release year; top {n} per year under the current method.",
    "rank.disputes.under": "Underrated by VNDB",
    "rank.disputes.over": "Overrated by VNDB",
    "rank.disputes.hint": "Within the top {n} of the current method, the titles whose VNDB rank differs the most.",
    "rank.movers.up": "Biggest risers",
    "rank.movers.down": "Biggest fallers",
    "rank.movers.none": "Rank history is still accumulating; movers appear after a few daily updates.",
    "length.1": "Very short",
    "length.2": "Short",
    "length.3": "Medium",
    "length.4": "Long",
    "length.5": "Very long",
    "vn.tab.overview": "Overview",
    "vn.tab.ratings": "Ratings",
    "vn.tab.ranks": "Ranks",
    "vn.tab.versus": "Versus",
    "vn.tab.similar": "Similar",
    "vn.ratings.dist": "Vote distribution",
    "vn.ratings.byYear": "Votes and mean vote per year",
    "vn.ratings.meanByYear": "Mean vote per year",
    "vn.ratings.labels": "List status",
    "vn.ratings.std": "Std. deviation",
    "vn.ratings.count": "Votes (used for ranking)",
    "vn.ratings.sp": "Where voters place it",
    "vn.ratings.spHint": "Each vote as a percentile within that voter's own list: the further right, the more voters count it among their favourites. Mean {p}.",
    "vn.ratings.bias": "Voter preference",
    "vn.ratings.biasHint": "Average of (vote − that voter's own mean): {b}. Positive means people rate it above their usual level.",
    "label.1": "Playing",
    "label.2": "Finished",
    "label.3": "Stalled",
    "label.4": "Dropped",
    "label.5": "Wishlist",
    "label.6": "Blacklist",
    "vn.history": "Rank history",
    "vn.historyHint": "Recorded at each daily update; higher is better.",
    "vn.historyEmpty": "Rank history accumulates from the first daily update.",
    "vn.compareAny": "Compare with any title",
    "vn.pickVn": "Search a title…",
    "vn.fullCompare": "Full comparison →",
    "vn.belowThreshold": "Fewer than {n} users voted on both, so this pair isn't recorded.",
    "vn.similarHint": "Titles with the most similar rating pattern: people who like this one tend to rate these highly too (centered cosine similarity).",
    "vn.common": "{n} common voters",
    "user.search.title": "Users",
    "user.search.hint": "Enter a VNDB username, user id (u12345) or profile link. Users need at least {n} votes on ranked titles to have a page.",
    "user.search.placeholder": "username / u12345",
    "user.search.go": "Open",
    "user.notFound": "User not found, or not enough public votes.",
    "user.tab.overview": "Overview",
    "user.tab.votes": "Votes",
    "user.tab.recs": "Recommendations",
    "user.tab.similar": "Similar users",
    "user.onVndb": "View on VNDB",
    "user.votes": "Votes",
    "user.mean": "Mean vote",
    "user.corr": "Correlation with VNDB rating",
    "user.generosity": "vs VNDB rating",
    "user.loved": "Liked more than most",
    "user.hated": "Liked less than most",
    "user.dist": "Vote distribution",
    "user.recsHint": "Unvoted titles recommended from the {k} users with the most similar votes. Prediction = your mean + how those users rated it relative to their own means.",
    "user.pred": "Predicted",
    "user.support": "rated by {n} similar users",
    "user.similarHint": "Users whose votes agree the most (only titles both have voted on are compared).",
    "user.sim": "Similarity",
    "user.common": "In common",
    "user.yourVote": "Vote",
    "user.diff": "vs VNDB",
    "user.compare": "Compare",
    "user.noRecs": "Not enough data for recommendations yet.",
    "compare.title": "Compare",
    "compare.vn": "Titles",
    "compare.user": "Users",
    "compare.pickA": "First",
    "compare.pickB": "Second",
    "compare.go": "Compare",
    "compare.h2h": "Head to head",
    "compare.common": "In common",
    "compare.pearson": "Vote correlation r",
    "compare.agreeRate": "Votes within 1 point",
    "compare.scatter": "Votes on common titles",
    "compare.disagree": "Biggest disagreements",
    "compare.agree": "Shared favourites",
    "compare.only": "{a} rated highly, {b} hasn't voted",
    "compare.hint": "Pick two titles or two users to compare.",
    "compare.swap": "Swap",
    "nav.devs": "Developers",
    "chart.cumulative": "Cumulative share",
    "common.more": "Show more",
    "common.all": "Show all",
    "compare.raw": "Raw score",
    "compare.sp": "Sample percentile",
    "compare.scale": "Scale",
    "compare.agreeRateSp": "Percentiles within 10%",
    "compare.spAxis": "Percentiles mapped onto a 1–10 scale",
    "compare.draw": "Draw",
    "compare.prefers": "prefer {x}",
    "compare.matrix": "Joint votes of common voters",
    "compare.matrixHint": "{n} users voted on both: each cell counts users with that (x, y) pair of votes. Cells below the diagonal prefer the title on the x axis.",
    "compare.sharedHigh": "Both rated highly",
    "compare.sharedLow": "Both rated low",
    "compare.onlyHigh": "{a} rated highly, {b} hasn't voted",
    "compare.onlyLow": "{a} rated low, {b} hasn't voted",
    "dev.title": "Developers",
    "dev.hint": "Each developer's ranked titles under {m}; lower ranks are better.",
    "dev.search": "Search developers…",
    "dev.minCount": "Min. titles",
    "dev.count": "{n} developers",
    "dev.vns": "Titles",
    "dev.best": "Best rank",
    "dev.median": "Median rank",
    "dev.meanRating": "Mean VNDB rating",
    "dev.byYear": "Titles and mean rating per year",
    "dev.compareWith": "Compare with another developer",
    "dev.inRanking": "Show in ranking",
    "dev.beats": "Better than {p} of {n} developers with 3+ titles",
    "dev.statsHint": "Based on {m}; only counts the {n} ranked titles.",
    "dev.titles": "Titles ({n})",
    "lb.title": "User leaderboards",
    "lb.hint": "Mean vote and mainstream-ness only include users with at least {n} votes on ranked titles.",
    "lb.mostVotes": "Most votes",
    "lb.mostVotesYear": "Most votes in {y}",
    "lb.highestMean": "Highest mean vote",
    "lb.lowestMean": "Lowest mean vote",
    "lb.mainstream": "Most mainstream (votes track title means)",
    "lb.contrarian": "Most contrarian",
    "lb.rankedVotes": "{n} votes on ranked titles",
    "notes.hint": "Notes from users' public lists (20+ characters), newest first.",
    "notes.userHint": "This user's public list notes on ranked titles.",
    "notes.none": "No notes yet.",
    "rank.advanced": "Advanced filters",
    "rank.devSearch": "Filter by developer…",
    "rank.filterDev": "Only this developer",
    "user.pages": "{n} users have a page",
    "vn.noCommon": "Nobody voted on both titles.",
    "vn.tab.notes": "Notes",
    "methods.credits": "Credits",
  },
} as const;

export type StringKey = keyof (typeof STRINGS)["zh"];

// ---------------------------------------------------------------------------
// Methods

type Text = { zh: string; en: string };

const PO: Record<string, { name: Text; desc: Text }> = {
  po_total: {
    name: { zh: "合计积分", en: "Total score" },
    desc: { zh: "对每个对手取 (x − y) 后平均。人数越多差值越大，因此明显偏向热门作品。", en: "Average of (x − y) over all opponents. Bigger audiences give bigger margins, so it clearly favours popular titles." },
  },
  po_percent: {
    name: { zh: "比例积分", en: "Percentage score" },
    desc: { zh: "对每个对手取 (x − y) / n 后平均，接近科学排名中的倾向性概率。", en: "Average of (x − y) / n; close to the preference probability of scientific ranking." },
  },
  po_simple: {
    name: { zh: "简易积分", en: "Simple score" },
    desc: { zh: "对每个对手取 sgn(x − y) 后平均：只看多数人的意见。", en: "Average of sgn(x − y): only the majority opinion counts." },
  },
  po_weighted: {
    name: { zh: "加权简易积分", en: "Weighted simple" },
    desc: { zh: "sgn(x − y) · √n，共同评分人数越多权重越大。", en: "sgn(x − y) · √n, weighting pairs with more common voters." },
  },
  po_rw: {
    name: { zh: "随机游走", en: "Random walk" },
    desc: { zh: "PageRank 式随机游走，权重从胜者流向败者，取驻留概率的倒数。", en: "PageRank-style walk that flows from winners to losers; score is the inverse stationary mass." },
  },
  po_elo: {
    name: { zh: "Elo", en: "Elo" },
    desc: { zh: "每对作品视作一场比赛，结果为偏好比例；多轮随机顺序，K 递减。", en: "Each pair is one match whose result is the preference share; shuffled passes with decaying K." },
  },
  po_entropy: {
    name: { zh: "熵加权", en: "Entropy-weighted" },
    desc: { zh: "以二元熵衡量每对比较的分歧程度并加权平均净偏好。", en: "Net preference weighted by the binary entropy of each comparison." },
  },
  po_bt: {
    name: { zh: "Bradley–Terry", en: "Bradley–Terry" },
    desc: { zh: "用 MM 算法拟合 Bradley–Terry 模型，平局各算半胜；得分为对数强度。", en: "Bradley–Terry strengths fitted with the MM algorithm, ties as half wins; score is log-strength." },
  },
};

const RANKERS: Record<string, Text> = {
  massey: { zh: "Massey", en: "Massey" },
  colley: { zh: "Colley", en: "Colley" },
  keener: { zh: "Keener", en: "Keener" },
  markov_rv: { zh: "Markov（胜率）", en: "Markov (win share)" },
  markov_rdv: { zh: "Markov（胜率差）", en: "Markov (share margin)" },
  markov_sdv: { zh: "Markov（分差）", en: "Markov (score margin)" },
  od: { zh: "攻防 (OD)", en: "Offence–defence" },
  difference: { zh: "差值", en: "Difference" },
};

const VARIABLES: Record<string, Text> = {
  prob: { zh: "偏好人数", en: "preference counts" },
  ari: { zh: "算术平均分", en: "arithmetic mean vote" },
  geo: { zh: "几何平均分", en: "geometric mean vote" },
  sp_ari: { zh: "百分位算术平均", en: "arithmetic mean percentile" },
  sp_geo: { zh: "百分位几何平均", en: "geometric mean percentile" },
};

const MERGED: Record<string, { name: Text; desc: Text }> = {
  borda_grand: {
    name: { zh: "综合排名", en: "Grand ranking" },
    desc: { zh: "对 14 种最稳定的科学排名（Massey、Colley、Markov 差值类、攻防 × 偏好人数 / 百分位）做 Borda 计数。", en: "Borda count over the 14 most stable scientific rankings (Massey, Colley, margin-based Markov and offence–defence × preference counts / percentiles)." },
  },
  borda_sci: {
    name: { zh: "科学排名合并", en: "Scientific merge" },
    desc: { zh: "合并「偏好人数」「算术平均分」「几何平均分」三组的 Borda 结果。", en: "Merges the Borda results for preference counts, arithmetic and geometric mean votes." },
  },
  borda_po: {
    name: { zh: "PONet 合并", en: "PONet merge" },
    desc: { zh: "7 种偏序网络方法的 Borda 计数。", en: "Borda count over the 7 PONet methods." },
  },
};

export const RANKER_CODES = Object.keys(RANKERS);
export const VARIABLE_CODES = Object.keys(VARIABLES);
export const rankerName = (code: string, lang: Lang) => RANKERS[code]?.[lang] ?? code;
export const variableName = (code: string, lang: Lang) => VARIABLES[code]?.[lang] ?? code;

export type MethodGroup = "merged" | "po" | "rankit" | "ref";

export function methodGroup(code: string): MethodGroup {
  if (code === "vndb") return "ref";
  if (code.startsWith("borda_")) return "merged";
  if (code.startsWith("po_")) return "po";
  return "rankit";
}

function splitRankit(code: string): [string, string] | null {
  for (const v of Object.keys(VARIABLES).sort((a, b) => b.length - a.length)) {
    if (code.endsWith(`_${v}`)) return [code.slice(0, -v.length - 1), v];
  }
  return null;
}

export function methodName(code: string, lang: Lang): string {
  if (code === "vndb") return lang === "zh" ? "VNDB 贝叶斯评分" : "VNDB Bayesian rating";
  if (MERGED[code]) return MERGED[code].name[lang];
  if (code.startsWith("borda_")) {
    const v = VARIABLES[code.slice(6)];
    return v ? (lang === "zh" ? `Borda · ${v.zh}` : `Borda · ${v.en}`) : code;
  }
  if (PO[code]) return `PONet · ${PO[code].name[lang]}`;
  const parts = splitRankit(code);
  if (parts && RANKERS[parts[0]]) return `${RANKERS[parts[0]][lang]} · ${VARIABLES[parts[1]][lang]}`;
  return code;
}

export function methodDesc(code: string, lang: Lang): string | null {
  if (code === "vndb") return lang === "zh" ? "VNDB 官方的贝叶斯平均评分，作为对照。" : "VNDB's own Bayesian-average rating, for reference.";
  if (MERGED[code]) return MERGED[code].desc[lang];
  if (PO[code]) return PO[code].desc[lang];
  if (code.startsWith("borda_")) {
    const v = VARIABLES[code.slice(6)];
    return v ? (lang === "zh" ? `以「${v.zh}」为比分的各种科学排名的 Borda 计数。` : `Borda count over the scientific rankings that use ${v.en} as game scores.`) : null;
  }
  const parts = splitRankit(code);
  if (!parts) return null;
  return lang === "zh"
    ? `把每对作品视作一场比赛、以「${VARIABLES[parts[1]].zh}」为比分，用 ${RANKERS[parts[0]].zh} 方法求解。`
    : `Treats each pair as a game scored by ${VARIABLES[parts[1]].en} and solves it with the ${RANKERS[parts[0]].en} method.`;
}

/** Short label for compact places (heatmap axes). */
export function methodShort(code: string, lang: Lang): string {
  if (code === "vndb") return "VNDB";
  if (MERGED[code]) return MERGED[code].name[lang];
  if (PO[code]) return PO[code].name[lang];
  return methodName(code, lang);
}

// ---------------------------------------------------------------------------

const LANG_NAMES: Record<string, Text> = {
  ja: { zh: "日语", en: "Japanese" },
  en: { zh: "英语", en: "English" },
  "zh-Hans": { zh: "简体中文", en: "Chinese (Simpl.)" },
  "zh-Hant": { zh: "繁体中文", en: "Chinese (Trad.)" },
  zh: { zh: "中文", en: "Chinese" },
  ko: { zh: "韩语", en: "Korean" },
  ru: { zh: "俄语", en: "Russian" },
  es: { zh: "西班牙语", en: "Spanish" },
  de: { zh: "德语", en: "German" },
  fr: { zh: "法语", en: "French" },
  vi: { zh: "越南语", en: "Vietnamese" },
  id: { zh: "印尼语", en: "Indonesian" },
};

export const langName = (code: string, lang: Lang) => LANG_NAMES[code]?.[lang] ?? code;

const RELATIONS: Record<string, Text> = {
  seq: { zh: "续作", en: "Sequel" },
  preq: { zh: "前作", en: "Prequel" },
  set: { zh: "同一世界观", en: "Same setting" },
  alt: { zh: "其他版本", en: "Alternative version" },
  char: { zh: "共享角色", en: "Shares characters" },
  side: { zh: "外传", en: "Side story" },
  par: { zh: "本篇", en: "Parent story" },
  ser: { zh: "同系列", en: "Same series" },
  fan: { zh: "衍生作品", en: "Fandisc" },
  orig: { zh: "原作", en: "Original game" },
};
export const relationName = (code: string, lang: Lang) => RELATIONS[code]?.[lang] ?? code;

// ---------------------------------------------------------------------------

interface I18n {
  lang: Lang;
  setLang: (l: Lang) => void;
  t: (key: StringKey, vars?: Record<string, string | number>) => string;
}

const Ctx = createContext<I18n | null>(null);

function initialLang(): Lang {
  try {
    const s = localStorage.getItem("lang");
    if (s === "zh" || s === "en") return s;
  } catch {
    /* storage unavailable */
  }
  return navigator.language.toLowerCase().startsWith("zh") ? "zh" : "en";
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang);
  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    try {
      localStorage.setItem("lang", l);
    } catch {
      /* storage unavailable */
    }
  }, []);
  useEffect(() => {
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  }, [lang]);
  const value = useMemo<I18n>(
    () => ({
      lang,
      setLang,
      t: (key, vars) => {
        let s: string = STRINGS[lang][key] ?? key;
        if (vars) for (const [k, v] of Object.entries(vars)) s = s.replace(`{${k}}`, String(v));
        return s;
      },
    }),
    [lang, setLang],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useI18n(): I18n {
  const v = useContext(Ctx);
  if (!v) throw new Error("useI18n outside provider");
  return v;
}
