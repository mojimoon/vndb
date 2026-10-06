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
    "stats.byYear": "每年评分数",
    "stats.meanByYear": "每年平均分",
    "stats.table": "表格",
    "stats.chart": "图表",
    "stats.year": "年份",
    "stats.count": "数量",
    "stats.score": "分数",
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
    "stats.byYear": "Votes per year",
    "stats.meanByYear": "Mean vote per year",
    "stats.table": "Table",
    "stats.chart": "Chart",
    "stats.year": "Year",
    "stats.count": "Count",
    "stats.score": "Score",
  },
} as const;

export type StringKey = keyof (typeof STRINGS)["zh"];

// ---------------------------------------------------------------------------
// Methods

type Text = { zh: string; en: string };

const PO: Record<string, { name: Text; desc: Text }> = {
  po_total: {
    name: { zh: "合计积分", en: "Total score" },
    desc: { zh: "对每个对手取 (x − y) 后平均。", en: "Average of (x − y) over all opponents." },
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
    desc: { zh: "对最稳定的 18 种科学排名（Massey、Colley、Markov 差值类、攻防、差值 × 偏好人数 / 百分位）做 Borda 计数。", en: "Borda count over the 18 most stable scientific rankings (Massey, Colley, margin-based Markov, OD, Difference × preference counts / percentiles)." },
  },
  borda_sci: {
    name: { zh: "科学排名合并", en: "Scientific merge" },
    desc: { zh: "合并「偏好人数」「算术平均分」「几何平均分」三组的 Borda 结果。", en: "Merges the Borda results for preference counts, arithmetic and geometric mean votes." },
  },
  borda_po: {
    name: { zh: "PONet 合并", en: "PONet merge" },
    desc: { zh: "8 种偏序网络方法的 Borda 计数。", en: "Borda count over the 8 PONet methods." },
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
    return v ? (lang === "zh" ? `以「${v.zh}」为比分的 8 种科学排名的 Borda 计数。` : `Borda count over the 8 scientific rankings using ${v.en} as game scores.`) : null;
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
