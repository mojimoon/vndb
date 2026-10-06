import { useEffect, useState } from "react";
import { NavLink, Outlet, ScrollRestoration, useLocation } from "react-router";
import { useI18n } from "../lib/i18n";
import { useMeta } from "../lib/api";

type Theme = "system" | "light" | "dark";

function useTheme(): [Theme, (t: Theme) => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    const t = document.documentElement.dataset.theme;
    return t === "light" || t === "dark" ? t : "system";
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") delete root.dataset.theme;
    else root.dataset.theme = theme;
    try {
      if (theme === "system") localStorage.removeItem("theme");
      else localStorage.setItem("theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);
  return [theme, setTheme];
}

const ICONS: Record<Theme, string> = {
  system: "M4 5h16v11H4zM9 20h6M12 16v4",
  light: "M12 4V2m0 20v-2m8-8h2M2 12h2m13.66-5.66 1.41-1.41M4.93 19.07l1.41-1.41m0-11.32L4.93 4.93m14.14 14.14-1.41-1.41M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z",
  dark: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z",
};

export default function Layout() {
  const { t, lang, setLang } = useI18n();
  const [theme, setTheme] = useTheme();
  const meta = useMeta();
  const location = useLocation();
  const next: Theme = theme === "system" ? "light" : theme === "light" ? "dark" : "system";

  const link = ({ isActive }: { isActive: boolean }) =>
    `whitespace-nowrap rounded-md px-2 py-1.5 text-sm font-medium transition-colors sm:px-3 ${
      isActive ? "bg-surface-2 text-ink" : "text-ink-2 hover:text-ink"
    }`;

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-20 border-b border-line bg-bg/85 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-2 px-4 py-3">
          <NavLink to="/" className="mr-1 flex shrink-0 items-center gap-2 font-semibold tracking-tight sm:mr-2" aria-label="VNDB Ranking+">
            <img src="/favicon.svg" alt="" className="h-6 w-6" />
            <span className="hidden sm:inline">VNDB Ranking+</span>
          </NavLink>
          <nav className="-my-1 flex min-w-0 flex-1 items-center gap-0.5 overflow-x-auto py-1 sm:gap-1">
            <NavLink to="/" className={({ isActive }) => link({ isActive: isActive && !/^\/(vn|user|dev|compare|methods|stats)/.test(location.pathname) })}>
              {t("nav.ranking")}
            </NavLink>
            <NavLink to="/dev" className={link}>
              {t("nav.devs")}
            </NavLink>
            <NavLink to="/user" className={link}>
              {t("nav.users")}
            </NavLink>
            <NavLink to="/compare" className={link}>
              {t("nav.compare")}
            </NavLink>
            <NavLink to="/methods" className={link}>
              {t("nav.methods")}
            </NavLink>
            <NavLink to="/stats" className={link}>
              {t("nav.stats")}
            </NavLink>
          </nav>
          <div className="ml-auto flex shrink-0 items-center gap-0.5">
            <button
              type="button"
              onClick={() => setLang(lang === "zh" ? "en" : "zh")}
              className="rounded-md px-2 py-1.5 text-sm text-ink-2 hover:bg-surface-2 hover:text-ink"
              aria-label="Switch language"
            >
              {lang === "zh" ? "EN" : "中文"}
            </button>
            <button
              type="button"
              onClick={() => setTheme(next)}
              className="rounded-md p-2 text-ink-2 hover:bg-surface-2 hover:text-ink"
              title={t(`theme.${theme}`)}
              aria-label={t(`theme.${theme}`)}
            >
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d={ICONS[theme]} />
              </svg>
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">
        <Outlet />
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-4 gap-y-1 px-4 py-5 text-xs text-ink-3">
          <span>
            {t("footer.data")} (<a className="underline hover:text-ink-2" href="https://vndb.org/d14">vndb.org/d14</a>, ODbL)
          </span>
          {meta.state === "ok" && (
            <span className="tabular">
              {t("footer.snapshot")}: {meta.data.info.dump_date}
            </span>
          )}
          <a className="underline hover:text-ink-2" href="https://github.com/mojimoon/vndb">
            {t("footer.source")}
          </a>
        </div>
      </footer>
      <ScrollRestoration />
    </div>
  );
}
