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

const GITHUB_ICON =
  "M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z";

/** The VNDB page matching the current page (title, user, developer), else VNDB's home page. */
export function vndbUrl(path: string): string {
  const m = path.match(/^\/(?:compare\/)?(vn|user|dev)\/(\d+)/);
  if (!m) return "https://vndb.org/";
  const prefix = { vn: "v", user: "u", dev: "p" }[m[1] as "vn" | "user" | "dev"];
  return `https://vndb.org/${prefix}${m[2]}`;
}

export default function Layout() {
  const { t, lang, setLang } = useI18n();
  const [theme, setTheme] = useTheme();
  const meta = useMeta();
  const location = useLocation();
  const next: Theme = theme === "system" ? "light" : theme === "light" ? "dark" : "system";

  const [menu, setMenu] = useState(false);
  // Close the mobile menu on navigation.
  useEffect(() => setMenu(false), [location.pathname]);

  const link = ({ isActive }: { isActive: boolean }) =>
    `whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
      isActive ? "bg-surface-2 text-ink" : "text-ink-2 hover:text-ink"
    }`;
  const mobileLink = ({ isActive }: { isActive: boolean }) =>
    `block rounded-md px-3 py-2 text-sm font-medium ${isActive ? "bg-surface-2 text-ink" : "text-ink-2 hover:bg-surface-2 hover:text-ink"}`;

  const navLinks = (cls: (s: { isActive: boolean }) => string) => (
    <>
      <NavLink to="/" className={({ isActive }) => cls({ isActive: isActive && !/^\/(vn|user|dev|compare|methods|stats)/.test(location.pathname) })}>
        {t("nav.ranking")}
      </NavLink>
      <NavLink to="/dev" className={cls}>
        {t("nav.devs")}
      </NavLink>
      <NavLink to="/user" className={cls}>
        {t("nav.users")}
      </NavLink>
      <NavLink to="/compare" className={cls}>
        {t("nav.compare")}
      </NavLink>
      <NavLink to="/stats" className={cls}>
        {t("nav.stats")}
      </NavLink>
      <NavLink to="/methods" className={cls}>
        {t("nav.methods")}
      </NavLink>
      <a href={vndbUrl(location.pathname)} target="_blank" rel="noopener noreferrer" className={cls({ isActive: false })} title={t("nav.vndbHint")}>
        VNDB ↗
      </a>
    </>
  );

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-20 border-b border-line bg-bg/85 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center gap-2 px-4 py-3">
          <button
            type="button"
            onClick={() => setMenu(!menu)}
            aria-expanded={menu}
            aria-controls="mobile-nav"
            aria-label={t("nav.menu")}
            className="-ml-1 rounded-md p-2 text-ink-2 hover:bg-surface-2 hover:text-ink lg:hidden"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d={menu ? "M6 6l12 12M18 6L6 18" : "M4 7h16M4 12h16M4 17h16"} />
            </svg>
          </button>
          <NavLink to="/" className="mr-1 flex min-w-0 shrink items-center gap-2 font-semibold tracking-tight lg:mr-2" aria-label={t("site.name")}>
            <img src="/favicon.svg" alt="" className="h-6 w-6 shrink-0" />
            <span className="truncate">{t("site.name")}</span>
          </NavLink>
          <nav className="hidden min-w-0 flex-1 items-center gap-1 lg:flex">{navLinks(link)}</nav>
          <div className="ml-auto flex shrink-0 items-center gap-0.5">
            <a href="https://github.com/mojimoon/vndb" target="_blank" rel="noopener noreferrer" className="mr-1 hidden shrink-0 lg:block">
              <img src="https://img.shields.io/github/stars/mojimoon/vndb?style=social" alt="GitHub stars" height="20" className="h-5" />
            </a>
            <a
              href="https://github.com/mojimoon/vndb"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="GitHub"
              title="GitHub"
              className="rounded-md p-2 text-ink-2 hover:bg-surface-2 hover:text-ink lg:hidden"
            >
              <svg viewBox="0 0 16 16" className="h-4 w-4" fill="currentColor" aria-hidden>
                <path d={GITHUB_ICON} />
              </svg>
            </a>
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
        {menu && (
          <nav id="mobile-nav" className="border-t border-line bg-bg px-4 py-2 lg:hidden">
            <div className="mx-auto grid max-w-6xl grid-cols-2 gap-1 sm:grid-cols-4">{navLinks(mobileLink)}</div>
          </nav>
        )}
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
