import { NavLink, useLocation } from "react-router";

/** URL-backed tabs: each tab is a sub-path; the query string is kept when switching. */
export function Tabs({ base, tabs }: { base: string; tabs: { path: string; label: string }[] }) {
  const { search } = useLocation();
  return (
    <nav className="-mx-4 overflow-x-auto px-4" aria-label="tabs">
      <div className="flex min-w-max gap-1 border-b border-line">
        {tabs.map((t) => (
          <NavLink
            key={t.path}
            to={{ pathname: t.path ? `${base}/${t.path}` : base || "/", search }}
            end
            className={({ isActive }) =>
              `-mb-px whitespace-nowrap border-b-2 px-3 py-2 text-sm font-medium ${
                isActive ? "border-accent text-ink" : "border-transparent text-ink-2 hover:text-ink"
              }`
            }
          >
            {t.label}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
