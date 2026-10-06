import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router";
import "./index.css";
import { I18nProvider } from "./lib/i18n";
import Layout from "./components/Layout";
import RankingLayout from "./pages/ranking/RankingLayout";
import RankTable from "./pages/ranking/RankTable";
import Years from "./pages/ranking/Years";
import Disputes from "./pages/ranking/Disputes";
import Movers from "./pages/ranking/Movers";
import VnLayout from "./pages/vn/VnLayout";
import VnOverview from "./pages/vn/Overview";
import VnRatings from "./pages/vn/Ratings";
import VnRanks from "./pages/vn/Ranks";
import VnVersus from "./pages/vn/Versus";
import VnSimilar from "./pages/vn/Similar";
import UserSearch from "./pages/user/UserSearch";
import UserLayout from "./pages/user/UserLayout";
import UserOverview from "./pages/user/Overview";
import UserVotes from "./pages/user/Votes";
import UserRecs from "./pages/user/Recs";
import UserSimilar from "./pages/user/Similar";
import ComparePicker from "./pages/compare/ComparePicker";
import CompareVn from "./pages/compare/CompareVn";
import CompareUser from "./pages/compare/CompareUser";
import Methods from "./pages/Methods";
import Stats from "./pages/Stats";

function NotFound() {
  return <p className="py-16 text-center text-sm text-ink-3">404</p>;
}

const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      {
        element: <RankingLayout />,
        children: [
          { index: true, element: <RankTable /> },
          { path: "years", element: <Years /> },
          { path: "disputes", element: <Disputes /> },
          { path: "movers", element: <Movers /> },
        ],
      },
      {
        path: "vn/:id",
        element: <VnLayout />,
        children: [
          { index: true, element: <VnOverview /> },
          { path: "ratings", element: <VnRatings /> },
          { path: "ranks", element: <VnRanks /> },
          { path: "versus", element: <VnVersus /> },
          { path: "similar", element: <VnSimilar /> },
        ],
      },
      { path: "user", element: <UserSearch /> },
      {
        path: "user/:uid",
        element: <UserLayout />,
        children: [
          { index: true, element: <UserOverview /> },
          { path: "votes", element: <UserVotes /> },
          { path: "recs", element: <UserRecs /> },
          { path: "similar", element: <UserSimilar /> },
        ],
      },
      { path: "compare", element: <ComparePicker /> },
      { path: "compare/vn/:a/:b", element: <CompareVn /> },
      { path: "compare/user/:a/:b", element: <CompareUser /> },
      { path: "methods", element: <Methods /> },
      { path: "stats", element: <Stats /> },
      { path: "*", element: <NotFound /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <I18nProvider>
      <RouterProvider router={router} />
    </I18nProvider>
  </StrictMode>,
);
