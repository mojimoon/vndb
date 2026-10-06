import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router";
import "./index.css";
import { I18nProvider } from "./lib/i18n";
import Layout from "./components/Layout";
import Ranking from "./pages/Ranking";
import VnDetail from "./pages/VnDetail";
import Methods from "./pages/Methods";
import Stats from "./pages/Stats";

const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { index: true, element: <Ranking /> },
      { path: "vn/:id", element: <VnDetail /> },
      { path: "methods", element: <Methods /> },
      { path: "stats", element: <Stats /> },
      { path: "*", element: <Ranking /> },
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
