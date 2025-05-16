import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VNDB PONet Ranking",
  description: "Partial Order Network ranking of visual novels on VNDB",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
