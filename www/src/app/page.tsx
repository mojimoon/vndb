import RankList from "./_components/rank-list";

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <h1 className="text-4xl font-bold">Hello, world!</h1>
      <RankList />
    </main>
  );
}