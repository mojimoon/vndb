"use client";

import { createClient } from "@/utils/supabase/client";
export default async function Instruments() {
  const supabase = createClient();
  const { data: instruments } = await supabase.from("instruments").select();
  return <pre>{JSON.stringify(instruments, null, 2)}</pre>
}