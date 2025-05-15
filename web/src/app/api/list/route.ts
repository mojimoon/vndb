// path: /api/list

import { NextResponse } from 'next/server';
import { FullOrder } from '@/app/common/struct';
import path from 'path';
import fs from 'fs';
import Papa from 'papaparse';

const fullOrderPath = path.join(process.cwd(), 'public', 'out', 'full_order.csv');

export async function GET(request: Request) {
  
  const csv = fs.readFileSync(fullOrderPath, 'utf-8');
  const parsedData = Papa.parse<FullOrder>(csv, {
    header: true,
    skipEmptyLines: true,
    dynamicTyping: true,
  });

  const { searchParams } = new URL(request.url);
  const q = searchParams.get('q')?.trim().toLowerCase() || '';

  let filtered = parsedData.data;

  if (q) {
    filtered = parsedData.data.filter((item) => item.search.includes(q));
  }

  return NextResponse.json({
    data: filtered,
  });
}