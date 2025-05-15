"use client";

import { FullOrder } from '@/app/common/struct';
import { Box, Container, TextField, Typography } from '@mui/material';
import { useEffect, useState } from 'react';
import MemoTable from './memo-table';

export default function RankList() {
  const [data, setData] = useState<FullOrder[]>([]);
  const [search, setSearch] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  
  const fetchData = async () => {
    setIsLoading(true);
    let response;
    if (search) {
      response = await fetch(`/api/list?q=${search}`);
    } else {
      response = await fetch('/api/list');
    }
    if (!response.ok) {
      console.error('Failed to fetch data');
      setIsLoading(false);
      return;
    }
    const result = await response.json();
    setData(result.data);
    setIsLoading(false);
  }

  useEffect(() => {
    fetchData();
  }, []);

  return (
    <Container maxWidth="xl" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom className="text-center">
        VNDB PONet 排行榜
      </Typography>
      <Box component="form" sx={{ mb: 2 }} display="flex" justifyContent="center">
        <TextField
          label="搜索 (可输入中/英/日文标题/简称/别名)"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && fetchData()}
          onBlur={() => fetchData()}
          variant="outlined"
        />
      </Box>
      <MemoTable data={data} isLoading={isLoading} />
    </Container>
  );
}