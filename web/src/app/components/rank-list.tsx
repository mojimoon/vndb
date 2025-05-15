"use client";

import { FullOrder } from '@/app/common/struct';
import { MaterialReactTable, MRT_ColumnDef } from 'material-react-table';
import { Box, Container, TextField, Typography } from '@mui/material';
import { useEffect, useState } from 'react';

const alignRight = {
  align: 'right',
};

const columns: MRT_ColumnDef<FullOrder>[] = [
  { accessorKey: 'idx', header: '序号', maxSize: 80 },
  { accessorKey: 'vid', header: 'VNDB ID', maxSize: 80,
    Cell: ({ cell }) => {
      const { row } = cell;
      const { original } = row;
      return (
        <Typography className="text-blue-500">
          <a href={`https://vndb.org/${original.vid}`} target="_blank" rel="noopener noreferrer">
            {original.vid}
          </a>
        </Typography>
      );
    },
  },
  { accessorKey: 'title_ja', header: '日文标题', maxSize: 160 },
  { accessorKey: 'title_en', header: '英文标题', maxSize: 160 },
  { accessorKey: 'title_zh', header: '中文标题', maxSize: 160 },
  { accessorKey: 'alias', header: '别名', maxSize: 160 },
  { accessorKey: 'total', header: '合计积分', maxSize: 80, muiTableBodyCellProps: { align: 'right' } },
  { accessorKey: 'percentage', header: '比例积分', maxSize: 80, muiTableBodyCellProps: { align: 'right' } },
  { accessorKey: 'simple', header: '简单积分', maxSize: 80, muiTableBodyCellProps: { align: 'right' } },
  { accessorKey: 'weighted_simple', header: '加权简单', maxSize: 80, muiTableBodyCellProps: { align: 'right' } },
  { accessorKey: 'pagerank', header: 'PageRank', maxSize: 80, muiTableBodyCellProps: { align: 'right' } },
  { accessorKey: 'elo', header: 'Elo', maxSize: 80, muiTableBodyCellProps: { align: 'right' } },
  { accessorKey: 'entropy', header: '熵值', maxSize: 80, muiTableBodyCellProps: { align: 'right' } },
  // { accessorKey: 'appear', header: '出现次数', maxSize: 80, muiTableBodyCellProps: { align: 'right' } },
  // { accessorKey: 'id', header: 'ID', maxSize: 80, muiTableBodyCellProps: { align: 'right' } },
  { accessorKey: 'c_votecount', header: '投票数', maxSize: 80, muiTableBodyCellProps: { align: 'right' } },
  { accessorKey: 'c_rating', header: '评分', maxSize: 80, muiTableBodyCellProps: { align: 'right' } },
];

export default function RankList() {
  const [data, setData] = useState<FullOrder[]>([]);
  const [search, setSearch] = useState<string | null>(null);
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
          label="搜索"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && fetchData()}
          onBlur={() => fetchData()}
          variant="outlined"
          size="small"
          sx={{ width: 320 }}
        />
      </Box>
      <MaterialReactTable
        columns={columns}
        data={data}
        enableColumnActions={false}
        sortDescFirst={false}
        state={{isLoading}}
        initialState={{
          columnVisibility: {
            idx: false,
            id: false,
            alias: false,
            title_en: false,
            c_rating: false,
            c_votecount: false,
          },
          density: 'compact',
          pagination: {
            pageIndex: 0,
            pageSize: 30,
          },
        }}
      />
    </Container>
  );
}