"use client";

import { FullOrder } from '@/app/common/struct';
import { MaterialReactTable, MRT_ColumnDef } from 'material-react-table';
import { Box, Container, TextField, Typography } from '@mui/material';
import { useEffect, useState } from 'react';

// export type FullOrder = {
//   idx: string;
//   vid: string;
//   total: number;
//   percentage: number;
//   simple: number;
//   weighted_simple: number;
//   pagerank: number;
//   elo: number;
//   entropy: number;
//   title_ja: string;
//   title_en: string;
//   title_zh: string;
//   alias: string;
//   search: string;
//   c_votecount: number;
//   c_rating: number;
//   c_average: number;
//   rank: number;
// };

const columns: MRT_ColumnDef<FullOrder>[] = [
  // { accessorKey: 'idx', header: '序号', maxSize: 80 },
  { accessorKey: 'vid', header: 'VNDB ID', maxSize: 80,
    Cell: ({ cell }) => {
      const vid = cell.getValue() as string;
      return (
        <span className="text-blue-500">
          <a href={`https://vndb.org/${vid}`} target="_blank" rel="noopener noreferrer">
            {vid}
          </a>
        </span>
      );
    },
  },
  { accessorKey: 'title_ja', header: '日文标题', maxSize: 160 },
  { accessorKey: 'title_en', header: '英文标题', maxSize: 160 },
  { accessorKey: 'title_zh', header: '中文标题', maxSize: 160 },
  { accessorKey: 'alias', header: '别名', maxSize: 160 },
  // { accessorKey: 'search', header: '搜索', maxSize: 160 },
  // { accessorKey: 'rank', header: '排名', maxSize: 80 },
  { accessorKey: 'c_votecount', header: '评分数', maxSize: 80, muiTableBodyCellProps: { align: 'right' },},
  { accessorKey: 'c_rating', header: 'VNDB 平均', maxSize: 80, muiTableBodyCellProps: { align: 'right' },
    Cell: ({ cell }) => (cell.getValue() as number).toFixed(2),
  },
  { accessorKey: 'c_average', header: '平均', maxSize: 80, muiTableBodyCellProps: { align: 'right' },
    Cell: ({ cell }) => (cell.getValue() as number).toFixed(2),
  },
  { accessorKey: 'total', header: '合计', maxSize: 80, muiTableBodyCellProps: { align: 'right' },
    Cell: ({ cell }) => {
      const _ = cell.getValue() as number;
      return <span className={_ > 5e-4 ? 'text-green-800' : _ < -5e-4 ? 'text-red-800' : ''}>{_.toFixed(3)}</span>;
    }
  },
  { accessorKey: 'percentage', header: '比例', maxSize: 80, muiTableBodyCellProps: { align: 'right' },
    Cell: ({ cell }) => {
      const _ = cell.getValue() as number;
      return <span className={_ > 5e-4 ? 'text-green-800' : _ < -5e-4 ? 'text-red-800' : ''}>{_.toFixed(3)}</span>;
    }
  },
  { accessorKey: 'simple', header: '简单', maxSize: 80, muiTableBodyCellProps: { align: 'right' },
    Cell: ({ cell }) => {
      const _ = cell.getValue() as number;
      return <span className={_ > 5e-4 ? 'text-green-800' : _ < -5e-4 ? 'text-red-800' : ''}>{_.toFixed(3)}</span>;
    }
  },
  { accessorKey: 'weighted_simple', header: '加权', maxSize: 80, muiTableBodyCellProps: { align: 'right' },
    Cell: ({ cell }) => {
      const _ = cell.getValue() as number;
      return <span className={_ > 5e-4 ? 'text-green-800' : _ < -5e-4 ? 'text-red-800' : ''}>{_.toFixed(3)}</span>;
    }
  },
  { accessorKey: 'pagerank', header: 'PageRank', maxSize: 80, muiTableBodyCellProps: { align: 'right' },
    Cell: ({ cell }) => (cell.getValue() as number).toFixed(0),
  },
  { accessorKey: 'elo', header: 'ELO', maxSize: 80, muiTableBodyCellProps: { align: 'right' },
    Cell: ({ cell }) => (cell.getValue() as number).toFixed(0),
  },
  { accessorKey: 'entropy', header: '熵值', maxSize: 80, muiTableBodyCellProps: { align: 'right' },
    Cell: ({ cell }) => (cell.getValue() as number).toFixed(3),
  },
];

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
          size="small"
          sx={{ width: 320 }}
        />
      </Box>
      <MaterialReactTable
        columns={columns}
        data={data}
        enableColumnActions={false}
        enableRowNumbers
        enableGlobalFilter={false}
        state={{isLoading}}
        initialState={{
          columnVisibility: {
            idx: false,
            alias: false,
            title_en: false,
            c_average: false,
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