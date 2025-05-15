import { MaterialReactTable, MRT_ColumnDef } from 'material-react-table';
import React from 'react';
import { FullOrder } from '@/app/common/struct';

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
  { accessorKey: 'title_ja', header: '日文标题', maxSize: 160, 
    Cell: ({ cell }) => (
      <div style={{ whiteSpace: 'normal', wordBreak: 'break-all' }}>
        {cell.getValue() as string}
      </div>
    )
  },
  { accessorKey: 'title_en', header: '英文标题', maxSize: 160, 
    Cell: ({ cell }) => (
      <div style={{ whiteSpace: 'normal', wordBreak: 'break-all' }}>
        {cell.getValue() as string}
      </div>
    )
  },
  { accessorKey: 'title_zh', header: '中文标题', maxSize: 160, 
    Cell: ({ cell }) => (
      <div style={{ whiteSpace: 'normal', wordBreak: 'break-all' }}>
        {cell.getValue() as string}
      </div>
    )
  },
  { accessorKey: 'alias', header: '别名', maxSize: 160, 
    Cell: ({ cell }) => {
      const _ = cell.getValue() as string;
      if (!_) return null;
      const parts = _.split('\\n');
      return (
        <div style={{ whiteSpace: 'normal', wordBreak: 'break-all' }}>
          {parts.map((item, index) => (
            <span key={index} className="text-gray-500">
              {item}
              {index < parts.length - 1 && <br />}
            </span>
          ))}
        </div>
      );
    }
  },
  // { accessorKey: 'search', header: '搜索', maxSize: 160 },
  { accessorKey: 'c_votecount', header: '评分数', maxSize: 80, muiTableBodyCellProps: { align: 'right' },},
  { accessorKey: 'rank', header: '排名', maxSize: 80, muiTableBodyCellProps: { align: 'right' },
    Cell: ({ cell }) => `#${cell.getValue()}`
  },
  { accessorKey: 'c_rating', header: '平均', maxSize: 80, muiTableBodyCellProps: { align: 'right' },
    Cell: ({ cell }) => (cell.getValue() as number).toFixed(2),
  },
  { accessorKey: 'c_average', header: '真实平均', maxSize: 80, muiTableBodyCellProps: { align: 'right' },
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

export default React.memo(function MemoTable({
    data,
    isLoading,
    }: {
    data: FullOrder[],
    isLoading: boolean,
    }) {
  return (
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
  );
});
