import type { ReactNode } from 'react';

import { ApiError } from '../lib/api';

interface Props {
  isLoading: boolean;
  error: unknown;
  isEmpty?: boolean;
  emptyMessage?: string;
  children: ReactNode;
}

/**
 * Ba trạng thái của một truy vấn, xử lý ở MỘT chỗ.
 *
 * Lỗi hiện đúng thông điệp backend gửi lên chứ không phải "Có lỗi xảy ra": ApiError giữ
 * nguyên `code` và `message` từ envelope, và ở một hệ giám sát thì biết chính xác hỏng ở
 * đâu quan trọng hơn là một câu báo lỗi cho đẹp.
 */
export function QueryState({ isLoading, error, isEmpty, emptyMessage, children }: Props) {
  if (isLoading) {
    return <div className="empty">Đang tải…</div>;
  }
  if (error) {
    const detail = error instanceof ApiError ? `${error.message} (${error.code})` : String(error);
    return (
      <div className="notice notice--error" role="alert">
        Không lấy được dữ liệu: {detail}
      </div>
    );
  }
  if (isEmpty) {
    return <div className="empty">{emptyMessage ?? 'Chưa có dữ liệu.'}</div>;
  }
  return <>{children}</>;
}
