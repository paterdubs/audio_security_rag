/** Định dạng hiển thị. Mọi mốc thời gian quy về giờ Việt Nam. */

import type { Severity } from './types';

/**
 * API trả UTC có hậu tố `Z` (đúng chuẩn). Ta ép hiển thị theo `Asia/Ho_Chi_Minh` thay vì
 * để trình duyệt tự lấy múi giờ máy: một sự kiện lúc 01:05 sáng 17/09 giờ VN, xem trên máy
 * đặt nhầm múi giờ, sẽ hiện thành "18:05 ngày 16/09" — sai cả giờ lẫn NGÀY, mà người xem
 * không có cách nào nhận ra. Câu trả lời RAG do backend sinh cũng dùng đúng múi giờ này
 * (`app/rag/document.py::LOCAL_TZ`), nên hai chỗ phải khớp nhau.
 */
export const VN_TIMEZONE = 'Asia/Ho_Chi_Minh';

const DATE_TIME = new Intl.DateTimeFormat('vi-VN', {
  timeZone: VN_TIMEZONE,
  day: '2-digit',
  month: '2-digit',
  year: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
});

const TIME_ONLY = new Intl.DateTimeFormat('vi-VN', {
  timeZone: VN_TIMEZONE,
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
});

export function formatDateTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return DATE_TIME.format(date);
}

export function formatTime(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return TIME_ONLY.format(date);
}

export const SEVERITY_VI: Record<Severity, string> = {
  LOW: 'Thấp',
  MEDIUM: 'Trung bình',
  HIGH: 'Cao',
  CRITICAL: 'Nghiêm trọng',
};

export const SEVERITY_ORDER: Severity[] = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];

export function severityLabel(severity: string): string {
  return SEVERITY_VI[severity as Severity] ?? severity;
}

/** 0.7734 → "77%" — độ tin cậy đọc bằng phần trăm dễ hơn số thập phân. */
export function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/** Khoảng thời gian của một detection, tương đối so với đầu cửa sổ sự kiện. */
export function formatSpan(onset: number, offset: number): string {
  return `${onset.toFixed(2)}s – ${offset.toFixed(2)}s`;
}
