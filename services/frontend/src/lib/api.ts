/** Một chỗ duy nhất bóc envelope và xử lý lỗi — đúng lý do envelope tồn tại (§4.4). */

import type { Envelope } from './types';

export const API_PREFIX = '/api/v1';

export class ApiError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Bóc `{success, data, error, meta}` và ném ApiError khi thất bại.
 *
 * Không được rơi về `response.ok` suông: API trả lỗi nghiệp vụ bên trong envelope kèm mã
 * riêng (`http_410` khi audio hết hạn, `validation_error`...), và chính cái mã đó là thứ
 * UI cần để nói cho người dùng biết chuyện gì xảy ra. Nuốt nó rồi hiện "Có lỗi xảy ra" là
 * vứt đi thông tin mà backend đã cố ý gửi lên.
 */
async function unwrap<T>(response: Response): Promise<T> {
  let body: Envelope<T> | null = null;
  try {
    body = (await response.json()) as Envelope<T>;
  } catch {
    // Không parse được JSON (proxy chết, HTML lỗi của nginx...) — nói đúng như vậy.
    throw new ApiError('invalid_response', `Máy chủ trả về dữ liệu không đọc được (HTTP ${response.status})`, response.status);
  }

  if (!body.success || body.data === null) {
    throw new ApiError(
      body.error?.code ?? `http_${response.status}`,
      body.error?.message ?? `Yêu cầu thất bại (HTTP ${response.status})`,
      response.status,
    );
  }
  return body.data;
}

export async function apiGet<T>(path: string): Promise<T> {
  return unwrap<T>(await fetch(`${API_PREFIX}${path}`));
}

export async function apiPostJson<T>(path: string, payload: unknown): Promise<T> {
  return unwrap<T>(
    await fetch(`${API_PREFIX}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  );
}

export async function apiPostForm<T>(path: string, form: FormData): Promise<T> {
  // KHÔNG đặt Content-Type bằng tay: trình duyệt phải tự sinh `boundary` của multipart,
  // đặt tay là mất boundary và server không tách được các phần.
  return unwrap<T>(await fetch(`${API_PREFIX}${path}`, { method: 'POST', body: form }));
}

/** URL WebSocket cảnh báo, suy ra từ origin hiện tại (ws: hay wss: theo trang). */
export function alertStreamUrl(): string {
  const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${window.location.host}${API_PREFIX}/alerts/stream`;
}

export function eventAudioUrl(eventId: string): string {
  return `${API_PREFIX}/events/${eventId}/audio`;
}
