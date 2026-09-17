/** Đường NÓNG: cảnh báo đẩy thẳng qua WebSocket, không đi qua LLM (SYSTEM.md §4.1). */

import { useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';

import { alertStreamUrl } from '../lib/api';
import type { AlertMessage } from '../lib/types';

const MAX_ALERTS = 20;
const RECONNECT_DELAY_MS = 3000;

export type StreamState = 'connecting' | 'open' | 'closed';

export function useAlertStream() {
  const [alerts, setAlerts] = useState<AlertMessage[]>([]);
  const [state, setState] = useState<StreamState>('connecting');
  const client = useQueryClient();
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      setState('connecting');
      socket = new WebSocket(alertStreamUrl());

      socket.onopen = () => setState('open');

      socket.onmessage = (message) => {
        let alert: AlertMessage;
        try {
          alert = JSON.parse(message.data as string) as AlertMessage;
        } catch {
          // Bản tin hỏng không được làm sập dashboard: bỏ qua gói này, giữ kết nối.
          return;
        }
        // Chỉ giữ một cửa sổ ngắn: đây là băng cảnh báo trực tiếp, còn lịch sử đầy đủ
        // đã nằm ở bảng sự kiện bên dưới — không nhân đôi dữ liệu.
        setAlerts((previous) => [alert, ...previous].slice(0, MAX_ALERTS));
        void client.invalidateQueries({ queryKey: ['events'] });
      };

      socket.onclose = () => {
        setState('closed');
        // Nối lại: api có thể đang khởi động lại. Mất kết nối im lặng nguy hiểm hơn hẳn
        // ở một dashboard an ninh — trạng thái `state` hiện ra trên thanh đầu trang.
        if (!disposed) {
          timerRef.current = window.setTimeout(connect, RECONNECT_DELAY_MS);
        }
      };
    };

    connect();

    return () => {
      disposed = true;
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      socket?.close();
    };
  }, [client]);

  return { alerts, state };
}
