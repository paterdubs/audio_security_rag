import { Link, useSearchParams } from 'react-router-dom';

import { QueryState } from '../components/QueryState';
import { SeverityTag } from '../components/SeverityTag';
import { UploadPanel } from '../components/UploadPanel';
import { useAlertStream } from '../hooks/useAlertStream';
import { formatDateTime, SEVERITY_ORDER, severityLabel } from '../lib/format';
import { useEvents, useTaxonomy } from '../lib/queries';

export function LiveFeed() {
  // Bộ lọc sống trong URL: dán link là người khác thấy đúng cái mình đang xem, và nút
  // Back của trình duyệt hoạt động đúng nghĩa.
  const [params, setParams] = useSearchParams();
  const filters = {
    severity: params.get('severity') ?? undefined,
    location: params.get('location') ?? undefined,
    class: params.get('class') ?? undefined,
  };

  const events = useEvents(filters);
  const taxonomy = useTaxonomy();
  const { alerts, state } = useAlertStream();

  const setFilter = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next, { replace: true });
  };

  return (
    <>
      <header className="page-head">
        <div>
          <h1>Băng sự kiện</h1>
          <p>
            Cảnh báo đi thẳng từ chấm điểm rủi ro ra WebSocket, không qua mô hình ngôn ngữ —
            nên nó không chờ ai và không bịa gì.
          </p>
        </div>
        <span className="stream-state" data-state={state}>
          {state === 'open' ? 'đang nhận' : state === 'connecting' ? 'đang nối' : 'mất kết nối'}
        </span>
      </header>

      {alerts.length > 0 && (
        <div className="alert-strip" aria-live="polite" aria-label="Cảnh báo trực tiếp">
          {alerts.map((alert) => (
            <Link
              key={alert.event_id}
              to={`/events/${alert.event_id}`}
              className="alert-card"
              data-sev={alert.severity}
            >
              <span className="alert-card__id">
                {alert.event_id} · {formatDateTime(alert.window_start)}
              </span>
              <p className="alert-card__caption">{alert.caption_vi}</p>
            </Link>
          ))}
        </div>
      )}

      <UploadPanel />

      <div className="toolbar" style={{ marginTop: 'var(--space-8)' }}>
        <label className="field">
          <span>Mức độ</span>
          <select value={filters.severity ?? ''} onChange={(e) => setFilter('severity', e.target.value)}>
            <option value="">Tất cả</option>
            {SEVERITY_ORDER.map((severity) => (
              <option key={severity} value={severity}>
                {severityLabel(severity)}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span>Lớp âm thanh</span>
          <select value={filters.class ?? ''} onChange={(e) => setFilter('class', e.target.value)}>
            <option value="">Tất cả</option>
            {(taxonomy.data?.classes ?? []).map((item) => (
              <option key={item.id} value={item.id}>
                {item.vi}
              </option>
            ))}
          </select>
        </label>
      </div>

      <QueryState
        isLoading={events.isLoading}
        error={events.error}
        isEmpty={events.data?.length === 0}
        emptyMessage="Chưa có sự kiện nào khớp bộ lọc. Nạp một file .wav ở trên để thử."
      >
        <div className="table-scroll">
          <table className="ledger">
            <thead>
              <tr>
                <th scope="col">Mã sự kiện</th>
                <th scope="col">Thời điểm</th>
                <th scope="col">Vị trí</th>
                <th scope="col">Mô tả</th>
                <th scope="col">Mức độ</th>
                <th scope="col" style={{ textAlign: 'right' }}>
                  Điểm rủi ro
                </th>
              </tr>
            </thead>
            <tbody>
              {(events.data ?? []).map((event) => (
                <tr key={event.event_id}>
                  <td>
                    <Link className="ledger__id" to={`/events/${event.event_id}`}>
                      {event.event_id}
                    </Link>
                  </td>
                  <td className="ledger__time">{formatDateTime(event.window_start)}</td>
                  <td>{event.location?.name ?? '—'}</td>
                  <td>{event.caption_vi}</td>
                  <td>
                    <SeverityTag severity={event.severity} />
                  </td>
                  <td className="ledger__score">{event.risk_score.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </QueryState>
    </>
  );
}
