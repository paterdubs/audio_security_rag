import { Link, useParams } from 'react-router-dom';

import { QueryState } from '../components/QueryState';
import { SeverityTag } from '../components/SeverityTag';
import { eventAudioUrl } from '../lib/api';
import { formatDateTime, formatSpan, percent } from '../lib/format';
import { useEvent, useTaxonomy } from '../lib/queries';

export function EventDetail() {
  const { eventId = '' } = useParams();
  const event = useEvent(eventId);
  const taxonomy = useTaxonomy();

  const viName = (classId: string) =>
    taxonomy.data?.classes.find((c) => c.id === classId)?.vi ?? classId;

  return (
    <>
      <header className="page-head">
        <div>
          <Link className="ledger__id" to="/">
            ← Về băng sự kiện
          </Link>
          <h1 style={{ marginTop: 'var(--space-2)' }}>{eventId}</h1>
        </div>
        {event.data && <SeverityTag severity={event.data.severity} />}
      </header>

      <QueryState isLoading={event.isLoading} error={event.error}>
        {event.data && (
          <div className="detail-grid">
            <section className="panel">
              <h2 className="panel__title">Mô tả sự kiện</h2>
              <p className="caption-vi">{event.data.caption_vi}</p>
              <p className="caption-en">{event.data.caption_en}</p>

              <h2 className="panel__title" style={{ marginTop: 'var(--space-8)' }}>
                Bằng chứng phát hiện ({event.data.detections.length})
              </h2>
              {event.data.detections.length === 0 ? (
                <p className="notice">
                  Không lớp nào vượt ngưỡng phát hiện. Đây là kết quả hợp lệ, không phải lỗi —
                  bản ghi được lưu để đối chiếu.
                </p>
              ) : (
                <Timeline
                  detections={event.data.detections}
                  duration={durationOf(event.data.window_start, event.data.window_end)}
                  viName={viName}
                />
              )}

              <h2 className="panel__title" style={{ marginTop: 'var(--space-8)' }}>
                Phát lại
              </h2>
              {event.data.has_audio ? (
                <audio controls preload="none" src={eventAudioUrl(event.data.event_id)}>
                  Trình duyệt không phát được audio.
                </audio>
              ) : (
                <p className="notice">
                  Audio không còn: đã hết thời hạn lưu trữ, hoặc hệ thống được cấu hình chỉ
                  giữ metadata và mô tả (SYSTEM.md §1.4).
                </p>
              )}
            </section>

            <aside className="panel">
              <h2 className="panel__title">Siêu dữ liệu</h2>
              <dl className="meta-list">
                <dt>Bắt đầu</dt>
                <dd>{formatDateTime(event.data.window_start)}</dd>
                <dt>Kết thúc</dt>
                <dd>{formatDateTime(event.data.window_end)}</dd>
                <dt>Vị trí</dt>
                <dd>{event.data.location?.name ?? '—'}</dd>
                <dt>Điểm rủi ro</dt>
                <dd>{event.data.risk_score.toFixed(4)}</dd>
                <dt>Grounding</dt>
                <dd>
                  {/* null ≠ 0: chưa đo grounding thì phải nói "chưa đo", không hiện 0.0000
                      rồi để người đọc tưởng mô tả hoàn toàn không bám bằng chứng. */}
                  {event.data.grounding_score === null
                    ? 'chưa đo (W4)'
                    : event.data.grounding_score.toFixed(4)}
                </dd>
              </dl>

              <h2 className="panel__title" style={{ marginTop: 'var(--space-6)' }}>
                Phiên bản mô hình
              </h2>
              <dl className="meta-list">
                {Object.entries(event.data.model_versions).map(([key, value]) => (
                  <div key={key} style={{ display: 'contents' }}>
                    <dt>{key}</dt>
                    <dd style={{ wordBreak: 'break-all' }}>{value}</dd>
                  </div>
                ))}
              </dl>

              <p className="placeholder-banner">
                Bản W1 dùng PANNs CNN14 + mô tả theo khuôn mẫu làm nền tham chiếu (baseline B0).
                Mô hình Grounded AAC của đề tài là việc của W4–W5 — đừng đọc số đo ở đây thành
                kết quả nghiên cứu.
              </p>
            </aside>
          </div>
        )}
      </QueryState>
    </>
  );
}

function durationOf(start: string, end: string): number {
  const seconds = (new Date(end).getTime() - new Date(start).getTime()) / 1000;
  // Cửa sổ 0 giây sẽ làm phép chia dưới đây ra Infinity và thanh thời gian biến mất.
  return seconds > 0 ? seconds : 1;
}

interface TimelineProps {
  detections: { class_id: string; onset: number; offset: number; confidence: number }[];
  duration: number;
  viName: (classId: string) => string;
}

function Timeline({ detections, duration, viName }: TimelineProps) {
  return (
    <div className="timeline">
      {detections.map((detection, index) => {
        const left = Math.max(0, Math.min(100, (detection.onset / duration) * 100));
        const width = Math.max(2, Math.min(100 - left, ((detection.offset - detection.onset) / duration) * 100));
        return (
          <div className="timeline__row" key={`${detection.class_id}-${index}`}>
            <span>
              {viName(detection.class_id)}
              <br />
              <span className="ledger__time">{formatSpan(detection.onset, detection.offset)}</span>
            </span>
            <div
              className="timeline__track"
              role="img"
              aria-label={`${viName(detection.class_id)} từ ${detection.onset.toFixed(2)} đến ${detection.offset.toFixed(2)} giây`}
            >
              <div className="timeline__span" style={{ left: `${left}%`, width: `${width}%` }} />
            </div>
            <span className="timeline__conf">{percent(detection.confidence)}</span>
          </div>
        );
      })}
    </div>
  );
}
