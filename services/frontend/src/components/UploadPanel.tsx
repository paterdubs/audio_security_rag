import { useState } from 'react';

import { ApiError } from '../lib/api';
import { percent } from '../lib/format';
import { useLocations, useUploadAudio } from '../lib/queries';
import { SeverityTag } from './SeverityTag';

const DEFAULT_LOCATION = 'HALL_03';

export function UploadPanel() {
  const locations = useLocations();
  const upload = useUploadAudio();
  const [file, setFile] = useState<File | null>(null);
  const [locationId, setLocationId] = useState(DEFAULT_LOCATION);

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (file) upload.mutate({ file, locationId });
  };

  return (
    <section className="panel" aria-labelledby="upload-title">
      <h2 className="panel__title" id="upload-title">
        Nạp bản ghi
      </h2>
      <form className="toolbar" onSubmit={submit}>
        <label className="field">
          <span>File âm thanh</span>
          <input
            type="file"
            accept="audio/wav,audio/x-wav,.wav"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </label>
        <label className="field">
          <span>Vị trí</span>
          <select value={locationId} onChange={(e) => setLocationId(e.target.value)}>
            {(locations.data ?? []).map((location) => (
              <option key={location.id} value={location.id}>
                {location.name}
              </option>
            ))}
          </select>
        </label>
        <button className="btn" type="submit" disabled={!file || upload.isPending}>
          {upload.isPending ? 'Đang phân tích…' : 'Phân tích'}
        </button>
      </form>

      {upload.isError && (
        <div className="notice notice--error" role="alert">
          {upload.error instanceof ApiError
            ? `${upload.error.message} (${upload.error.code})`
            : String(upload.error)}
        </div>
      )}

      {upload.isSuccess && (
        <div className="notice notice--ok" role="status">
          <strong>{upload.data.event_id}</strong> · <SeverityTag severity={upload.data.severity} />{' '}
          {upload.data.caption_vi}{' '}
          {upload.data.detections.length > 0 && (
            <>
              — {upload.data.detections.map((d) => `${d.class_id} ${percent(d.confidence)}`).join(', ')}
            </>
          )}
          {/* Không có detection là kết quả HỢP LỆ (mọi lớp dưới ngưỡng), không phải lỗi —
              nói rõ ra để người dùng không tưởng hệ thống hỏng. */}
          {upload.data.detections.length === 0 && ' — không lớp nào vượt ngưỡng phát hiện'}
        </div>
      )}
    </section>
  );
}
