import { NavLink, Route, Routes } from 'react-router-dom';

import { EventDetail } from './routes/EventDetail';
import { LiveFeed } from './routes/LiveFeed';
import { RagChat } from './routes/RagChat';

export function App() {
  return (
    <div className="shell">
      <nav className="rail" aria-label="Điều hướng chính">
        <div className="rail__brand">
          <strong>Giám sát âm thanh</strong>
          <span>walking skeleton · w1</span>
        </div>

        <div className="rail__nav">
          <NavLink className="rail__link" to="/" end>
            Băng sự kiện
          </NavLink>
          <NavLink className="rail__link" to="/hoi">
            Hỏi lịch sử
          </NavLink>
        </div>

        <div className="rail__foot">
          <span>PANNs CNN14 + mô tả khuôn mẫu</span>
          <span>Grounded AAC: W4–W5</span>
        </div>
      </nav>

      <main className="main">
        <Routes>
          <Route path="/" element={<LiveFeed />} />
          <Route path="/events/:eventId" element={<EventDetail />} />
          <Route path="/hoi" element={<RagChat />} />
          <Route
            path="*"
            element={<div className="empty">Không có trang này.</div>}
          />
        </Routes>
      </main>
    </div>
  );
}
