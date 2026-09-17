import { useState } from 'react';
import { Link } from 'react-router-dom';

import { SeverityTag } from '../components/SeverityTag';
import { ApiError } from '../lib/api';
import { formatDateTime } from '../lib/format';
import { useRagQuery } from '../lib/queries';
import type { RagAnswer } from '../lib/types';

interface Turn {
  question: string;
  answer: RagAnswer | null;
  error: string | null;
}

const GOI_Y = [
  'đêm qua có sự kiện gì nghiêm trọng',
  'có tiếng kính vỡ ở hành lang không',
  'có sự cố gì ở nhà xe không',
];

export function RagChat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState('');
  const ask = useRagQuery();

  const send = (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || ask.isPending) return;
    setQuestion('');
    ask.mutate(trimmed, {
      onSuccess: (answer) => setTurns((prev) => [...prev, { question: trimmed, answer, error: null }]),
      onError: (error) =>
        setTurns((prev) => [
          ...prev,
          {
            question: trimmed,
            answer: null,
            error: error instanceof ApiError ? `${error.message} (${error.code})` : String(error),
          },
        ]),
    });
  };

  return (
    <>
      <header className="page-head">
        <div>
          <h1>Hỏi lịch sử</h1>
          <p>
            Mỗi câu trả lời phải kèm trích dẫn trỏ tới sự kiện có thật. Không tìm được bằng
            chứng thì hệ thống nói thẳng là không có, chứ không suy đoán.
          </p>
        </div>
      </header>

      <div className="chat">
        <div className="chat__log">
          {turns.length === 0 && (
            <div className="empty">
              <p>Chưa có câu hỏi nào. Thử một trong các câu sau:</p>
              <div className="toolbar" style={{ justifyContent: 'center', marginTop: 'var(--space-4)' }}>
                {GOI_Y.map((goi) => (
                  <button key={goi} className="btn btn--ghost" type="button" onClick={() => send(goi)}>
                    {goi}
                  </button>
                ))}
              </div>
            </div>
          )}

          {turns.map((turn, index) => (
            <div key={index} style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
              <div className="bubble bubble--user">{turn.question}</div>

              {turn.error ? (
                <div className="notice notice--error" role="alert">
                  {turn.error}
                </div>
              ) : (
                turn.answer && <AnswerBlock answer={turn.answer} />
              )}
            </div>
          ))}

          {ask.isPending && <div className="empty">Đang tra cứu bằng chứng…</div>}
        </div>

        <form
          className="chat__form"
          onSubmit={(e) => {
            e.preventDefault();
            send(question);
          }}
        >
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Hỏi bằng tiếng Việt, ví dụ: tối qua có tiếng súng không"
            rows={2}
            onKeyDown={(e) => {
              // Enter gửi, Shift+Enter xuống dòng — thói quen của mọi khung chat.
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                send(question);
              }
            }}
          />
          <button className="btn" type="submit" disabled={!question.trim() || ask.isPending}>
            Gửi
          </button>
        </form>
      </div>

      <p className="chat__hint">
        Gõ tiếng Việt <strong>có dấu</strong>. Mô hình nhúng coi chuỗi không dấu là chuỗi
        khác hẳn, nên câu hỏi không dấu sẽ bị ngưỡng bằng chứng từ chối (đo thật: 0.34 so với
        0.68 cho cùng một câu).
      </p>
    </>
  );
}

function AnswerBlock({ answer }: { answer: RagAnswer }) {
  return (
    <div className="bubble bubble--bot">
      <div className="bubble__answer">{answer.answer}</div>

      {answer.citations.length > 0 ? (
        <div className="citations">
          <span className="citations__title">
            Trích dẫn ({answer.citations.length}) · nguồn: {answer.provider}
          </span>
          {answer.citations.map((citation) => (
            <Link key={citation.event_id} className="citation" to={`/events/${citation.event_id}`}>
              <span className="citation__id">{citation.event_id}</span>
              <SeverityTag severity={citation.severity} />
              <span>{citation.caption_vi}</span>
              <span className="citation__sim">
                {formatDateTime(citation.window_start)} · {citation.similarity.toFixed(3)}
              </span>
            </Link>
          ))}
        </div>
      ) : (
        <p className="chat__hint">
          Không có trích dẫn vì không sự kiện nào vượt ngưỡng bằng chứng — đây là hành vi
          đúng, không phải lỗi.
        </p>
      )}
    </div>
  );
}
