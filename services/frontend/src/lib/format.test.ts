import { describe, expect, it } from 'vitest';

import { formatDateTime, formatSpan, percent, severityLabel } from './format';

describe('formatDateTime', () => {
  it('quy UTC về giờ Việt Nam, không phụ thuộc múi giờ của máy', () => {
    // 18:05 UTC ngày 16/09 = 01:05 ngày 17/09 giờ VN — đổi cả giờ lẫn NGÀY.
    // Đây chính là lỗi đã gặp ở phía backend, nên phải có test chặn ở phía frontend.
    expect(formatDateTime('2026-09-16T18:05:16.869561Z')).toContain('01:05');
    expect(formatDateTime('2026-09-16T18:05:16.869561Z')).toContain('17/09/2026');
  });

  it('trả lại nguyên chuỗi khi không phải mốc thời gian hợp lệ', () => {
    // Hiện chuỗi thô còn hơn hiện "Invalid Date" — người xem biết dữ liệu có vấn đề.
    expect(formatDateTime('khong-phai-ngay')).toBe('khong-phai-ngay');
  });
});

describe('severityLabel', () => {
  it('dịch mức độ sang tiếng Việt', () => {
    expect(severityLabel('CRITICAL')).toBe('Nghiêm trọng');
  });

  it('giữ nguyên mức lạ thay vì bịa tên', () => {
    expect(severityLabel('UNKNOWN')).toBe('UNKNOWN');
  });
});

describe('percent / formatSpan', () => {
  it('làm tròn độ tin cậy về phần trăm', () => {
    expect(percent(0.7734)).toBe('77%');
  });

  it('in khoảng thời gian với hai chữ số thập phân', () => {
    expect(formatSpan(0.4, 1.523)).toBe('0.40s – 1.52s');
  });
});
