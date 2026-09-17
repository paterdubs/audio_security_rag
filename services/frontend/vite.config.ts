// defineConfig lấy từ `vitest/config` chứ không phải `vite`: bản của vite không biết
// khoá `test`, nên tsc sẽ báo lỗi kiểu ngay ở file cấu hình.
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// Dev server proxy /api sang service `api`. Lý do dùng proxy thay vì gọi thẳng
// localhost:8000: cùng một đường dẫn tương đối `/api/v1/...` chạy được ở CẢ dev lẫn
// production (nginx cũng proxy y hệt), nên không có biến môi trường base-URL nào phải
// nhớ đặt, và cũng không có chuyện dev chạy mà bản dựng thì gọi sai địa chỉ.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET ?? 'http://localhost:8000',
        changeOrigin: true,
        ws: true, // /api/v1/alerts/stream là WebSocket
      },
    },
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
