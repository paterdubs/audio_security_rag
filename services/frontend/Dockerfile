# Dựng bằng node, phục vụ bằng nginx. Image cuối KHÔNG chứa node_modules (~200 MB) lẫn
# mã nguồn TypeScript — chỉ có file tĩnh đã dựng.
FROM node:22-alpine AS build

WORKDIR /build

# Cài dependency trước, theo lockfile: tầng này chỉ dựng lại khi dependency đổi.
# `npm ci` chứ không phải `npm install` — ci cài ĐÚNG lockfile và fail nếu lockfile lệch
# package.json, nên bản dựng lặp lại được.
COPY services/frontend/package.json services/frontend/package-lock.json ./
RUN npm ci

COPY services/frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine

# Cấu hình mặc định của image lắng nghe cổng 80 và sẽ đá nhau với cấu hình của ta.
RUN rm /etc/nginx/conf.d/default.conf
COPY services/frontend/nginx.conf /etc/nginx/conf.d/dashboard.conf
COPY --from=build /build/dist /usr/share/nginx/html

EXPOSE 3000

# 127.0.0.1 chứ KHÔNG phải `localhost`: trong container localhost phân giải ra ::1 trước,
# mà `listen 3000` của nginx chỉ bind IPv4 — healthcheck báo "connection refused" trong
# khi trang vẫn phục vụ bình thường, và container bị đánh dấu unhealthy oan.
HEALTHCHECK --interval=10s --timeout=5s --retries=6 --start-period=5s \
    CMD wget --spider -q http://127.0.0.1:3000/ || exit 1
