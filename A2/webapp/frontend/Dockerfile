FROM node:24-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:stable-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
# Optional Basic Auth, configured from the environment each time the container starts.
COPY --chmod=755 basic-auth.sh /docker-entrypoint.d/40-basic-auth.sh
RUN mkdir -p /etc/nginx/auth && touch /etc/nginx/auth/basic.conf
COPY --from=build /app/dist /usr/share/nginx/html
