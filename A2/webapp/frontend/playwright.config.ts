import { defineConfig, devices } from '@playwright/test'

/**
 * End-to-end tests drive the real stack: browser -> Nginx or Vite -> FastAPI -> worker -> PostgreSQL.
 * Default target is the Docker Compose stack; use E2E_BASE_URL=http://localhost:5173 for `npm run dev`.
 */
export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : 4,
  timeout: 180_000,
  expect: { timeout: 10_000 },
  reporter: process.env.CI
    ? [['github'], ['html', { open: 'never' }]]
    : [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: process.env.E2E_BASE_URL ?? 'http://localhost:8080',
    // Basic Auth of a compose.vps.yaml stack (docs/vps.md): E2E_USER=... E2E_PASSWORD=...
    httpCredentials: process.env.E2E_USER
      ? { username: process.env.E2E_USER, password: process.env.E2E_PASSWORD ?? '' }
      : undefined,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
})
