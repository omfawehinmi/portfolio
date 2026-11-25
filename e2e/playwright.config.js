// Playwright gate for the portfolio. There is no consent spec here.
// Default `npx playwright test` without PLAYWRIGHT_GATE or a visual/a11y
// file argument loads no viewport projects, so a later CI job that runs
// `npx playwright test` will not start failing on missing visual baselines.
// The four Chromium viewport projects are chromium-375, chromium-768,
// chromium-1280, and chromium-1440.
// Serve the Vite production build on port 3000 at /portfolio/ via
// serve-portfolio.js (vite preview). serve@14.2.6 -s lists /portfolio/
// as a directory instead of returning index.html.
const { defineConfig, devices } = require('@playwright/test');

const VIEWPORTS = [
  { name: '375', width: 375, height: 812 },
  { name: '768', width: 768, height: 1024 },
  { name: '1280', width: 1280, height: 800 },
  { name: '1440', width: 1440, height: 900 },
];

function chromiumUse(viewport) {
  return {
    ...devices['Desktop Chrome'],
    viewport,
  };
}

const viewportProjects = VIEWPORTS.map((vp) => ({
  name: `chromium-${vp.name}`,
  use: chromiumUse({ width: vp.width, height: vp.height }),
}));

const idleProject = {
  name: 'chromium',
  use: { ...devices['Desktop Chrome'] },
  testMatch: /consent\.spec\.js/,
};

function wantsViewportProjects() {
  if (process.env.PLAYWRIGHT_GATE === '1') return true;
  const job = process.env.GITHUB_JOB || '';
  if (job === 'test-gate' || job === 'update-visual-baselines') return true;
  const args = process.argv.join(' ');
  if (/(visual|a11y|accessibility|geometry)[^ ]*\.spec\.js/.test(args)) return true;
  if (/--project(?:=|\s+)chromium-\d+/.test(args)) return true;
  return false;
}

module.exports = defineConfig({
  testDir: '.',
  timeout: 60000,
  retries: 0,
  workers: 2,
  reporter: process.env.CI ? 'list' : 'line',
  use: {
    baseURL: process.env.E2E_BASE_URL || 'http://localhost:3000',
    trace: 'retain-on-failure',
  },
  webServer: process.env.E2E_BASE_URL
    ? undefined
    : {
        command: 'node serve-portfolio.js',
        url: 'http://localhost:3000/portfolio/',
        reuseExistingServer: !process.env.CI,
        timeout: 60000,
        cwd: __dirname,
      },
  projects: wantsViewportProjects() ? viewportProjects : [idleProject],
});
