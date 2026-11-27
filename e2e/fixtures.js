// Opt-in fixtures for the Playwright gate specs.
const { test: base, expect } = require('@playwright/test');

const THIRD_PARTY = /(unicons\.iconscout|googletagmanager|google-analytics|cloudflareinsights|vimeo|vimeocdn|cloudflarecaptcha|challenges\.cloudflare|dnstest\.(dev|cloudflare)|font-size:0|color:transparent)/i;
const GENERIC_RESOURCE_ERROR = /Failed to load resource/i;
const FIRST_PARTY_ASSET_TYPES = new Set(['script', 'stylesheet', 'font', 'image']);

const test = base.extend({
  page: async ({ page }, use) => {
    const errors = [];
    const seen = new Set();
    const note = (line) => {
      if (seen.has(line)) return;
      seen.add(line);
      errors.push(line);
    };
    page.on('pageerror', (err) => {
      const stack = err.stack || '';
      if (THIRD_PARTY.test(err.message || '') || THIRD_PARTY.test(stack)) return;
      note(`pageerror: ${err.message}`);
    });
    page.on('console', (msg) => {
      if (msg.type() !== 'error') return;
      const text = msg.text();
      const url = (msg.location() && msg.location().url) || '';
      if (THIRD_PARTY.test(text) || THIRD_PARTY.test(url)) return;
      if (GENERIC_RESOURCE_ERROR.test(text)) return;
      note(`console: ${text}`);
    });
    page.on('response', (res) => {
      const status = res.status();
      if (status < 400) return;
      const url = res.url();
      if (THIRD_PARTY.test(url)) return;
      const kind = res.request().resourceType();
      if (!FIRST_PARTY_ASSET_TYPES.has(kind)) return;
      note(`asset ${status} ${kind} ${url}`);
    });
    await use(page);
    expect(errors, `Unexpected console or page errors:\n${errors.join('\n')}`).toEqual([]);
  },
});

function viewportWidth() {
  return (test.info().project.use.viewport && test.info().project.use.viewport.width) || 1280;
}

function isMobileNav() {
  // styles.css hides the toggle from 768px up.
  return viewportWidth() < 768;
}

function header(page) {
  return page.locator('header#header, header.header').first();
}

async function openRoute(page, path) {
  await page.goto(path);
  await expect(page.locator('body')).toBeVisible();
  await expect(header(page)).toBeVisible();
}

async function assertUsableDestination(page) {
  await expect(page.locator('body')).toBeVisible();
  const statusish = await page.evaluate(() => {
    const text = (document.body && document.body.innerText) || '';
    return /Internal Server Error|This site can.?t be reached|Application Error|Page not found/i.test(text);
  });
  expect(statusish, 'destination looked like an error page').toBe(false);
}

module.exports = {
  test,
  expect,
  viewportWidth,
  isMobileNav,
  header,
  openRoute,
  assertUsableDestination,
};
