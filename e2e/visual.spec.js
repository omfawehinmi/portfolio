// Full-page screenshot baselines at each project viewport.
// Do not commit the PNG files from a local run. Courtland owns a
// workflow_dispatch workflow that renders baselines in
// mcr.microsoft.com/playwright:v1.64.1-noble and commits them to the
// pull-request branch. See README.md.
const { test, expect, openRoute } = require('./fixtures');
const { ROUTES } = require('./routes');

test.describe('Visual baselines', { tag: '@visual' }, () => {
  test.use({ reducedMotion: 'reduce' });
  for (const route of ROUTES) {
    test(`${route} full page`, async ({ page }) => {
      test.setTimeout(90000);
      await openRoute(page, route);
      await expect(page).toHaveScreenshot('portfolio.png', {
        fullPage: true,
        animations: 'disabled',
        caret: 'hide',
      });
    });
  }
});
