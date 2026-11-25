// axe checks on the portfolio page. Fail on serious or critical issues.
const { test, expect, openRoute } = require('./fixtures');
const { ROUTES } = require('./routes');
const { AxeBuilder } = require('@axe-core/playwright');

function formatViolations(violations) {
  return violations
    .map((v) => {
      const nodes = v.nodes
        .slice(0, 5)
        .map((n) => `    - ${n.target.join(' ')}`)
        .join('\n');
      return `${v.id} [${v.impact}] ${v.help} (${v.nodes.length} nodes)\n${nodes}`;
    })
    .join('\n');
}

test.describe('Accessibility', { tag: '@a11y' }, () => {
  for (const route of ROUTES) {
    test(`${route} has no serious or critical axe issues`, async ({ page }) => {
      test.setTimeout(90000);
      await openRoute(page, route);
      const results = await new AxeBuilder({ page }).analyze();
      const serious = results.violations.filter(
        (v) => v.impact === 'serious' || v.impact === 'critical',
      );
      expect(serious, formatViolations(serious)).toEqual([]);
    });
  }
});
