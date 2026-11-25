// Click every header nav link, in-page anchor, menu control, and in-page
// interactive control. There is no backend form; contact is mailto / copy.
const { test, expect, isMobileNav, header, openRoute, assertUsableDestination } = require('./fixtures');
const { ROUTES, SECTION_IDS } = require('./routes');

async function openMobileMenu(page) {
  const openBtn = page.getByRole('button', { name: 'Toggle navigation menu' });
  await openBtn.click();
  await expect(page.getByRole('button', { name: 'Close navigation menu' })).toBeVisible();
}

test.describe('Click-through', () => {
  test('header links and section anchors reach their targets', async ({ page }) => {
    test.setTimeout(120000);
    await openRoute(page, '/portfolio/');

    for (const id of SECTION_IDS) {
      await openRoute(page, '/portfolio/');
      if (isMobileNav()) {
        await openMobileMenu(page);
        await page.locator(`#nav-menu a[href="#${id}"]`).click();
      } else {
        await header(page).locator(`a[href="#${id}"]`).first().click();
      }
      await expect(page).toHaveURL(new RegExp(`#${id}$`));
      const target = page.locator('#' + id);
      await expect(target).toHaveCount(1);
      await expect.poll(async () => page.evaluate((targetId) => {
        const el = document.getElementById(targetId);
        const head = document.querySelector('header');
        if (!el || !head) return false;
        const focus = el.querySelector('h1, h2, h3, .section__title') || el;
        const fr = focus.getBoundingClientRect();
        const hr = head.getBoundingClientRect();
        return fr.bottom <= hr.top + 4 || fr.top >= hr.bottom - 4;
      }, id), { timeout: 2500 }).toBe(true);
      await assertUsableDestination(page);
    }
  });

  test('the page loads and in-page anchors in main are not covered by the header', async ({ page }) => {
    await openRoute(page, '/portfolio/');
    await assertUsableDestination(page);
    const ids = await page.locator('main a[href^="#"]').evaluateAll((els) => {
      const seen = [];
      for (const el of els) {
        const href = el.getAttribute('href') || '';
        const id = href.startsWith('#') ? href.slice(1) : '';
        if (id && !seen.includes(id)) seen.push(id);
      }
      return seen;
    });
    const covered = [];
    for (const id of ids) {
      if (!/^[A-Za-z][\w-]*$/.test(id)) continue;
      const link = page.locator(`main a[href="#${id}"]`).first();
      if (!(await link.count())) continue;
      await link.click();
      await expect(page.locator('#' + id)).toHaveCount(1);
      try {
        await expect.poll(async () => page.evaluate((targetId) => {
          const el = document.getElementById(targetId);
          const head = document.querySelector('header');
          if (!el || !head) return false;
          const focus = el.querySelector('h1, h2, h3, .section__title') || el;
          const fr = focus.getBoundingClientRect();
          const hr = head.getBoundingClientRect();
          return fr.bottom <= hr.top + 4 || fr.top >= hr.bottom - 4;
        }, id), { timeout: 2500 }).toBe(true);
      } catch (err) {
        covered.push('#' + id);
      }
    }
    expect(covered, `sticky header covers: ${covered.join(', ')}`).toEqual([]);
  });

  test('project dialog opens, dismisses with Escape, Close, and the backdrop, and returns focus', async ({ page }) => {
    await openRoute(page, '/portfolio/');
    if (isMobileNav()) {
      await openMobileMenu(page);
      await page.locator('#nav-menu a[href="#projects"]').click();
    } else {
      await header(page).locator('a[href="#projects"]').first().click();
    }
    const tile = page.locator('.project-tile, [data-testid="project-tile"]').first();
    await expect(tile).toBeVisible();
    await tile.click();
    const dialog = page.getByRole('dialog');
    await expect(dialog).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(dialog).toHaveCount(0);
    await expect(tile).toBeVisible();

    await tile.click();
    await expect(dialog).toBeVisible();
    await dialog.getByRole('button', { name: 'Close' }).click();
    await expect(dialog).toHaveCount(0);

    await tile.click();
    await expect(dialog).toBeVisible();
    // Main uses a full-page .services__modal overlay. The redesign uses a
    // native <dialog>, whose dimmed ::backdrop is outside the panel. A
    // click at the viewport corner hits the overlay on main and the
    // backdrop on the redesign.
    const overlay = page.locator('.services__modal.active-modal, dialog.project-dialog[open]');
    await expect(overlay).toBeVisible();
    await page.mouse.click(2, 2);
    await expect(dialog).toHaveCount(0);
  });

  test('skills panels open and close', async ({ page }) => {
    await openRoute(page, '/portfolio/#skills');
    const trigger = page.locator('.skills__header').first();
    await expect(trigger).toBeVisible();
    await trigger.click();
    await expect(trigger).toHaveAttribute('aria-expanded', 'true');
    await trigger.click();
    await expect(trigger).toHaveAttribute('aria-expanded', 'false');
  });

  test('contact copy control works without posting to a backend', async ({ page }) => {
    await page.context().grantPermissions(['clipboard-write', 'clipboard-read']).catch(() => {});
    await openRoute(page, '/portfolio/#contact');
    // exact: true, so project tiles whose accessible name includes the
    // word "copy" (the redesign) do not match. Main and the redesign
    // both label this control "Copy".
    const copy = page.getByRole('button', { name: 'Copy', exact: true });
    if (await copy.count()) {
      await copy.click();
      await expect(page.getByRole('button', { name: 'Copied', exact: true })).toBeVisible();
    }
    await assertUsableDestination(page);
    // No form posts to intercept on this page.
  });
});
