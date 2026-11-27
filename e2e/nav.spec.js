// Header nav clicks at desktop (768, 1280, 1440) and the mobile toggle at 375.
// The CSS hide for the toggle starts at 768px.
const { test, expect, isMobileNav, header, openRoute, assertUsableDestination } = require('./fixtures');
const { SECTION_IDS } = require('./routes');

test.describe('Header navigation', () => {
  test('desktop section links', async ({ page }) => {
    test.skip(isMobileNav(), 'desktop nav links are in the row from 768px');
    await openRoute(page, '/portfolio/');
    for (const id of SECTION_IDS) {
      const link = header(page).locator(`a[href="#${id}"]`).first();
      await expect(link).toBeVisible();
      await link.click();
      await expect(page).toHaveURL(new RegExp(`#${id}$`));
      await expect(page.locator('#' + id)).toHaveCount(1);
      await assertUsableDestination(page);
    }
  });

  test('mobile toggle, Esc, close, and links', async ({ page }) => {
    test.skip(!isMobileNav(), 'mobile toggle is visible below 768px');
    await openRoute(page, '/portfolio/');

    const openBtn = page.getByRole('button', { name: 'Toggle navigation menu' });
    await expect(openBtn).toBeVisible();
    await openBtn.click();
    const closeBtn = page.getByRole('button', { name: 'Close navigation menu' });
    await expect(closeBtn).toBeVisible();
    await expect(page.locator('#nav-menu a[href="#projects"]')).toBeVisible();

    await page.keyboard.press('Escape');
    await expect(page.locator('#nav-menu')).not.toHaveClass(/show-menu/);

    await openBtn.click();
    await expect(closeBtn).toBeVisible();
    await closeBtn.click();
    await expect(page.locator('#nav-menu')).not.toHaveClass(/show-menu/);

    await openBtn.click();
    await page.locator('#nav-menu a[href="#projects"]').click();
    await expect(page).toHaveURL(/#projects$/);
    await expect(page.locator('#nav-menu')).not.toHaveClass(/show-menu/);
    await assertUsableDestination(page);
  });
});
