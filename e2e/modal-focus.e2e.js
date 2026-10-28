import { expect, test } from '@playwright/test'

// Real-browser check: jsdom has no CSS transitions, so it cannot catch focus()
// being ignored while the overlay is still visibility: hidden.
async function expectCloseFocused(page) {
  await page.waitForFunction(
    () => document.activeElement?.matches('[role="dialog"] .services__modal-close'),
    null,
    { timeout: 1000 },
  )
  await expect(page.getByRole('dialog').getByRole('button', { name: 'Close' })).toBeFocused()
}

test('Pipelines modal moves focus to the close button on open', async ({ page }) => {
  await page.goto('#pipelines')
  await page.locator('#pipelines').getByRole('button', { name: /Pipeline 01/ }).first().click()
  await expectCloseFocused(page)
  await page.keyboard.press('Escape')
})

test('Projects modal moves focus to the close button on open', async ({ page }) => {
  await page.goto('#projects')
  await page.locator('#projects').getByTestId('project-tile').first().click()
  await expectCloseFocused(page)
})