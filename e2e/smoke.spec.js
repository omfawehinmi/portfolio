import { test, expect } from '@playwright/test'

test('home page renders without console errors', async ({ page }) => {
  const errors = []
  page.on('pageerror', (e) => errors.push(e.message))
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })
  await page.goto('./')
  await expect(page).toHaveTitle(/Michael Fawehinmi/)
  await expect(page.locator('#root')).not.toBeEmpty()
  expect(errors).toEqual([])
})
