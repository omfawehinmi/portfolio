import { test, expect } from '@playwright/test'

// Fails on JS errors and on broken same-origin assets. Third-party requests
// (embeds, analytics) are ignored: they can 401/block on CI runners.
test('home page renders without JS errors or broken local assets', async ({ page, baseURL }) => {
  const origin = new URL(baseURL).origin
  const errors = []
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`))
  page.on('response', (r) => {
    if (r.url().startsWith(origin) && r.status() >= 400) errors.push(`${r.status()} ${r.url()}`)
  })
  await page.goto('./')
  await expect(page).toHaveTitle(/Michael Fawehinmi/)
  await expect(page.locator('#root')).not.toBeEmpty()
  expect(errors).toEqual([])
})
