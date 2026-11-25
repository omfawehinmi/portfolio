# End-to-end checks

Playwright 1.64.1 lives in this folder. Do not bump it. There is no consent
spec in this repo. The Playwright gate is opt-in.

## Serve the build

The Vite base path is `/portfolio/`. Build first, then serve the app on
port 3000. `serve@14.2.6 -s` lists the `/portfolio/` directory instead of
returning `index.html`, so the helper uses `vite preview`, which honors
the base path:

```
npm run build
cd e2e
node serve-portfolio.js
```

If `E2E_BASE_URL` is unset, `playwright.config.js` starts that server for
you (it reuses an already-running server when one is present). The app is
then at `http://127.0.0.1:3000/portfolio/`.

## Commands

From this `e2e/` folder, after `npm ci` and `npx playwright install chromium`:

The four Chromium viewport projects are `chromium-375`, `chromium-768`,
`chromium-1280`, and `chromium-1440`.

- Functional gate (nav + click-through + tile geometry) at those viewports:

  `npm run test:e2e`

  Equivalent: `PLAYWRIGHT_GATE=1 npx playwright test --project=chromium-375 --project=chromium-768 --project=chromium-1280 --project=chromium-1440 nav.spec.js clickthrough.spec.js geometry.spec.js`

- Tile media geometry (same-row height, centering, aspect, overflow):

  `npm run test:e2e:geometry`

  Equivalent: `PLAYWRIGHT_GATE=1 npx playwright test --project=chromium-375 --project=chromium-768 --project=chromium-1280 --project=chromium-1440 geometry.spec.js`

- Accessibility (axe, serious and critical fail the test):

  `npm run test:e2e:a11y`

  Equivalent: `PLAYWRIGHT_GATE=1 npx playwright test --project=chromium-375 --project=chromium-768 --project=chromium-1280 --project=chromium-1440 a11y.spec.js`

  Tagged `@a11y`. A later Test gate job can also pass the file:
  `npx playwright test a11y.spec.js`

- Visual baselines:

  `npm run test:e2e:visual`

  Equivalent: `PLAYWRIGHT_GATE=1 npx playwright test --project=chromium-375 --project=chromium-768 --project=chromium-1280 --project=chromium-1440 visual.spec.js`

  Tagged `@visual`.

`PLAYWRIGHT_GATE=1` is required for every gate command. Without it, the
config loads only a placeholder project that matches no files, so a later
CI job that runs `npx playwright test` will not start failing on missing
visual baselines. This folder does not add a required CI job.

## Visual baselines (Courtland)

Do not commit baseline PNGs from a laptop or from this box. Baselines must
be generated on the CI runner OS.

Placeholder for Courtland: add a `workflow_dispatch` workflow that runs in
`mcr.microsoft.com/playwright:v1.64.1-noble`, renders the `@visual` /
`visual-*` project screenshots, and commits the PNGs to the pull-request
branch. That workflow is owned by Courtland and is not part of this change.

If you run `test:e2e:visual` locally only to confirm the specs start, delete
any generated PNG files (under `e2e/*snapshots*` and `e2e/test-results`)
before you commit.
