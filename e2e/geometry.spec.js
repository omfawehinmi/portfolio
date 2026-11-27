// DOM geometry checks for project tile media and open-dialog figures.
// Same-row media must share a height, sit in the center of their frame,
// keep the intended aspect, and stay inside the tile and the viewport.
const { test, expect, openRoute } = require('./fixtures');

const TOL = 2;
const ASPECT_TOL = 0.02;

function formatIssues(issues) {
  return issues
    .map((item) => {
      const bits = [item.type, item.title || '', item.detail || '']
        .filter(Boolean)
        .join(' | ');
      return bits;
    })
    .join('\n');
}

async function collectClosedIssues(page) {
  return page.evaluate(({ TOL, ASPECT_TOL }) => {
    const issues = [];
    const vw = document.documentElement.clientWidth;
    const tiles = [...document.querySelectorAll('.bx-tile, .bx-feat-block')];

    const records = tiles.map((tile, index) => {
      const media = tile.querySelector('.bx-media, .bx-fig');
      const inner = media
        ? media.querySelector('img, svg, .bx-svg')
        : null;
      const titleEl = tile.querySelector('.bx-h1, .bx-h2, .bx-h3, .bx-kicker');
      const title = (titleEl && titleEl.textContent.trim()) || `tile-${index}`;
      const tr = tile.getBoundingClientRect();
      const mr = media ? media.getBoundingClientRect() : null;
      const ir = inner ? inner.getBoundingClientRect() : null;
      return { tile, media, inner, title, tr, mr, ir };
    });

    const used = new Set();
    const withMedia = records.filter((row) => row.mr);
    for (let i = 0; i < withMedia.length; i += 1) {
      if (used.has(i)) continue;
      const group = [withMedia[i]];
      used.add(i);
      for (let j = i + 1; j < withMedia.length; j += 1) {
        if (used.has(j)) continue;
        if (Math.abs(withMedia[j].mr.top - withMedia[i].mr.top) <= TOL) {
          group.push(withMedia[j]);
          used.add(j);
        }
      }
      if (group.length < 2) continue;
      const heights = group.map((row) => row.mr.height);
      const delta = Math.max(...heights) - Math.min(...heights);
      if (delta > TOL) {
        issues.push({
          type: 'unequal-row-media-height',
          title: group.map((row) => row.title).join(', '),
          detail: `delta ${delta.toFixed(2)}px (limit ${TOL}px)`,
        });
      }
    }

    const aspectOk = (actual, expected, title, label) => {
      if (!expected || !actual || actual <= 0) return;
      if (Math.abs(actual - expected) / expected > ASPECT_TOL) {
        issues.push({
          type: 'aspect-ratio',
          title,
          detail: `${label} ${actual.toFixed(4)} vs ${expected.toFixed(4)}`,
        });
      }
    };

    for (const row of records) {
      const { tile, title, tr, mr, ir, inner, media } = row;
      if (tile.scrollWidth > tile.clientWidth + TOL) {
        issues.push({
          type: 'overflow',
          title,
          detail: `tile scrollWidth ${tile.scrollWidth} clientWidth ${tile.clientWidth}`,
        });
      }
      if (mr && (mr.left < -TOL || mr.right > vw + TOL)) {
        issues.push({
          type: 'overflow',
          title,
          detail: `media x ${mr.left.toFixed(1)}-${mr.right.toFixed(1)} vw ${vw}`,
        });
      }
      if (mr && ir) {
        const dx = Math.abs((mr.left + mr.width / 2) - (ir.left + ir.width / 2));
        const dy = Math.abs((mr.top + mr.height / 2) - (ir.top + ir.height / 2));
        if (mr.width - ir.width > TOL * 2 && dx > TOL) {
          issues.push({
            type: 'off-center',
            title,
            detail: `horizontal ${dx.toFixed(2)}px`,
          });
        }
        if (mr.height - ir.height > TOL * 2 && dy > TOL) {
          issues.push({
            type: 'off-center',
            title,
            detail: `vertical ${dy.toFixed(2)}px`,
          });
        }
        if (ir.width > mr.width + TOL || ir.height > mr.height + TOL) {
          issues.push({
            type: 'overflow',
            title,
            detail: `inner ${ir.width.toFixed(1)}x${ir.height.toFixed(1)} in media ${mr.width.toFixed(1)}x${mr.height.toFixed(1)}`,
          });
        }
      }

      if (media && media.classList.contains('bx-feat') === false && tile.classList.contains('bx-feat') && mr) {
        aspectOk(mr.width / mr.height, 16 / 9, title, 'featured media');
      }
      if (tile.className.includes('bx-feat') && media && media.classList.contains('bx-media') && mr) {
        aspectOk(mr.width / mr.height, 16 / 9, title, 'featured media');
      }
      if (tile.className.includes('bx-row') && mr && mr.width > 80 && mr.width < 130) {
        aspectOk(mr.width / mr.height, 104 / 70, title, 'row thumb');
      }
      if (inner && inner.tagName && inner.tagName.toLowerCase() === 'svg' && ir && ir.width > 8 && ir.height > 8) {
        const vb = inner.viewBox && inner.viewBox.baseVal;
        if (vb && vb.width > 0 && vb.height > 0) {
          aspectOk(ir.width / ir.height, vb.width / vb.height, title, 'svg viewBox');
        }
      }
    }

    if (document.documentElement.scrollWidth > vw + TOL) {
      issues.push({
        type: 'overflow',
        title: 'page',
        detail: `scrollWidth ${document.documentElement.scrollWidth} vw ${vw}`,
      });
    }
    return issues;
  }, { TOL, ASPECT_TOL });
}

async function collectDialogIssues(page) {
  return page.evaluate(({ TOL, ASPECT_TOL }) => {
    const issues = [];
    const dialog = document.querySelector('dialog.project-dialog[open], dialog[open], [role="dialog"]');
    if (!dialog) {
      return [{ type: 'dialog-missing', title: 'dialog', detail: 'no open dialog' }];
    }
    const vw = document.documentElement.clientWidth;
    const fig = dialog.querySelector('.bx-fig-big, .bx-fig, .bx-media, .bx-big');
    const inner = fig ? fig.querySelector('img, svg') : null;
    const titleEl = dialog.querySelector('h2, h3, .bx-h1, .bx-h2, [class*="modal"] h3');
    const title = (titleEl && titleEl.textContent.trim()) || 'dialog';
    if (!fig || !inner) return issues;
    const fr = fig.getBoundingClientRect();
    const ir = inner.getBoundingClientRect();
    const dx = Math.abs((fr.left + fr.width / 2) - (ir.left + ir.width / 2));
    if (fr.width - ir.width > TOL * 2 && dx > TOL) {
      issues.push({ type: 'off-center', title, detail: `horizontal ${dx.toFixed(2)}px` });
    }
    // A .bx-fig-big may scroll horizontally on small viewports (min-width
    // on the svg). That is contained overflow. Fail only when the figure
    // itself spills out of the dialog or the viewport, or when a
    // non-scrollable figure's inner media overflows its frame.
    const figScrollsX = fig.scrollWidth > fig.clientWidth + TOL;
    if (fr.left < -TOL || fr.right > vw + TOL) {
      issues.push({
        type: 'overflow',
        title,
        detail: `figure x ${fr.left.toFixed(1)}-${fr.right.toFixed(1)} vw ${vw}`,
      });
    }
    if (!figScrollsX && (ir.width > fr.width + TOL || ir.right > vw + TOL || ir.left < -TOL)) {
      issues.push({
        type: 'overflow',
        title,
        detail: `inner ${ir.width.toFixed(1)} at x ${ir.left.toFixed(1)}-${ir.right.toFixed(1)} fig ${fr.width.toFixed(1)} vw ${vw}`,
      });
    }
    if (dialog.scrollWidth > dialog.clientWidth + TOL) {
      issues.push({
        type: 'overflow',
        title,
        detail: `dialog scrollWidth ${dialog.scrollWidth} clientWidth ${dialog.clientWidth}`,
      });
    }
    // Raster previews should keep their box aspect. Dialog SVGs are sized
    // with width 100% and a max-height, then object-fit: contain, so the
    // element box can differ from the viewBox while the drawing is not stretched.
    if (inner.tagName.toLowerCase() === 'img' && ir.width > 8 && ir.height > 8) {
      const natW = inner.naturalWidth;
      const natH = inner.naturalHeight;
      if (natW && natH) {
        const fit = getComputedStyle(inner).objectFit;
        if (fit === 'contain') {
          const actual = ir.width / ir.height;
          const expected = natW / natH;
          if (Math.abs(actual - expected) / expected > ASPECT_TOL) {
            issues.push({
              type: 'aspect-ratio',
              title,
              detail: `img ${actual.toFixed(4)} vs ${expected.toFixed(4)}`,
            });
          }
        }
      }
    }
    return issues;
  }, { TOL, ASPECT_TOL });
}

test.describe('Tile media geometry', () => {
  test('closed tiles keep even row heights, centering, aspect, and no overflow', async ({ page }) => {
    await openRoute(page, '/portfolio/#projects');
    // The bento tiles exist on the redesign. Main still uses .project-tile,
    // so this check waits until that build is the page under test.
    const tiles = page.locator('.bx-tile');
    test.skip((await tiles.count()) === 0, 'bento project tiles are not in this build');
    await expect(tiles.first()).toBeVisible();
    const issues = await collectClosedIssues(page);
    expect(issues, formatIssues(issues)).toEqual([]);
  });

  test('open dialog figures stay in frame, centered, and in aspect', async ({ page }) => {
    await openRoute(page, '/portfolio/#projects');
    const tiles = page.locator('.bx-tile');
    test.skip((await tiles.count()) === 0, 'bento project tiles are not in this build');
    const selectors = [
      '.bx-lead',
      '[data-testid="mychart-tile"]',
      '.bx-t-mcp',
      '.bx-t-stack',
      'article.bx-row, .bx-row',
    ];
    const all = [];
    for (const sel of selectors) {
      const tile = page.locator(sel).first();
      if (!(await tile.count())) continue;
      await tile.scrollIntoViewIfNeeded();
      await tile.click();
      await expect(page.locator('dialog[open], dialog.project-dialog[open], [role="dialog"]')).toBeVisible();
      const issues = await collectDialogIssues(page);
      for (const item of issues) {
        item.detail = `${sel}: ${item.detail || ''}`.trim();
        all.push(item);
      }
      await page.keyboard.press('Escape');
      await expect(page.locator('dialog[open]')).toHaveCount(0);
    }
    expect(all, formatIssues(all)).toEqual([]);
  });
});
