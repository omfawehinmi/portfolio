export const publicUrl = (path) =>
  `${import.meta.env.BASE_URL}${String(path).replace(/^\//, '')}`

export const IMAGE_INTRINSIC_SIZE = {
  'camera.png': { width: 442, height: 584 },
  'cortex-self-serve.png': { width: 2460, height: 835 },
  'home.png': { width: 374, height: 500 },
  'household-gmail.png': { width: 1280, height: 720 },
  'household-grok.gif': { width: 1280, height: 720 },
  'household-grok.png': { width: 1280, height: 720 },
  'household-ledger.png': { width: 1280, height: 720 },
  'household-monarch.png': { width: 1280, height: 720 },
  'household-pbi-headcount.png': { width: 1024, height: 565 },
  'household-pbi-term.png': { width: 1024, height: 571 },
  'household-semantics.gif': { width: 1280, height: 720 },
  'household-semantics.png': { width: 1280, height: 720 },
  'household-stage.png': { width: 1280, height: 720 },
  'household-tools.gif': { width: 1280, height: 720 },
  'household-tools.png': { width: 1280, height: 720 },
  'inpainting-ensemble.png': { width: 865, height: 685 },
  'inpainting.png': { width: 795, height: 400 },
  'michael_fawehinmi.png': { width: 522, height: 390 },
  'myorglab.png': { width: 3325, height: 1816 },
  'palms-punch.png': { width: 3325, height: 1816 },
  'semantic-hr.png': { width: 3325, height: 1816 },
  'sof-finance.png': { width: 1280, height: 720 },
  'sof-overview.png': { width: 2158, height: 1171 },
  'sof-portal.png': { width: 2326, height: 1437 },
  'sof-tax.png': { width: 2159, height: 1103 },
  'sof.png': { width: 3325, height: 1816 },
}

export function intrinsicImageSize(src) {
  const filename = String(src || '').split('/').pop() || ''
  return IMAGE_INTRINSIC_SIZE[filename] || null
}
