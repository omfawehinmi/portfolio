// Intrinsic sizes of public/assets/img files, for width/height attributes.
export const imageSizes = {
  'cortex-self-serve.png': [2460, 835],
  'household-gmail.png': [1280, 720],
  'household-grok.gif': [1280, 720],
  'household-ledger.png': [1280, 720],
  'household-monarch.png': [1280, 720],
  'household-semantics.gif': [1280, 720],
  'household-stage.png': [1280, 720],
  'household-tools.gif': [1280, 720],
  'inpainting.png': [795, 400],
  'myorglab.png': [3325, 1816],
  'semantic-hr.png': [3325, 1816],
  'sof-finance.png': [1280, 720],
  'sof-overview.png': [2158, 1171],
  'sof-portal.png': [2326, 1437],
  'sof-tax.png': [2159, 1103],
}

export function sizeOf(src) {
  const name = String(src || '').split('/').pop()
  return imageSizes[name] || [1280, 720]
}
