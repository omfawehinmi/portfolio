// Intrinsic sizes of public/assets/img files, for width/height attributes.
export const imageSizes = {
  'cortex-self-serve.png': [2460, 835],
  'ledger-email.png': [1280, 720],
  'ledger-readonly.gif': [1280, 720],
  'finance-ledger.png': [1280, 720],
  'ledger-aggregator.png': [1280, 720],
  'ledger-semantics.gif': [1280, 720],
  'ledger-documents.png': [1280, 720],
  'ledger-tools.gif': [1280, 720],
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
