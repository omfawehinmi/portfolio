// WebP first with the original as fallback. width/height reserve space (CLS).
const WEBP = new Set(['myorglab', 'semantic-hr', 'sof-overview', 'finance-ledger', 'cortex-self-serve', 'home'])

export default function Picture({ src, alt = '', width, height, eager = false, className, style }) {
  const base = String(src).split('/').pop().replace(/\.(png|jpe?g)$/i, '')
  const webp = WEBP.has(base) ? src.replace(/\.(png|jpe?g)$/i, '.webp') : null
  const img = (
    <img
      src={src}
      alt={alt}
      width={width}
      height={height}
      loading={eager ? 'eager' : 'lazy'}
      fetchPriority={eager ? 'high' : undefined}
      decoding="async"
      className={className}
      style={style}
    />
  )
  if (!webp) return img
  return (
    <picture>
      <source type="image/webp" srcSet={webp} />
      {img}
    </picture>
  )
}
