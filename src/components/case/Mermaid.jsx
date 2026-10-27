import { useEffect, useId, useRef, useState } from 'react'

// Resolve a site CSS token (which may nest var()) to a concrete computed color.
function token(name) {
  const probe = document.createElement('span')
  probe.style.color = `var(${name})`
  document.body.appendChild(probe)
  const value = getComputedStyle(probe).color
  probe.remove()
  return value
}

function themeFromSite() {
  const accent = token('--first-color-alt')
  const surface = token('--container-color')
  const body = token('--body-color')
  const title = token('--title-color')
  const text = token('--text-color')
  const soft = token('--input-color')
  return {
    fontFamily: 'Poppins, sans-serif',
    fontSize: '14px',
    background: body,
    primaryColor: surface,
    primaryTextColor: title,
    primaryBorderColor: accent,
    secondaryColor: soft,
    tertiaryColor: soft,
    lineColor: text,
    textColor: text,
    clusterBkg: soft,
    clusterBorder: accent,
    edgeLabelBackground: surface,
    nodeTextColor: title,
  }
}

const useNarrow = () => {
  const query = '(max-width: 600px)'
  const [narrow, setNarrow] = useState(() => window.matchMedia(query).matches)
  useEffect(() => {
    const mql = window.matchMedia(query)
    const onChange = () => setNarrow(mql.matches)
    mql.addEventListener('change', onChange)
    return () => mql.removeEventListener('change', onChange)
  }, [])
  return narrow
}

// chart: (dir) => mermaid source. Diagrams run TB at all widths.
export default function Mermaid({ chart, wide = 'TB', label }) {
  const ref = useRef(null)
  const id = `mmd${useId().replace(/[^a-zA-Z0-9]/g, '')}`
  const narrow = useNarrow()
  const [dark, setDark] = useState(() => document.body.classList.contains('dark-theme'))

  useEffect(() => {
    const observer = new MutationObserver(() => setDark(document.body.classList.contains('dark-theme')))
    observer.observe(document.body, { attributes: true, attributeFilter: ['class'] })
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    let cancelled = false
    import('mermaid').then(async ({ default: mermaid }) => {
      mermaid.initialize({ startOnLoad: false, theme: 'base', themeVariables: themeFromSite(), securityLevel: 'strict' })
      const { svg } = await mermaid.render(`${id}${narrow ? 'n' : 'w'}${dark ? 'd' : 'l'}`, chart(narrow ? 'TB' : wide))
      if (cancelled || !ref.current) return
      ref.current.innerHTML = svg
      // Render at natural size (text stays >= 12px); the card scrolls horizontally when needed.
      const el = ref.current.querySelector('svg')
      const vb = el?.viewBox?.baseVal
      if (el && vb?.width) {
        el.style.maxWidth = 'none'
        el.style.width = `${vb.width}px`
        el.style.height = `${vb.height}px`
      }
    })
    return () => {
      cancelled = true
    }
  }, [chart, wide, narrow, dark, id])

  return <figure className="case-diagram" role="img" aria-label={label} ref={ref} />
}
