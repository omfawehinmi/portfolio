import { highlights } from '../data/projects.js'
import SiteHero, { heroProps } from './SiteHero.jsx'
import McpSpotlight from './McpSpotlight.jsx'
import ImpactBand from './ImpactBand.jsx'

const impactItems = highlights.map((item) => ({
  projectId: item.projectId,
  icon: item.icon,
  title: item.label,
  body: item.detail,
  href: '#projects',
}))

export default function Home({ onOpenProject, onOpen }) {
  return (
    <>
      <SiteHero {...heroProps} />
      <section className="section rb-lead" id="mcp" aria-labelledby="t1-mcp-h">
        <div className="rb-wrap">
          <McpSpotlight onOpen={onOpen || onOpenProject} headingLevel={2} />
        </div>
      </section>
      <ImpactBand items={impactItems} onOpen={onOpenProject} />
    </>
  )
}
