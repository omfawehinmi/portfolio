import { highlights } from '../data/projects.js'
import SiteHero, { heroProps } from './SiteHero.jsx'
import ImpactBand from './ImpactBand.jsx'

const impactItems = highlights.map((item) => ({
  projectId: item.projectId,
  icon: item.icon,
  title: item.label,
  body: item.detail,
  href: '#projects',
}))

export default function Home({ onOpenProject }) {
  return (
    <>
      <SiteHero {...heroProps} />
      <ImpactBand items={impactItems} onOpen={onOpenProject} />
    </>
  )
}
