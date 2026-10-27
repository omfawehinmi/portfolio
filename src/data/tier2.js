import { publicUrl } from '../utils.js'

// Q12: SOF moves to Tier 2. Q11: MyOrgLab and Semantic HR stay Tier 2.
export const tier2Heading = 'More production systems'

export const tier2 = [
  { id: 'myorglab', kicker: 'Production', thumb: { src: publicUrl('assets/img/myorglab.png'), width: 3325, height: 1816 } },
  { id: 'semantic-hr', kicker: 'Production', thumb: { src: publicUrl('assets/img/semantic-hr.png'), width: 3325, height: 1816 } },
  { id: 'sof', kicker: 'Financial services', thumb: { src: publicUrl('assets/img/sof-overview.png'), width: 2158, height: 1171, position: 'center top' } },
]

export const labIds = ['camera-tracking', 'palms-bag', 'inpainting']
