import { publicUrl } from '../utils.js'

// Tier 1 order (Michael, 6:13 PM): MCP, Cortex self-service, re-architecture, Ledger.
export const featureRows = [
  {
    id: 't1-cortex',
    projectId: 'cortex-self-serve',
    kicker: 'Production',
    flip: true,
    media: {
      src: publicUrl('assets/img/cortex-self-serve.png'),
      alt: 'Cortex self-service screenshot',
      width: 2460,
      height: 835,
      position: 'center',
    },
  },
  {
    id: 't1-migration',
    projectId: 'snowflake-migration',
    kicker: 'Production',
    chipsLabel: 'Case study stats',
    chips: [
      { label: '211 PRs', opens: 'snowflake-migration' },
      { label: '15 decisions', opens: 'snowflake-migration' },
      { label: '5 + 4 PROD + DEV databases', opens: 'snowflake-migration' },
    ],
    // Q13: text-only until a diagram and its alt text exist.
    media: { alt: 'Data flow diagram placeholder', icon: 'uil-server-network' },
  },
  {
    id: 't1-ledger',
    projectId: 'household-ledger',
    kicker: 'Production',
    flip: true,
    media: {
      src: publicUrl('assets/img/household-ledger.png'),
      alt: 'Ledger dashboard with example data',
      width: 1280,
      height: 720,
      position: 'center top',
      badge: 'Example data',
    },
  },
]
