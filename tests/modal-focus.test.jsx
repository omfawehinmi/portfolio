// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import Pipelines from '../src/components/pipelines/Pipelines.jsx'

afterEach(cleanup)

const dialog = () => screen.getByRole('dialog')

describe('pipeline modal focus', () => {
  it('moves focus into the modal on open', async () => {
    const user = userEvent.setup()
    render(<Pipelines />)
    await user.click(screen.getByRole('button', { name: /Pipeline 01/ }))
    expect(dialog().contains(document.activeElement)).toBe(true)
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Close' }))
  })

  it('traps Tab and Shift+Tab inside the modal', async () => {
    const user = userEvent.setup()
    render(<Pipelines />)
    await user.click(screen.getByRole('button', { name: /Pipeline 02/ }))
    const close = screen.getByRole('button', { name: 'Close' })
    await user.tab({ shift: true })
    expect(dialog().contains(document.activeElement)).toBe(true)
    expect(document.activeElement).not.toBe(close)
    await user.tab()
    expect(document.activeElement).toBe(close)
    for (let i = 0; i < 6; i += 1) {
      await user.tab()
      expect(dialog().contains(document.activeElement)).toBe(true)
    }
  })

  it('returns focus to the opening card on close', async () => {
    const user = userEvent.setup()
    render(<Pipelines />)
    const card = screen.getByRole('button', { name: /Pipeline 03/ })
    await user.click(card)
    await user.keyboard('{Escape}')
    expect(document.activeElement).toBe(card)
    await user.click(card)
    await user.click(screen.getByRole('button', { name: 'Close' }))
    expect(document.activeElement).toBe(card)
  })

  it('arrow keys move between tabs', async () => {
    const user = userEvent.setup()
    render(<Pipelines />)
    await user.click(screen.getByRole('button', { name: /Pipeline 04/ }))
    const arch = screen.getByRole('tab', { name: 'Architecture' })
    const flow = screen.getByRole('tab', { name: 'Process flow' })
    arch.focus()
    await user.keyboard('{ArrowRight}')
    expect(document.activeElement).toBe(flow)
    expect(flow.getAttribute('aria-selected')).toBe('true')
    await user.keyboard('{ArrowRight}')
    expect(document.activeElement).toBe(arch)
    await user.keyboard('{ArrowLeft}')
    expect(document.activeElement).toBe(flow)
    expect(arch.getAttribute('tabindex')).toBe('-1')
  })
})
