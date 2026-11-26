import { useEffect, useState } from 'react'

function initialDark() {
  const stored = localStorage.getItem('selected-theme')
  // Dark is the site default; only force light when the visitor chose it.
  if (stored === 'light') return false
  return true
}

export default function useTheme() {
  const [dark, setDark] = useState(initialDark)

  useEffect(() => {
    document.body.classList.toggle('dark-theme', dark)
    localStorage.setItem('selected-theme', dark ? 'dark' : 'light')
    localStorage.setItem('selected-icon', dark ? 'uil-sun' : 'uil-moon')
  }, [dark])

  return {
    dark,
    toggleTheme: () => setDark((value) => !value),
  }
}
