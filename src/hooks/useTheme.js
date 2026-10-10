import { useEffect, useState } from 'react'

export default function useTheme() {
  const [dark, setDark] = useState(() => localStorage.getItem('selected-theme') === 'dark')

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
