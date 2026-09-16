import { useEffect, useState } from 'react'

export default function useScrollSpy(sectionSelector = 'section[id]') {
  const [scrolled, setScrolled] = useState(false)
  const [showScroll, setShowScroll] = useState(false)
  const [activeSection, setActiveSection] = useState('home')

  useEffect(() => {
    const onScroll = () => {
      setScrolled(window.scrollY >= 80)
      setShowScroll(window.scrollY >= 200)

      const sections = document.querySelectorAll(sectionSelector)
      const scrollY = window.pageYOffset

      sections.forEach((current) => {
        const sectionHeight = current.offsetHeight
        const sectionTop = current.offsetTop - 80
        const sectionId = current.getAttribute('id')

        if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
          setActiveSection(sectionId)
        }
      })
    }

    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [sectionSelector])

  return { scrolled, showScroll, activeSection }
}
