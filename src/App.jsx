import { useEffect, useState } from 'react'
import Header from './components/Header.jsx'
import Home from './components/Home.jsx'
import About from './components/About.jsx'
import Skills from './components/Skills.jsx'
import Qualification from './components/Qualification.jsx'
import Projects from './components/Projects.jsx'
import Contact from './components/Contact.jsx'
import Footer from './components/Footer.jsx'
import ScrollUp from './components/ScrollUp.jsx'

export default function App() {
  const [dark, setDark] = useState(() => localStorage.getItem('selected-theme') === 'dark')
  const [scrolled, setScrolled] = useState(false)
  const [showScroll, setShowScroll] = useState(false)
  const [activeSection, setActiveSection] = useState('home')

  useEffect(() => {
    document.body.classList.toggle('dark-theme', dark)
    localStorage.setItem('selected-theme', dark ? 'dark' : 'light')
    localStorage.setItem('selected-icon', dark ? 'uil-sun' : 'uil-moon')
  }, [dark])

  useEffect(() => {
    const onScroll = () => {
      setScrolled(window.scrollY >= 80)
      setShowScroll(window.scrollY >= 200)

      const sections = document.querySelectorAll('section[id]')
      const scrollY = window.pageYOffset

      sections.forEach((current) => {
        const sectionHeight = current.offsetHeight
        const sectionTop = current.offsetTop - 50
        const sectionId = current.getAttribute('id')

        if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
          setActiveSection(sectionId)
        }
      })
    }

    onScroll()
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <>
      <Header
        dark={dark}
        onToggleTheme={() => setDark((value) => !value)}
        scrolled={scrolled}
        activeSection={activeSection}
      />
      <main className="main">
        <Home />
        <About />
        <Skills />
        <Qualification />
        <Projects />
        <Contact />
      </main>
      <Footer />
      <ScrollUp visible={showScroll} />
    </>
  )
}
