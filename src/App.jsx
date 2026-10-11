import { useState } from 'react'
import Header from './components/Header.jsx'
import Home from './components/Home.jsx'
import Qualification from './components/Qualification.jsx'
import Projects from './components/Projects.jsx'
import Leadership from './components/Leadership.jsx'
import Skills from './components/Skills.jsx'
import Contact from './components/Contact.jsx'
import Footer from './components/Footer.jsx'
import ScrollUp from './components/ScrollUp.jsx'
import useScrollSpy from './hooks/useScrollSpy.js'
import useTheme from './hooks/useTheme.js'

export default function App() {
  const { dark, toggleTheme } = useTheme()
  const { scrolled, showScroll, activeSection } = useScrollSpy()
  const [openProjectId, setOpenProjectId] = useState(null)

  return (
    <>
      <Header dark={dark} onToggleTheme={toggleTheme} scrolled={scrolled} activeSection={activeSection} />
      <main className="main">
        <Home onOpenProject={setOpenProjectId} />
        <Projects openId={openProjectId} onOpen={setOpenProjectId} />
        <Qualification />
        <Leadership />
        <Skills />
        <Contact />
      </main>
      <Footer />
      <ScrollUp visible={showScroll} />
    </>
  )
}
