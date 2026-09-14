export default function ScrollUp({ visible }) {
  return (
    <a
      href="#home"
      className={`scrollup${visible ? ' show-scroll' : ''}`}
      id="scroll-top"
      onClick={(event) => {
        event.preventDefault()
        window.scrollTo({ top: 0, behavior: 'smooth' })
      }}
    >
      <i className="uil uil-arrow-up scrollup__icon"></i>
    </a>
  )
}
