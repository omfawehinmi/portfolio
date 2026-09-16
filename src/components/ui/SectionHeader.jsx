export default function SectionHeader({ title, subtitle }) {
  return (
    <>
      <h2 className="section__title">{title}</h2>
      {subtitle ? <span className="section__subtitle">{subtitle}</span> : null}
    </>
  )
}
