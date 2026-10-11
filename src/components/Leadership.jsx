import SectionHeader from './ui/SectionHeader.jsx'

export default function Leadership() {
  return (
    <section className="qualification section" id="leadership">
      <SectionHeader title="Leadership" />

      <div className="qualification__container container">
        <div className="role-list">
          <article className="role">
            <div className="role__head">
              <h3 className="role__title">People management</h3>
            </div>
            <ul className="role__bullets">
              <li>Manages 3 direct reports.</li>
              <li>5 years of people management.</li>
            </ul>
          </article>
        </div>
      </div>
    </section>
  )
}
