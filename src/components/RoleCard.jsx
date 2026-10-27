export default function RoleCard({ title, org, years, bullets = [] }) {
  return (
    <article className="rb-card rb-role">
      <h3 className="rb-role__title">{title}</h3>
      <p className="rb-role__org">{org}</p>
      {years ? <p className="rb-role__years">{years}</p> : null}
      {bullets.length ? (
        <ul className="rb-role__bullets">
          {bullets.map((bullet) => (
            <li key={bullet}>{bullet}</li>
          ))}
        </ul>
      ) : null}
    </article>
  )
}
