export default function Tags({ items = [] }) {
  if (!items.length) return null

  return (
    <ul className="tag-list">
      {items.map((item) => (
        <li className="tag" key={item}>
          {item}
        </li>
      ))}
    </ul>
  )
}
