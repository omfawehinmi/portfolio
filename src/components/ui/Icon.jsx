export default function Icon({ name, className = '' }) {
  return <i className={`uil ${name} ${className}`.trim()} aria-hidden="true" />
}
