import Icon from './Icon.jsx'
import { intrinsicImageSize } from '../../utils.js'

export default function ProjectMedia({ project }) {
  if (project.videoFile) {
    return <video src={project.videoFile} muted loop playsInline autoPlay preload="metadata" aria-hidden="true" />
  }
  if (project.embed) {
    return <iframe src={project.embed} title="" allow="autoplay; fullscreen" tabIndex={-1} />
  }
  if (project.thumb) {
    const size = intrinsicImageSize(project.thumb)
    return (
      <img
        src={project.thumb}
        alt=""
        width={size?.width}
        height={size?.height}
        style={{ objectPosition: project.thumbPos || 'center top' }}
      />
    )
  }
  return <Icon name={project.icon} />
}
