// The portfolio is a single-page app under the /portfolio/ base path
// (vite.config.js). Section targets come from src/data/profile.js navLinks
// and the section ids on the page.
const ROUTES = ['/portfolio/'];

const SECTION_IDS = ['home', 'projects', 'experience', 'leadership', 'skills', 'contact'];

module.exports = { ROUTES, SECTION_IDS };
