import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/portfolio/',
  // Local/tunnel previews only; GitHub Pages ignores this.
  preview: { headers: { 'X-Robots-Tag': 'noindex, nofollow' }, allowedHosts: ['.trycloudflare.com'] },
})
