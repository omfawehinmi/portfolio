import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/portfolio/',
  server: {
    host: '0.0.0.0',
    port: 5173,
    strictPort: true,
    // trycloudflare quick tunnels need the public hostname allowed.
    allowedHosts: ['.trycloudflare.com', 'localhost', '127.0.0.1'],
  },
})
