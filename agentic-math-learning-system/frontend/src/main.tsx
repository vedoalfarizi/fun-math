// src/main.tsx
// ─────────────────────────────────────────────────────────────────────
// Application entry point. Mounts the React app into the #root div
// and imports global Tailwind CSS styles.
// ─────────────────────────────────────────────────────────────────────
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
