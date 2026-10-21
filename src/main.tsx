import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

if (import.meta.env.DEV) {
  void import('./state/studio').then((m) => {
    ;(window as unknown as { __studio: unknown }).__studio = m.useStudio
  })
  void import('./enhance/store').then((m) => {
    ;(window as unknown as { __enhance: unknown }).__enhance = m.useEnhance
  })
}
