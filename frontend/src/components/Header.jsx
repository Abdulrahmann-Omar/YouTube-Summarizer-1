import { FiSun, FiMoon } from 'react-icons/fi'
import '../styles/Header.css'

function Header({ darkMode, toggleDarkMode }) {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <h1>📺 YouTube Summarizer</h1>
          <p className="subtitle">AI-Powered Video Summaries</p>
        </div>
        
        <button
          className="theme-toggle"
          onClick={toggleDarkMode}
          aria-label="Toggle dark mode"
        >
          {darkMode ? <FiSun size={20} /> : <FiMoon size={20} />}
        </button>
      </div>
    </header>
  )
}

export default Header

