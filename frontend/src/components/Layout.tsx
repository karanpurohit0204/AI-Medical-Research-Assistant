import { Activity, Stethoscope } from 'lucide-react'
import { NavLink, Outlet } from 'react-router-dom'
export function Layout() { return <><header><NavLink to="/" className="brand"><span><Stethoscope size={21}/></span> Evidence</NavLink><nav aria-label="Main navigation"><NavLink to="/" end>Ask</NavLink><NavLink to="/papers">Explore papers</NavLink></nav><div className="secure"><Activity size={15}/> Research mode</div></header><main><Outlet /></main><footer>For research support only — not a substitute for clinical judgment.</footer></> }
