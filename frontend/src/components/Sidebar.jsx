import { NavLink } from "react-router-dom";
import { LayoutDashboard, Activity, History, Settings } from "lucide-react";

const links = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/analysis", label: "Analysis", icon: Activity },
  { to: "/history", label: "History", icon: History },
  { to: "/settings", label: "Settings", icon: Settings },
];

const Sidebar = () => (
  <aside className="hidden lg:flex w-64 flex-col gap-8 glass-panel p-6 rounded-[24px]">
    <div>
      <p className="text-xs uppercase tracking-[0.3em] text-textSecondary">Core</p>
      <h1 className="mt-2 text-xl font-orbitron text-neon">NEON VERIFY</h1>
    </div>
    <nav className="flex flex-col gap-3">
      {links.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          className={({ isActive }) =>
            `flex items-center gap-3 rounded-[14px] px-4 py-3 text-sm transition ${
              isActive
                ? "bg-neon/10 text-neon shadow-glowCyan"
                : "text-textSecondary hover:text-textPrimary hover:bg-white/5"
            }`
          }
        >
          <Icon size={18} />
          {label}
        </NavLink>
      ))}
    </nav>
  </aside>
);

export default Sidebar;
