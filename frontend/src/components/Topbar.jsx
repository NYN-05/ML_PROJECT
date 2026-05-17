import { UserCircle } from "lucide-react";
import { useAppContext } from "../context/AppContext.jsx";

const Topbar = () => {
  const { user } = useAppContext();

  return (
    <div className="flex items-center justify-between">
      <div>
        <p className="text-xs uppercase tracking-[0.3em] text-textSecondary">System</p>
        <h2 className="text-2xl font-orbitron text-textPrimary">AI Control Hub</h2>
      </div>
      <div className="flex items-center gap-3 rounded-full bg-white/5 px-4 py-2">
        <UserCircle size={20} className="text-neon" />
        <span className="text-sm text-textSecondary">{user?.name}</span>
      </div>
    </div>
  );
};

export default Topbar;
