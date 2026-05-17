import { Link } from "react-router-dom";

import BackgroundShell from "../components/BackgroundShell.jsx";
import NeonButton from "../components/NeonButton.jsx";

const NotFound = () => (
  <BackgroundShell>
    <div className="min-h-screen flex items-center justify-center px-6 py-16">
      <div className="space-y-4 text-center">
        <p className="text-xs uppercase tracking-[0.3em] text-neon">404</p>
        <h2 className="text-3xl font-orbitron text-textPrimary">Signal Lost</h2>
        <p className="text-textSecondary">
          The route you requested is not reachable in this system.
        </p>
        <Link to="/">
          <NeonButton className="px-6 py-3">Return Home</NeonButton>
        </Link>
      </div>
    </div>
  </BackgroundShell>
);

export default NotFound;
