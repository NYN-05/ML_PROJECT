import { Outlet } from "react-router-dom";

import BackgroundShell from "../components/BackgroundShell.jsx";
import Sidebar from "../components/Sidebar.jsx";

const AppShell = () => (
  <BackgroundShell>
    <div className="min-h-screen px-6 py-8">
      <div className="mx-auto flex w-full max-w-6xl gap-6">
        <Sidebar />
        <main className="flex-1 space-y-8">
          <Outlet />
        </main>
      </div>
    </div>
  </BackgroundShell>
);

export default AppShell;
