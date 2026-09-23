import { useCallback, useMemo } from "react";
import { Outlet, Link, useLocation } from "react-router-dom";
import { useAuth } from "../lib/AuthContext";
import {
  Activity,
  LayoutDashboard,
  ShieldCheck,
  Stethoscope,
  Microscope,
  LogOut,
  ScanLine,
  Hexagon,
} from "lucide-react";
import Particles from "react-tsparticles";
import { loadSlim } from "tsparticles-slim";
import type { Engine } from "tsparticles-engine";

export default function AppShell() {
  const location = useLocation();
  const { logout } = useAuth();

  const particlesInit = useCallback(async (engine: Engine) => {
    await loadSlim(engine);
  }, []);

  const particlesOptions = useMemo(
    () => ({
      fullScreen: { enable: true, zIndex: -10 },
      background: { color: { value: "#020617" } },
      fpsLimit: 60,
      particles: {
        color: { value: ["#06b6d4", "#10b981", "#6366f1"] },
        links: {
          color: "random",
          distance: 150,
          enable: true,
          opacity: 0.4,
          width: 1.5,
          triangles: { enable: true, opacity: 0.05 },
        },
        move: {
          enable: true,
          speed: 1.2,
          direction: "none" as const,
          outModes: "out" as const,
        },
        number: { density: { enable: true, width: 800 }, value: 60 },
        opacity: {
          value: 0.7,
          animation: { enable: true, speed: 0.5, minimumValue: 0.3 },
        },
        shape: { type: "circle" },
        size: { value: { min: 1, max: 3 } },
      },
      interactivity: {
        events: { onHover: { enable: true, mode: "connect" } },
        modes: {
          connect: { distance: 120, radius: 150, links: { opacity: 0.5 } },
        },
      },
      detectRetina: true,
    }),
    [],
  );

  const navigation = [
    { name: "Dashboard", href: "/", icon: LayoutDashboard },
    { name: "Admin Panel", href: "/admin", icon: ShieldCheck },
    { name: "Doctor Queue", href: "/doctor", icon: Stethoscope },
    { name: "New Scan", href: "/predict", icon: ScanLine },
    { name: "Explainability", href: "/explain", icon: Microscope },
    { name: "3D Volumetric", href: "/viewer3d", icon: Hexagon },
  ];

  return (
    <div className="flex h-screen bg-transparent text-slate-50 font-sans selection:bg-cyan-500/30 relative z-0">
      {/* Optimized Ultra-Realistic Neural Particle Web */}
      <Particles
        id="tsparticles"
        init={particlesInit}
        options={particlesOptions}
      />

      {/* Deep Vignette Overlay for Depth */}
      <div className="absolute inset-0 z-[-5] pointer-events-none bg-[radial-gradient(circle_at_center,transparent_0%,#020617_100%)]" />

      {/* Sidebar */}
      <aside className="w-64 bg-slate-900/50 backdrop-blur-xl border-r border-slate-800 flex flex-col z-20">
        <div className="h-16 flex items-center px-6 border-b border-slate-800">
          <Activity className="h-6 w-6 text-cyan-400 mr-2" />
          <span className="text-lg font-bold text-white tracking-tight">
            MedShield-FL
          </span>
        </div>

        <nav className="flex-1 px-4 py-6 space-y-1 overflow-y-auto">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`flex items-center px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 shadow-[0_0_15px_rgba(6,182,212,0.1)]"
                    : "text-slate-400 border border-transparent hover:bg-slate-800 hover:text-slate-200 hover:border-slate-700"
                }`}
              >
                <item.icon
                  className={`flex-shrink-0 h-5 w-5 mr-3 ${
                    isActive ? "text-cyan-400" : "text-slate-500"
                  }`}
                />
                {item.name}
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t border-slate-800 space-y-4">
          <div className="flex flex-col items-center justify-center p-3 bg-cyan-900/20 border border-cyan-500/30 rounded-xl shadow-[0_0_15px_rgba(6,182,212,0.15)] relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-emerald-500/10 opacity-50" />
            <ShieldCheck className="w-6 h-6 text-cyan-400 mb-2 group-hover:scale-110 transition-transform duration-300 drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
            <span className="text-[10px] font-bold text-slate-300 uppercase tracking-widest text-center relative z-10">
              Privacy Status
            </span>
            <span className="text-xs font-mono font-bold text-emerald-400 mt-1 relative z-10 drop-shadow-[0_0_5px_rgba(16,185,129,0.8)]">
              100% Data Stayed Local
            </span>
          </div>

          <button
            onClick={logout}
            className="flex w-full items-center px-3 py-2.5 text-sm font-medium text-slate-400 border border-transparent hover:bg-red-500/10 hover:text-red-400 hover:border-red-500/20 rounded-lg transition-all duration-200"
          >
            <LogOut className="flex-shrink-0 h-5 w-5 mr-3 text-slate-500 group-hover:text-red-400" />
            Sign out
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        {/* Ambient background glow */}
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-[128px] pointer-events-none z-0" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-emerald-500/10 rounded-full blur-[128px] pointer-events-none z-0" />

        <header className="h-16 bg-slate-900/50 backdrop-blur-xl border-b border-slate-800 flex items-center justify-between px-8 z-10">
          <h1 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            {navigation.find((item) => item.href === location.pathname)?.name ||
              "MedShield-FL"}
          </h1>
          <div className="flex items-center space-x-4">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.2)]">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-2 animate-pulse"></span>
              System Online
            </span>
          </div>
        </header>

        <div className="flex-1 overflow-auto p-8 relative z-10">
          <div className="mx-auto max-w-7xl">
            <Outlet />
          </div>
        </div>
      </main>
    </div>
  );
}
