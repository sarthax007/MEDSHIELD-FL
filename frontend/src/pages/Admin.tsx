import React, { useState, useEffect, useRef } from "react";
import {
  ShieldCheck,
  Activity,
  Lock,
  Database,
  Play,
  CheckCircle2,
  Loader2,
  Network,
} from "lucide-react";

interface NodeData {
  id: string;
  name: string;
  location: string;
  uptime: string;
  contributions: number;
  status: "active" | "inactive";
}

const mockNodes: NodeData[] = [
  {
    id: "NODE-1A",
    name: "VIT Medical Center",
    location: "Pune, India",
    uptime: "99.9%",
    contributions: 12450,
    status: "active",
  },
  {
    id: "NODE-2B",
    name: "Pune City Hospital",
    location: "Pune, India",
    uptime: "99.8%",
    contributions: 8200,
    status: "active",
  },
  {
    id: "NODE-3C",
    name: "Nanded General",
    location: "Nanded, India",
    uptime: "99.5%",
    contributions: 5400,
    status: "active",
  },
  {
    id: "NODE-4D",
    name: "Mumbai Care Institute",
    location: "Mumbai, India",
    uptime: "99.9%",
    contributions: 15600,
    status: "active",
  },
];

const mockLogs = [
  "[SYSTEM] Initializing Federated Command Center...",
  "[SECURE] Establishing TLS 1.3 connections to all nodes...",
  "[NODE-1A] Connected successfully. Key exchange verified.",
  "[NODE-2B] Connected successfully. Key exchange verified.",
  "[NODE-3C] Connected successfully. Key exchange verified.",
  "[NODE-4D] Connected successfully. Key exchange verified.",
  "[ENCRYPT] TenSEAL CKKS context loaded and synchronized.",
  "[IDLE] Awaiting manual aggregation trigger...",
];

export default function Admin() {
  const [logs, setLogs] = useState<string[]>([]);
  const [isTriggering, setIsTriggering] = useState(false);
  const [triggerSuccess, setTriggerSuccess] = useState(false);
  const [livePackets, setLivePackets] = useState(2450123);
  const terminalRef = useRef<HTMLDivElement>(null);

  // Live Packet Counter Simulation
  useEffect(() => {
    const interval = setInterval(() => {
      setLivePackets((prev) => prev + Math.floor(Math.random() * 15) + 1);
    }, 150);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    let currentIndex = 0;
    const interval = setInterval(() => {
      if (currentIndex < mockLogs.length) {
        setLogs((prev) => [...prev, mockLogs[currentIndex]]);
        currentIndex++;
      } else {
        clearInterval(interval);
      }
    }, 800);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [logs]);

  const handleTriggerRound = async () => {
    if (isTriggering || triggerSuccess) return;

    setIsTriggering(true);
    setLogs((prev) => [
      ...prev,
      "[MANUAL] Federated Learning round triggered by Administrator.",
    ]);

    // Simulate loading states with logs
    setTimeout(
      () =>
        setLogs((prev) => [
          ...prev,
          "[SYNC] Broadcasting model weights to nodes...",
        ]),
      1000,
    );
    setTimeout(
      () =>
        setLogs((prev) => [
          ...prev,
          "[COMPUTE] Nodes computing local gradients with Homomorphic Encryption...",
        ]),
      2500,
    );
    setTimeout(() => {
      setLogs((prev) => [
        ...prev,
        "[SECURE] Received encrypted gradients. Aggregating...",
      ]);
    }, 4000);

    // Resolve API call
    setTimeout(() => {
      setIsTriggering(false);
      setTriggerSuccess(true);
      setLogs((prev) => [
        ...prev,
        "[SUCCESS] Global model updated and secured. Idle.",
      ]);

      // Reset success state after a few seconds
      setTimeout(() => setTriggerSuccess(false), 3000);
    }, 5500);
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <ShieldCheck className="w-8 h-8 text-cyan-400" />
            System Administration
          </h1>
          <p className="text-slate-400 mt-2 text-sm">
            Secure node management and terminal logs.
          </p>
        </div>
      </div>

      {/* Security & Health Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="backdrop-blur-2xl bg-slate-900/60 border border-emerald-500/20 shadow-[inset_0_0_20px_rgba(16,185,129,0.05)] rounded-2xl p-6 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 transform translate-x-4 -translate-y-4 group-hover:scale-110 transition-transform duration-500">
            <Activity className="w-24 h-24 text-emerald-500" />
          </div>
          <div className="relative z-10">
            <div className="flex items-center justify-between mb-4">
              <p className="text-slate-400 text-sm font-medium uppercase tracking-wider">
                Network Status
              </p>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-xs text-emerald-400 font-bold tracking-widest">
                  ONLINE
                </span>
              </div>
            </div>
            <h2 className="text-4xl font-bold text-white tracking-tight">
              Healthy
            </h2>
            <p className="text-emerald-400/80 text-xs mt-2 font-mono">
              4/4 Nodes Active
            </p>
          </div>
        </div>

        <div className="backdrop-blur-2xl bg-slate-900/60 border border-cyan-500/20 shadow-[inset_0_0_20px_rgba(6,182,212,0.05)] rounded-2xl p-6 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 transform translate-x-4 -translate-y-4 group-hover:scale-110 transition-transform duration-500">
            <Lock className="w-24 h-24 text-cyan-500" />
          </div>
          <div className="relative z-10">
            <div className="flex items-center justify-between mb-4">
              <p className="text-slate-400 text-sm font-medium uppercase tracking-wider">
                Encryption Scheme
              </p>
            </div>
            <h2 className="text-4xl font-bold text-white tracking-tight">
              CKKS
            </h2>
            <p className="text-cyan-400/80 text-xs mt-2 font-mono">
              TenSEAL Homomorphic Encryption
            </p>
          </div>
        </div>

        <div className="backdrop-blur-2xl bg-slate-900/60 border border-purple-500/20 shadow-[inset_0_0_20px_rgba(168,85,247,0.05)] rounded-2xl p-6 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 transform translate-x-4 -translate-y-4 group-hover:scale-110 transition-transform duration-500">
            <Database className="w-24 h-24 text-purple-500" />
          </div>
          <div className="relative z-10">
            <div className="flex items-center justify-between mb-4">
              <p className="text-slate-400 text-sm font-medium uppercase tracking-wider">
                Encrypted Packets
              </p>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-purple-400 animate-ping" />
              </div>
            </div>
            <h2 className="text-4xl font-bold text-white tracking-tight">
              {livePackets.toLocaleString()}
            </h2>
            <p className="text-purple-400/80 text-xs mt-2 font-mono flex items-center gap-1">
              <Activity className="w-3 h-3" /> Live traffic stream
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Node Security Table */}
        <div className="lg:col-span-2 backdrop-blur-2xl bg-slate-900/60 border border-cyan-500/20 shadow-[inset_0_0_20px_rgba(6,182,212,0.1)] rounded-2xl p-6 flex flex-col relative overflow-hidden">
          <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
            <Network className="w-5 h-5 text-cyan-400" />
            Active Node Security Protocol
          </h3>
          <div className="overflow-x-auto flex-1">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 text-xs font-semibold uppercase tracking-wider">
                  <th className="py-3 px-4">Node ID</th>
                  <th className="py-3 px-4">Location</th>
                  <th className="py-3 px-4 text-center">Uptime</th>
                  <th className="py-3 px-4 text-right">Contributions</th>
                  <th className="py-3 px-4 text-center">Security Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {mockNodes.map((node) => (
                  <tr
                    key={node.id}
                    className="hover:bg-slate-800/20 transition-colors"
                  >
                    <td className="py-4 px-4 font-mono text-cyan-400 text-sm">
                      {node.id}
                    </td>
                    <td className="py-4 px-4">
                      <p className="text-white text-sm font-medium">
                        {node.name}
                      </p>
                      <p className="text-slate-500 text-xs">{node.location}</p>
                    </td>
                    <td className="py-4 px-4 text-center text-slate-300 text-sm font-mono">
                      {node.uptime}
                    </td>
                    <td className="py-4 px-4 text-right text-slate-300 text-sm font-mono">
                      {node.contributions.toLocaleString()}
                    </td>
                    <td className="py-4 px-4 text-center">
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                        Encrypted
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Live System Terminal & Controls */}
        <div className="flex flex-col gap-6">
          <div className="backdrop-blur-2xl bg-black/80 border border-slate-700/50 shadow-lg rounded-2xl p-4 flex flex-col relative overflow-hidden h-[400px]">
            <div className="flex items-center gap-2 mb-3 pb-3 border-b border-slate-800/50">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <div className="w-3 h-3 rounded-full bg-yellow-500" />
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span className="ml-2 text-xs text-slate-500 font-mono">
                root@medshield-fl:~
              </span>
            </div>
            <div
              ref={terminalRef}
              className="flex-1 overflow-y-auto font-mono text-xs text-green-400 space-y-1.5 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent pr-2"
            >
              {logs.map((log, index) => {
                if (!log) return null;
                return (
                  <div
                    key={index}
                    className={`${
                      log.includes("[SUCCESS]")
                        ? "text-cyan-400"
                        : log.includes("[SECURE]")
                          ? "text-emerald-400"
                          : log.includes("[MANUAL]")
                            ? "text-yellow-400"
                            : "text-slate-400"
                    }`}
                  >
                    <span className="text-slate-600 mr-2">{">"}</span>
                    {log}
                  </div>
                );
              })}
              {isTriggering && (
                <div className="text-cyan-400 animate-pulse">
                  <span className="text-slate-600 mr-2">{">"}</span>_
                </div>
              )}
            </div>
          </div>

          <button
            onClick={handleTriggerRound}
            disabled={isTriggering || triggerSuccess}
            className={`
              relative w-full overflow-hidden rounded-xl font-bold text-sm tracking-wide py-4 px-6 flex items-center justify-center gap-3 transition-all duration-300
              ${
                isTriggering
                  ? "bg-slate-800 text-slate-300 border border-slate-700 cursor-not-allowed"
                  : triggerSuccess
                    ? "bg-emerald-900/50 text-emerald-400 border border-emerald-500 shadow-[0_0_20px_rgba(16,185,129,0.3)]"
                    : "bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-[0_0_20px_rgba(6,182,212,0.4)] hover:shadow-[0_0_30px_rgba(6,182,212,0.6)] border border-cyan-400/50"
              }
            `}
          >
            {isTriggering ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Initiating Secure Aggregation...
              </>
            ) : triggerSuccess ? (
              <>
                <CheckCircle2 className="w-5 h-5" />
                Model Sync Complete
              </>
            ) : (
              <>
                <Play className="w-5 h-5 fill-current" />
                Trigger Global FL Round
              </>
            )}

            {/* Shimmer effect for default state */}
            {!isTriggering && !triggerSuccess && (
              <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/20 to-transparent hover:animate-[shimmer_2s_infinite]" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
