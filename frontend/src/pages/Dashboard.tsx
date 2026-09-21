import { useEffect, useState } from "react";
import { apiClient } from "../lib/api";
import { Shield, Activity, Network, Server, Zap, Database } from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

interface CurrentRound {
  round_number: number;
  status: string;
}

interface AccuracyTrend {
  round_number: number;
  global_accuracy: number;
}

interface HospitalParticipation {
  hospital_name: string;
  contribution_count: number;
}

export default function Dashboard() {
  const [currentRound, setCurrentRound] = useState<CurrentRound | null>(null);
  const [accuracyTrend, setAccuracyTrend] = useState<AccuracyTrend[]>([]);
  const [hospitalData, setHospitalData] = useState<HospitalParticipation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [roundRes, accuracyRes, hospitalRes] = await Promise.all([
          apiClient.get("/metrics/current-round"),
          apiClient.get("/metrics/accuracy-trend"),
          apiClient.get("/metrics/hospital-participation"),
        ]);

        setCurrentRound(roundRes.data);
        setAccuracyTrend(accuracyRes.data);
        setHospitalData(hospitalRes.data);
        setError(null);
      } catch (err: any) {
        setError(
          "Failed to securely fetch federated metrics from the network.",
        );
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-[70vh]">
        <div className="relative">
          <div className="w-16 h-16 border-4 border-cyan-500/20 rounded-full animate-spin border-t-cyan-500" />
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
            <Activity className="w-6 h-6 text-cyan-400 animate-pulse" />
          </div>
        </div>
        <p className="mt-6 text-slate-400 font-medium tracking-widest text-sm uppercase">
          Decrypting Network State
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-[70vh]">
        <div className="backdrop-blur-xl bg-red-950/30 border border-red-900/50 p-8 rounded-2xl text-center max-w-md">
          <Shield className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-xl font-bold text-red-100 mb-2">
            Network Desync
          </h3>
          <p className="text-red-400/80">{error}</p>
        </div>
      </div>
    );
  }

  const latestAccuracy =
    accuracyTrend.length > 0
      ? (accuracyTrend[accuracyTrend.length - 1].global_accuracy * 100).toFixed(
          2,
        )
      : "0.00";

  const totalContributions = hospitalData.reduce(
    (acc, curr) => acc + curr.contribution_count,
    0,
  );

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header Section */}
      <div className="flex justify-between items-end mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight flex items-center gap-3">
            <Activity className="w-8 h-8 text-cyan-400" />
            Federated Command Center
          </h1>
          <p className="text-slate-400 mt-2 text-sm">
            Real-time homomorphic encryption metrics and active node telemetry.
          </p>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-emerald-400 text-sm font-semibold tracking-wide">
            Secure Sync Active
          </span>
        </div>
      </div>

      {/* Top Stat Cards (Bento Grid) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Round Card */}
        <div className="group relative overflow-hidden backdrop-blur-2xl bg-slate-900/60 border border-cyan-500/20 shadow-[inset_0_0_15px_rgba(6,182,212,0.1)] p-6 rounded-2xl transition-all duration-300 hover:border-cyan-500/40 hover:bg-slate-800/80 hover:shadow-[inset_0_0_25px_rgba(6,182,212,0.2)]">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Server className="w-24 h-24 text-cyan-500 transform rotate-12" />
          </div>
          <div className="relative z-10">
            <p className="text-slate-400 text-sm font-medium uppercase tracking-wider mb-1">
              Current FL Round
            </p>
            <div className="flex items-baseline gap-2">
              <h2 className="text-5xl font-bold text-white tracking-tight">
                {currentRound?.round_number || 0}
              </h2>
            </div>
            <p className="text-cyan-400 text-xs mt-2 font-mono flex items-center gap-1">
              <Zap className="w-3 h-3" /> {currentRound?.status || "Idle"}
            </p>
          </div>
        </div>

        {/* Accuracy Card */}
        <div className="group relative overflow-hidden backdrop-blur-2xl bg-slate-900/60 border border-emerald-500/20 shadow-[inset_0_0_15px_rgba(16,185,129,0.1)] p-6 rounded-2xl transition-all duration-300 hover:border-emerald-500/40 hover:bg-slate-800/80 hover:shadow-[inset_0_0_25px_rgba(16,185,129,0.2)]">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Activity className="w-24 h-24 text-emerald-500 transform -rotate-12" />
          </div>
          <div className="relative z-10">
            <p className="text-slate-400 text-sm font-medium uppercase tracking-wider mb-1">
              Global ViT Accuracy
            </p>
            <div className="flex items-baseline gap-2">
              <h2 className="text-5xl font-bold text-white tracking-tight">
                {latestAccuracy}%
              </h2>
            </div>
            <p className="text-emerald-400 text-xs mt-2 font-mono flex items-center gap-1">
              <Shield className="w-3 h-3" /> Aggregated securely
            </p>
          </div>
        </div>

        {/* Contributions Card */}
        <div className="group relative overflow-hidden backdrop-blur-2xl bg-slate-900/60 border border-purple-500/20 shadow-[inset_0_0_15px_rgba(168,85,247,0.1)] p-6 rounded-2xl transition-all duration-300 hover:border-purple-500/40 hover:bg-slate-800/80 hover:shadow-[inset_0_0_25px_rgba(168,85,247,0.2)]">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Database className="w-24 h-24 text-purple-500 transform rotate-12" />
          </div>
          <div className="relative z-10">
            <p className="text-slate-400 text-sm font-medium uppercase tracking-wider mb-1">
              Total Datapoints
            </p>
            <div className="flex items-baseline gap-2">
              <h2 className="text-5xl font-bold text-white tracking-tight">
                {totalContributions.toLocaleString()}
              </h2>
            </div>
            <p className="text-purple-400 text-xs mt-2 font-mono flex items-center gap-1">
              <Network className="w-3 h-3" /> Distributed globally
            </p>
          </div>
        </div>
      </div>

      {/* Main Charts & Visualizer Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recharts Model Convergence */}
        <div className="lg:col-span-2 backdrop-blur-2xl bg-slate-900/60 border border-cyan-500/20 shadow-[inset_0_0_20px_rgba(6,182,212,0.1)] rounded-2xl p-6 relative overflow-hidden group">
          <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2 relative z-20">
            <Activity className="w-5 h-5 text-cyan-400" />
            Model Convergence Trajectory
          </h3>
          <div className="flex-1 min-h-[300px] w-full relative z-20">
            {accuracyTrend.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={accuracyTrend}
                  margin={{ top: 10, right: 10, left: -20, bottom: 20 }}
                >
                  <defs>
                    <linearGradient
                      id="colorGlobal"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#334155"
                    vertical={false}
                    opacity={0.2}
                  />
                  <XAxis
                    dataKey="round_number"
                    stroke="#64748b"
                    tick={{ fill: "#64748b", fontSize: 12 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    stroke="#64748b"
                    tick={{ fill: "#64748b", fontSize: 12 }}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(val) => `${(val * 100).toFixed(0)}`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      border: "1px solid #1e293b",
                      borderRadius: "12px",
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="global_accuracy"
                    stroke="#06b6d4"
                    strokeWidth={3}
                    fillOpacity={1}
                    fill="url(#colorGlobal)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm">
                Awaiting first federated round completion...
              </div>
            )}
          </div>
        </div>

        {/* CSS 3D Orbital Node Rings */}
        <div className="backdrop-blur-2xl bg-slate-900/60 border border-emerald-500/20 shadow-[inset_0_0_20px_rgba(16,185,129,0.1)] rounded-2xl p-6 flex flex-col">
          <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
            <Network className="w-5 h-5 text-emerald-400" />
            Active Hospital Nodes
          </h3>

          <div className="flex-1 flex flex-col justify-center relative min-h-[300px] preserve-3d perspective-[1000px]">
            {/* 3D Central Aggregator Core */}
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-20 animate-float-3d">
              <div className="relative">
                {/* Complex Interweaving Rings */}
                <div className="absolute inset-[-10px] border border-cyan-500/30 rounded-full animate-[spin_6s_linear_infinite] [transform:rotateX(70deg)]"></div>
                <div className="absolute inset-[-20px] border border-emerald-500/20 rounded-full animate-[spin_8s_linear_reverse_infinite] [transform:rotateY(70deg)]"></div>

                <div className="w-16 h-16 bg-slate-900 rounded-full border-2 border-cyan-500 flex items-center justify-center animate-pulse-glow shadow-[0_0_30px_rgba(6,182,212,0.6)]">
                  <Server className="w-8 h-8 text-cyan-400 drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]" />
                </div>
                <div className="absolute inset-0 border-2 border-cyan-400 rounded-full animate-[ping_3s_cubic-bezier(0,0,0.2,1)_infinite] opacity-40"></div>
              </div>
            </div>

            {/* 3D Orbital Container for Nodes */}
            <div className="absolute inset-0 z-10 animate-[spin_30s_linear_infinite] preserve-3d [transform:rotateX(60deg)]">
              {hospitalData.map((hospital, index) => {
                const total = hospitalData.length;
                const angle = (index / total) * 2 * Math.PI - Math.PI / 2;
                const radius = 130;
                const x = Math.cos(angle) * radius;
                const y = Math.sin(angle) * radius;

                return (
                  <div
                    key={hospital.hospital_name}
                    className="absolute top-1/2 left-1/2 z-10 transition-all duration-1000 preserve-3d"
                    style={{
                      marginLeft: `${x}px`,
                      marginTop: `${y}px`,
                      transform: `translate(-50%, -50%) rotate(${
                        (index / total) * 360 + 180
                      }deg)`,
                    }}
                  >
                    <div className="relative group cursor-pointer animate-[spin_30s_linear_reverse_infinite] preserve-3d">
                      {/* Connection Line */}
                      <div
                        className="absolute top-1/2 left-1/2 h-[2px] bg-gradient-to-l from-emerald-500 to-transparent origin-left z-0 opacity-50"
                        style={{ width: `${radius}px` }}
                      />

                      {/* 3D Micro-encapsulated sphere */}
                      <div
                        className="relative z-10 w-6 h-6 bg-slate-900 border border-emerald-500 rounded-full shadow-[0_0_20px_rgba(16,185,129,0.8)] [transform:rotateX(-60deg)] animate-float-3d flex items-center justify-center"
                        style={{ animationDelay: `${index * 0.5}s` }}
                      >
                        <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                      </div>

                      {/* Lock-and-key ring system */}
                      <div
                        className="absolute inset-[-4px] border-t-2 border-r-2 border-emerald-400 rounded-full animate-spin opacity-80 [transform:rotateX(-60deg)]"
                        style={{ animationDuration: `${2 + index * 0.2}s` }}
                      />

                      {/* Floating Tooltip facing camera */}
                      <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity bg-slate-900/90 backdrop-blur-md border border-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.3)] px-4 py-2 rounded-lg text-xs text-nowrap pointer-events-none z-30 [transform:rotateX(-60deg)] translate-z-[50px]">
                        <span className="text-white font-bold">
                          {hospital.hospital_name}
                        </span>
                        <span className="text-emerald-400 ml-2">
                          ({hospital.contribution_count} img)
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {hospitalData.length === 0 && (
              <div className="text-center text-slate-500 text-sm mt-32">
                Waiting for hospital nodes to connect...
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
