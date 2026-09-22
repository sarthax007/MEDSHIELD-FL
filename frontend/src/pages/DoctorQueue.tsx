import { useState } from "react";
import {
  Stethoscope,
  Activity,
  CheckCircle,
  XCircle,
  SkipForward,
  AlertTriangle,
  Brain,
  List,
} from "lucide-react";

interface QueueItem {
  id: string;
  patientId: string;
  confidence: number;
  predictedClass: string;
}

const initialQueueData: QueueItem[] = [
  {
    id: "SCAN-8892",
    patientId: "PT-10024",
    confidence: 48,
    predictedClass: "Anomalous Tissue Detected",
  },
  {
    id: "SCAN-8893",
    patientId: "PT-10025",
    confidence: 52,
    predictedClass: "Benign Cyst",
  },
  {
    id: "SCAN-8894",
    patientId: "PT-10026",
    confidence: 41,
    predictedClass: "Malignant Pattern",
  },
];

export default function DoctorQueue() {
  const [queue, setQueue] = useState<QueueItem[]>(initialQueueData);
  const [activeItem, setActiveItem] = useState<QueueItem | null>(
    initialQueueData[0],
  );
  const [isProcessing, setIsProcessing] = useState(false);

  const handleAction = (_actionName: string) => {
    if (!activeItem) return;
    setIsProcessing(true);

    // Simulate network delay for the AI to register the label
    setTimeout(() => {
      const newQueue = queue.filter((item) => item.id !== activeItem.id);
      setQueue(newQueue);
      setActiveItem(newQueue.length > 0 ? newQueue[0] : null);
      setIsProcessing(false);
    }, 600);
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header Area */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <Stethoscope className="w-8 h-8 text-cyan-400" />
            Doctor Labeling Queue - Active Learning
          </h1>
          <p className="text-slate-400 mt-2 text-sm">
            Manual expert review required for uncertain model predictions.
          </p>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-cyan-500/10 border border-cyan-500/20 shadow-[0_0_15px_rgba(6,182,212,0.2)]">
          <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          <span className="text-cyan-400 text-sm font-semibold tracking-wide">
            {queue.length} Pending Scans
          </span>
        </div>
      </div>

      {queue.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-center space-y-4">
          <CheckCircle className="w-16 h-16 text-emerald-500 animate-bounce" />
          <h2 className="text-2xl font-bold text-white">Queue Empty</h2>
          <p className="text-slate-400">
            All uncertain scans have been labeled by experts. The active
            learning pool is up to date.
          </p>
        </div>
      ) : activeItem ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Medical Image Viewer (Span 2) */}
          <div className="lg:col-span-2 backdrop-blur-2xl bg-slate-900/60 border border-cyan-500/20 shadow-[inset_0_0_20px_rgba(6,182,212,0.1)] rounded-2xl p-6 flex flex-col relative overflow-hidden group">
            <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2 relative z-20">
              <Brain className="w-5 h-5 text-cyan-400" />
              High-Resolution Scan Interface
            </h3>

            <div className="flex-1 w-full relative bg-black rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center min-h-[400px]">
              {/* Background Medical Image (Unsplash) */}
              <img
                src="https://images.unsplash.com/photo-1559757175-5700dde675bc?w=800&q=80"
                alt="Brain MRI Scan"
                className="absolute inset-0 w-full h-full object-cover opacity-80 mix-blend-screen filter grayscale contrast-125"
              />

              {/* CSS Overlays */}
              {/* Radar Grid */}
              <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.1)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none" />
              <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.05)_1px,transparent_1px)] bg-[size:10px_10px] pointer-events-none" />

              {/* Crosshairs */}
              <div className="absolute top-1/2 left-0 w-full h-[1px] bg-cyan-500/30 pointer-events-none" />
              <div className="absolute top-0 left-1/2 w-[1px] h-full bg-cyan-500/30 pointer-events-none" />
              <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 border-2 border-cyan-400/50 rounded-full pointer-events-none" />

              {/* Scanning Line */}
              <div className="absolute top-0 left-0 w-full h-1 bg-cyan-400/80 shadow-[0_0_20px_rgba(6,182,212,1)] animate-scan pointer-events-none z-10" />

              {/* Data Overlays */}
              <div className="absolute top-4 left-4 text-cyan-400 font-mono text-xs z-20">
                <p>ID: {activeItem.id}</p>
                <p>PATIENT: {activeItem.patientId}</p>
                <p>SLICE: 42/128</p>
              </div>
              <div className="absolute bottom-4 right-4 text-emerald-400 font-mono text-xs z-20 text-right">
                <p>ZOOM: 1.0x</p>
                <p>ENHANCEMENT: ON</p>
              </div>

              {/* Simulated Heatmap Box */}
              <div className="absolute top-[40%] left-[45%] w-24 h-24 border border-rose-500/50 bg-rose-500/10 rounded pointer-events-none">
                <div className="absolute -top-6 left-0 text-[10px] font-mono text-rose-400">
                  ROI DETECTED
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-6">
            {/* AI Uncertainty Metrics */}
            <div className="backdrop-blur-2xl bg-slate-900/60 border border-emerald-500/20 shadow-[inset_0_0_20px_rgba(16,185,129,0.1)] rounded-2xl p-6 relative overflow-hidden">
              <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
                <Activity className="w-5 h-5 text-emerald-400" />
                Model Uncertainty Metrics
              </h3>

              <div className="space-y-6">
                <div>
                  <p className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-2">
                    Current Hypothesis
                  </p>
                  <div className="bg-slate-950 border border-slate-800 rounded-lg p-4 flex items-center justify-between">
                    <span className="text-white font-semibold">
                      {activeItem.predictedClass}
                    </span>
                    <AlertTriangle className="w-5 h-5 text-rose-400" />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs font-medium uppercase tracking-wider mb-2">
                    <span className="text-slate-400">Confidence Score</span>
                    <span className="text-rose-400">
                      {activeItem.confidence}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-rose-500 to-orange-400 h-2 rounded-full"
                      style={{ width: `${activeItem.confidence}%` }}
                    />
                  </div>
                  <p className="text-[10px] text-slate-500 mt-2 font-mono text-right">
                    Threshold: 80%
                  </p>
                </div>
              </div>

              {/* Labeling Controls */}
              <div className="mt-8 space-y-3 relative">
                {isProcessing && (
                  <div className="absolute inset-0 z-10 flex items-center justify-center bg-slate-900/80 rounded-xl backdrop-blur-sm">
                    <div className="flex flex-col items-center gap-2">
                      <Activity className="w-6 h-6 text-cyan-400 animate-spin" />
                      <span className="text-xs font-mono text-cyan-400">
                        Updating Model...
                      </span>
                    </div>
                  </div>
                )}
                <p className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-2">
                  Expert Override Action
                </p>

                <button
                  onClick={() => handleAction("confirm")}
                  disabled={isProcessing}
                  className="w-full relative overflow-hidden rounded-xl font-bold text-sm tracking-wide py-3 px-4 flex items-center justify-center gap-3 transition-all duration-300 bg-emerald-900/40 hover:bg-emerald-800/60 text-emerald-400 border border-emerald-500/50 hover:border-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.2)] hover:shadow-[0_0_25px_rgba(16,185,129,0.4)] group"
                >
                  <CheckCircle className="w-5 h-5 group-hover:scale-110 transition-transform" />
                  Confirm Diagnosis
                </button>

                <button
                  onClick={() => handleAction("reject")}
                  disabled={isProcessing}
                  className="w-full relative overflow-hidden rounded-xl font-bold text-sm tracking-wide py-3 px-4 flex items-center justify-center gap-3 transition-all duration-300 bg-rose-900/40 hover:bg-rose-800/60 text-rose-400 border border-rose-500/50 hover:border-rose-400 shadow-[0_0_15px_rgba(225,29,72,0.2)] hover:shadow-[0_0_25px_rgba(225,29,72,0.4)] group"
                >
                  <XCircle className="w-5 h-5 group-hover:scale-110 transition-transform" />
                  Reject & Relabel
                </button>

                <button
                  onClick={() => handleAction("skip")}
                  disabled={isProcessing}
                  className="w-full relative overflow-hidden rounded-xl font-bold text-sm tracking-wide py-3 px-4 flex items-center justify-center gap-3 transition-all duration-300 bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-600 hover:border-slate-500 group"
                >
                  <SkipForward className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  Skip / Unclear
                </button>
              </div>
            </div>

            {/* Upcoming Queue List */}
            <div className="backdrop-blur-2xl bg-slate-900/60 border border-slate-700/50 shadow-lg rounded-2xl p-6 flex flex-col relative overflow-hidden flex-1">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <List className="w-5 h-5 text-slate-400" />
                Active Queue
              </h3>
              <div className="flex-1 overflow-y-auto space-y-2 pr-2 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
                {queue.map((item) => (
                  <div
                    key={item.id}
                    onClick={() => !isProcessing && setActiveItem(item)}
                    className={`p-3 rounded-lg border cursor-pointer transition-all ${
                      activeItem.id === item.id
                        ? "bg-cyan-900/30 border-cyan-500/50 text-cyan-400"
                        : "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-600 hover:bg-slate-800"
                    } ${isProcessing ? "opacity-50 cursor-not-allowed" : ""}`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="font-mono text-xs">{item.id}</span>
                      <span className="text-[10px] bg-black px-2 py-1 rounded font-mono">
                        {item.confidence}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
