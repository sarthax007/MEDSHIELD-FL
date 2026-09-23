import React from "react";
import { useLocation, Link } from "react-router-dom";
import {
  Microscope,
  ArrowLeft,
  Brain,
  Layers,
  Activity,
  AlertTriangle,
  Fingerprint,
} from "lucide-react";

export default function Explainability() {
  const location = useLocation();
  const [imageUrl, setImageUrl] = React.useState<string | null>(() => {
    if (location.state?.imageUrl) {
      sessionStorage.setItem("prediction_previewUrl", location.state.imageUrl);
      return location.state.imageUrl;
    }
    return sessionStorage.getItem("prediction_previewUrl");
  });

  return (
    <div className="max-w-6xl mx-auto space-y-6 pb-20">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-3">
            <Link
              to="/predict"
              className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-400 hover:text-white"
            >
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <Microscope className="w-8 h-8 text-rose-400" />
              AI Explainability (Grad-CAM)
            </h1>
          </div>
          <p className="text-slate-400 mt-2 text-sm ml-14">
            Visualizing Neural Network Attention and Feature Saliency Maps.
          </p>
        </div>
      </div>

      {!imageUrl ? (
        <div className="mt-12 flex flex-col items-center justify-center py-20 px-4 text-center rounded-3xl backdrop-blur-2xl bg-slate-900/60 border-2 border-dashed border-slate-700">
          <Brain className="w-16 h-16 text-slate-600 mb-4" />
          <h2 className="text-xl font-bold text-slate-300 mb-2">
            No Scan Data Found
          </h2>
          <p className="text-slate-500 max-w-md mb-8">
            Please run an AI inference on an MRI scan first to view its
            explainability report.
          </p>
          <Link
            to="/predict"
            className="px-6 py-3 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl transition-all shadow-[0_0_20px_rgba(6,182,212,0.4)]"
          >
            Run New Scan
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mt-8">
          {/* Left Column: Visualizations */}
          <div className="lg:col-span-2 space-y-6">
            {/* Primary Grad-CAM View */}
            <div className="backdrop-blur-2xl bg-slate-900/60 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <Layers className="w-5 h-5 text-rose-400" />
                  Primary Saliency Map
                </h3>
                <span className="text-xs font-mono font-bold text-slate-500 uppercase tracking-widest bg-slate-950 px-3 py-1 rounded-full border border-slate-800">
                  Layer: block4_conv3
                </span>
              </div>

              <div className="relative w-full aspect-square md:aspect-video rounded-xl overflow-hidden border border-slate-700 bg-black">
                {/* Base Image */}
                <img
                  src={imageUrl}
                  alt="Original Scan"
                  className="absolute inset-0 w-full h-full object-contain filter grayscale contrast-125"
                />

                {/* CSS Grad-CAM Overlay */}
                <div
                  className="absolute inset-0 mix-blend-color-burn opacity-90 pointer-events-none"
                  style={{
                    background:
                      "radial-gradient(circle at 45% 40%, rgba(225,29,72,0.85) 0%, rgba(245,158,11,0.6) 15%, rgba(16,185,129,0.2) 30%, transparent 50%)",
                  }}
                />
                <div
                  className="absolute inset-0 mix-blend-screen opacity-75 pointer-events-none"
                  style={{
                    background:
                      "radial-gradient(circle at 45% 40%, rgba(255,255,255,0.9) 0%, rgba(253,224,71,0.8) 5%, transparent 20%)",
                  }}
                />

                {/* Scanner Grid Overlay */}
                <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.15)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.15)_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />

                {/* Bounding Box Mock */}
                <div className="absolute top-[32%] left-[38%] w-[15%] h-[15%] border-2 border-rose-500 rounded z-10 pointer-events-none shadow-[0_0_15px_rgba(225,29,72,0.5)]">
                  <div className="absolute -top-6 left-0 text-[10px] font-mono font-bold text-rose-400 bg-slate-950 px-2 py-0.5 rounded border border-rose-500/30">
                    High Attention
                  </div>
                </div>
              </div>

              {/* Heatmap Legend */}
              <div className="mt-4 flex items-center justify-between text-xs font-mono font-bold text-slate-400">
                <span>Low Importance</span>
                <div className="flex-1 mx-4 h-2 rounded-full bg-gradient-to-r from-transparent via-emerald-500 and-amber-500 to-rose-600" />
                <span>High Importance</span>
              </div>
            </div>

            {/* Secondary Comparison */}
            <div className="grid grid-cols-2 gap-6">
              <div className="backdrop-blur-2xl bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                <h4 className="text-sm font-bold text-slate-300 mb-3 text-center">
                  Original Scan
                </h4>
                <div className="aspect-square bg-black rounded-lg border border-slate-700 overflow-hidden relative">
                  <img
                    src={imageUrl}
                    alt="Original"
                    className="w-full h-full object-cover filter grayscale contrast-125"
                  />
                </div>
              </div>

              <div className="backdrop-blur-2xl bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                <h4 className="text-sm font-bold text-slate-300 mb-3 text-center">
                  Guided Backpropagation
                </h4>
                <div className="aspect-square bg-black rounded-lg border border-slate-700 overflow-hidden relative">
                  <img
                    src={imageUrl}
                    alt="Edge"
                    className="w-full h-full object-cover filter invert contrast-150 grayscale"
                  />
                  <div className="absolute inset-0 bg-cyan-900/40 mix-blend-color" />
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: AI Metrics */}
          <div className="space-y-6">
            {/* Diagnosis Card */}
            <div className="backdrop-blur-2xl bg-slate-900/60 border border-rose-500/30 rounded-2xl p-6 shadow-[inset_0_0_40px_rgba(225,29,72,0.05)]">
              <h3 className="text-slate-400 text-xs font-bold uppercase tracking-wider mb-2">
                Model Conclusion
              </h3>
              <div className="flex items-center gap-2 mb-4">
                <AlertTriangle className="w-6 h-6 text-rose-500" />
                <span className="text-2xl font-bold text-white">Malignant</span>
              </div>

              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-xs font-medium uppercase tracking-wider mb-1">
                    <span className="text-slate-400">Confidence</span>
                    <span className="text-rose-400 font-mono font-bold">
                      94.2%
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-1.5">
                    <div className="bg-gradient-to-r from-rose-600 to-rose-400 h-1.5 rounded-full w-[94.2%] shadow-[0_0_10px_rgba(225,29,72,0.6)]" />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs font-medium uppercase tracking-wider mb-1">
                    <span className="text-slate-400">
                      Epistemic Uncertainty
                    </span>
                    <span className="text-emerald-400 font-mono font-bold">
                      4.1%
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-1.5">
                    <div className="bg-emerald-500 h-1.5 rounded-full w-[4.1%]" />
                  </div>
                </div>
              </div>
            </div>

            {/* Feature Importance */}
            <div className="backdrop-blur-2xl bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <h3 className="text-sm font-bold text-white flex items-center gap-2 mb-4">
                <Activity className="w-4 h-4 text-cyan-400" />
                Feature Importance
              </h3>

              <div className="space-y-4">
                {[
                  {
                    label: "Tissue Density Asymmetry",
                    val: 87,
                    color: "bg-rose-500",
                  },
                  { label: "Irregular Border", val: 72, color: "bg-amber-500" },
                  {
                    label: "Microcalcifications",
                    val: 45,
                    color: "bg-cyan-500",
                  },
                  {
                    label: "Vascular Integration",
                    val: 28,
                    color: "bg-slate-500",
                  },
                ].map((feature, i) => (
                  <div key={i}>
                    <div className="flex justify-between text-xs font-bold text-slate-300 mb-1">
                      <span>{feature.label}</span>
                      <span className="font-mono text-slate-400">
                        {feature.val}%
                      </span>
                    </div>
                    <div className="w-full bg-slate-800 rounded-full h-1">
                      <div
                        className={`h-1 rounded-full ${feature.color}`}
                        style={{ width: `${feature.val}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Model Provenance */}
            <div className="backdrop-blur-2xl bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
              <h3 className="text-sm font-bold text-white flex items-center gap-2 mb-4">
                <Fingerprint className="w-4 h-4 text-emerald-400" />
                Inference Provenance
              </h3>

              <div className="space-y-3 text-xs font-mono">
                <div className="flex justify-between border-b border-slate-800 pb-2">
                  <span className="text-slate-500">Model Hash</span>
                  <span className="text-emerald-400">0x7F...3B92</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2">
                  <span className="text-slate-500">Global Round</span>
                  <span className="text-slate-300">12</span>
                </div>
                <div className="flex justify-between border-b border-slate-800 pb-2">
                  <span className="text-slate-500">Node Validation</span>
                  <span className="text-cyan-400">Verified</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">FHE Scheme</span>
                  <span className="text-slate-300">TenSEAL CKKS</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
