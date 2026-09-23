import React, { useState, useRef } from "react";
import { Link } from "react-router-dom";
import {
  UploadCloud,
  Activity,
  Brain,
  ShieldCheck,
  ArrowRight,
  CheckCircle,
  AlertTriangle,
  ScanLine,
  Hexagon,
} from "lucide-react";

export default function Prediction() {
  const [dragActive, setDragActive] = useState(false);
  const [, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(() =>
    sessionStorage.getItem("prediction_previewUrl"),
  );

  const [status, setStatus] = useState<
    "idle" | "uploading" | "analyzing" | "result"
  >(() => (sessionStorage.getItem("prediction_status") as any) || "idle");
  const inputRef = useRef<HTMLInputElement>(null);

  const updateStatus = (
    newStatus: "idle" | "uploading" | "analyzing" | "result",
  ) => {
    setStatus(newStatus);
    sessionStorage.setItem("prediction_status", newStatus);
  };

  const updatePreviewUrl = (url: string | null) => {
    setPreviewUrl(url);
    if (url) {
      sessionStorage.setItem("prediction_previewUrl", url);
    } else {
      sessionStorage.removeItem("prediction_previewUrl");
    }
  };

  // Handle drag events
  const handleDrag = function (e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  // Triggers when file is dropped
  const handleDrop = function (e: React.DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  // Triggers when file is selected via click
  const handleChange = function (e: React.ChangeEvent<HTMLInputElement>) {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (selectedFile: File) => {
    setFile(selectedFile);

    // If it's a standard web image, preview it directly.
    // If it's a medical file (like .nii.gz or .npy), use a placeholder for the UI animation.
    if (selectedFile.type.startsWith("image/")) {
      const objectUrl = URL.createObjectURL(selectedFile);
      updatePreviewUrl(objectUrl);
    } else {
      // Fallback placeholder for .nii.gz, .npy, etc.
      updatePreviewUrl(
        "https://images.unsplash.com/photo-1559757175-5700dde675bc?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
      );
    }

    // Start fake upload/analysis sequence
    startAnalysisSequence();
  };

  const startAnalysisSequence = () => {
    updateStatus("uploading");

    // Simulate secure transfer delay
    setTimeout(() => {
      updateStatus("analyzing");

      // Simulate heavy AI inference delay
      setTimeout(() => {
        updateStatus("result");
      }, 3500);
    }, 1500);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 pb-20">
      {/* Header Area */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <div>
          <h1 className="text-3xl font-bold text-white flex items-center gap-3">
            <ScanLine className="w-8 h-8 text-cyan-400" />
            AI Tumor Prediction
          </h1>
          <p className="text-slate-400 mt-2 text-sm">
            Upload an MRI scan for secure, on-device homomorphic encrypted
            inference.
          </p>
        </div>

        <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.15)]">
          <ShieldCheck className="w-4 h-4" />
          <span className="text-xs font-bold tracking-wider uppercase">
            End-to-End Encrypted
          </span>
        </div>
      </div>

      {status === "idle" && (
        <form
          className="relative mt-8"
          onDragEnter={handleDrag}
          onSubmit={(e) => e.preventDefault()}
        >
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            accept=".nii,.nii.gz,.npy,image/*"
            onChange={handleChange}
          />

          <div
            className={`
              relative flex flex-col items-center justify-center py-20 px-4 text-center cursor-pointer
              transition-all duration-300 rounded-3xl backdrop-blur-2xl bg-slate-900/60
              ${
                dragActive
                  ? "border-2 border-cyan-400 bg-cyan-900/20 shadow-[0_0_40px_rgba(6,182,212,0.3)]"
                  : "border-2 border-dashed border-slate-700 hover:border-cyan-500/50 hover:bg-slate-800/80"
              }
            `}
            onClick={() => inputRef.current?.click()}
          >
            <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.03)_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none rounded-3xl" />

            <div className="w-20 h-20 mb-6 rounded-full bg-slate-800 flex items-center justify-center border border-slate-700 shadow-xl relative z-10 group-hover:scale-110 transition-transform">
              <UploadCloud
                className={`w-10 h-10 ${
                  dragActive ? "text-cyan-400" : "text-slate-400"
                }`}
              />
            </div>

            <h3 className="text-xl font-semibold text-white mb-2 relative z-10">
              Drag & Drop MRI Scan
            </h3>
            <p className="text-slate-400 text-sm max-w-sm relative z-10">
              Supported formats: DICOM, NIfTI, JPEG, PNG. Max size: 50MB. All
              files are encrypted before processing.
            </p>

            <button
              type="button"
              className="mt-8 px-8 py-3 rounded-xl font-bold text-sm bg-cyan-500/10 text-cyan-400 border border-cyan-500/50 hover:bg-cyan-500/20 transition-all relative z-10"
            >
              Browse Files
            </button>
          </div>

          {dragActive && (
            <div
              className="absolute inset-0 w-full h-full z-20"
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            />
          )}
        </form>
      )}

      {(status === "uploading" || status === "analyzing") && previewUrl && (
        <div className="mt-8 relative backdrop-blur-2xl bg-slate-900/60 border border-cyan-500/20 rounded-2xl p-12 flex flex-col items-center justify-center shadow-[inset_0_0_30px_rgba(6,182,212,0.05)] overflow-hidden">
          {/* Animated Tech Background */}
          <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.05)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none" />

          <div className="relative w-64 h-64 rounded-xl overflow-hidden border border-slate-700 shadow-2xl z-10">
            <img
              src={previewUrl}
              alt="Scan Preview"
              className="w-full h-full object-cover filter grayscale contrast-125 brightness-75"
            />

            {/* Scanning Line Animation */}
            {status === "analyzing" && (
              <div className="absolute top-0 left-0 w-full h-1 bg-cyan-400 shadow-[0_0_20px_rgba(6,182,212,1)] animate-scan z-20" />
            )}

            {/* Grid overlay */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.2)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.2)_1px,transparent_1px)] bg-[size:10px_10px] pointer-events-none mix-blend-overlay" />
          </div>

          <div className="mt-8 w-full max-w-md space-y-4 z-10">
            <div className="flex items-center justify-between text-sm font-mono font-semibold">
              <span
                className={
                  status === "uploading"
                    ? "text-cyan-400 animate-pulse"
                    : "text-slate-400"
                }
              >
                {status === "uploading"
                  ? "Encrypting & Uploading..."
                  : "Data Secured"}
              </span>
              <span
                className={
                  status === "analyzing"
                    ? "text-emerald-400 animate-pulse"
                    : "text-slate-400"
                }
              >
                {status === "analyzing"
                  ? "Running ViT Inference..."
                  : "Pending"}
              </span>
            </div>

            {/* Progress Bar Container */}
            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden border border-slate-700 relative">
              <div
                className={`h-full rounded-full transition-all duration-[3000ms] ease-out ${
                  status === "uploading"
                    ? "w-1/3 bg-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.8)]"
                    : "w-[95%] bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.8)]"
                }`}
              />
            </div>
          </div>
        </div>
      )}

      {status === "result" && previewUrl && (
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6 animate-fade-in">
          {/* Left: Final Image */}
          <div className="backdrop-blur-2xl bg-slate-900/60 border border-slate-800 rounded-2xl p-4 flex flex-col relative shadow-xl">
            <div className="flex-1 rounded-xl overflow-hidden border border-slate-800 bg-black relative flex items-center justify-center min-h-[350px]">
              <img
                src={previewUrl}
                alt="Analyzed Scan"
                className="absolute inset-0 w-full h-full object-cover opacity-60 filter grayscale contrast-125"
              />
              <div className="absolute inset-0 bg-[linear-gradient(rgba(6,182,212,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(6,182,212,0.1)_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />

              {/* Highlight Box mock */}
              <div className="absolute top-[35%] left-[40%] w-24 h-24 border-2 border-rose-500/50 bg-rose-500/20 rounded z-10 pointer-events-none shadow-[0_0_20px_rgba(225,29,72,0.3)]">
                <div className="absolute -top-6 left-0 text-[10px] font-mono font-bold text-rose-400 bg-black/80 px-2 py-0.5 rounded border border-rose-500/30">
                  ANOMALY
                </div>
              </div>
            </div>
          </div>

          {/* Right: Results Card */}
          <div className="backdrop-blur-2xl bg-slate-900/60 border border-emerald-500/30 rounded-2xl p-8 flex flex-col shadow-[inset_0_0_40px_rgba(16,185,129,0.05)]">
            <div className="flex items-center gap-3 mb-6">
              <CheckCircle className="w-8 h-8 text-emerald-400" />
              <h3 className="text-2xl font-bold text-white tracking-tight">
                Analysis Complete
              </h3>
            </div>

            <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 mb-6">
              <p className="text-slate-400 text-xs font-bold uppercase tracking-wider mb-2">
                Global Model Prediction
              </p>
              <div className="flex items-center justify-between">
                <span className="text-xl font-bold text-rose-400 flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5" />
                  Malignant Pattern Detected
                </span>
              </div>
            </div>

            <div className="space-y-4 mb-8">
              <div>
                <div className="flex justify-between text-xs font-medium uppercase tracking-wider mb-2">
                  <span className="text-slate-400">Confidence Interval</span>
                  <span className="text-emerald-400 font-mono text-sm">
                    94.2%
                  </span>
                </div>
                <div className="w-full bg-slate-800 rounded-full h-2">
                  <div className="bg-gradient-to-r from-emerald-500 to-emerald-300 h-2 rounded-full w-[94.2%] shadow-[0_0_10px_rgba(16,185,129,0.5)]" />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mt-6">
                <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                  <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-1">
                    Processing Time
                  </p>
                  <p className="font-mono text-cyan-400 text-sm flex items-center gap-2">
                    <Activity className="w-3 h-3" />
                    1.42s
                  </p>
                </div>
                <div className="bg-slate-950 p-4 rounded-lg border border-slate-800">
                  <p className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-1">
                    Enc. Protocol
                  </p>
                  <p className="font-mono text-emerald-400 text-sm flex items-center gap-2">
                    <ShieldCheck className="w-3 h-3" />
                    CKKS
                  </p>
                </div>
              </div>
            </div>

            {/* CTA to Explainability */}
            <div className="mt-auto space-y-4">
              <Link
                to="/explain"
                state={{ imageUrl: previewUrl }}
                className="w-full relative overflow-hidden rounded-xl font-bold text-sm tracking-wide py-4 px-5 flex items-center justify-between transition-all duration-300 bg-cyan-900/40 hover:bg-cyan-800/60 text-cyan-300 border border-cyan-500/50 hover:border-cyan-400 shadow-[0_0_20px_rgba(6,182,212,0.2)] hover:shadow-[0_0_30px_rgba(6,182,212,0.4)] group"
              >
                <div className="flex items-center gap-3">
                  <Brain className="w-5 h-5 group-hover:scale-110 transition-transform text-cyan-400" />
                  <span>View 2D Explainability</span>
                </div>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>

              <Link
                to="/viewer3d"
                state={{ imageUrl: previewUrl }}
                className="w-full relative overflow-hidden rounded-xl font-bold text-sm tracking-wide py-4 px-5 flex items-center justify-between transition-all duration-300 bg-purple-900/40 hover:bg-purple-800/60 text-purple-300 border border-purple-500/50 hover:border-purple-400 shadow-[0_0_20px_rgba(168,85,247,0.2)] hover:shadow-[0_0_30px_rgba(168,85,247,0.4)] group"
              >
                <div className="flex items-center gap-3">
                  <Hexagon className="w-5 h-5 group-hover:scale-110 transition-transform text-purple-400" />
                  <span>View 3D Volumetric</span>
                </div>
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>

            <button
              onClick={() => {
                updateStatus("idle");
                updatePreviewUrl(null);
                setFile(null);
              }}
              className="mt-6 text-xs text-slate-500 hover:text-slate-300 transition-colors uppercase tracking-widest font-bold"
            >
              Click to New Scan
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
