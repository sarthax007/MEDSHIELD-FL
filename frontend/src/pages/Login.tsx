import React, { useState } from "react";
import { Activity, Lock, User, AlertCircle } from "lucide-react";
import { apiClient } from "../lib/api";
import { useAuth } from "../lib/AuthContext";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { login } = useAuth();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      // FastAPI OAuth2PasswordRequestForm expects x-www-form-urlencoded
      const formData = new URLSearchParams();
      formData.append("username", username);
      formData.append("password", password);

      const response = await apiClient.post("/auth/login", formData, {
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
      });

      // Based on our FastAPI response model
      const { access_token } = response.data;

      // In a real app, you might decode the JWT here to get the role/username,
      // or fetch the current user profile. For now, we simulate.
      login(access_token, username, "admin");
    } catch (err: any) {
      if (err.response && err.response.status === 401) {
        setError("Invalid username or password");
      } else {
        setError("Network error. Is the backend running?");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center relative overflow-hidden">
      {/* Aesthetic glowing background blobs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-900/30 rounded-full blur-[128px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-emerald-900/20 rounded-full blur-[128px] pointer-events-none" />

      {/* Glassmorphism Login Card */}
      <div className="relative z-10 w-full max-w-md p-8 rounded-2xl backdrop-blur-md bg-slate-900/80 border border-slate-800 shadow-2xl">
        <div className="flex flex-col items-center mb-8">
          <div className="p-3 bg-slate-800/50 rounded-xl border border-slate-700/50 mb-4 shadow-[0_0_15px_rgba(6,182,212,0.3)]">
            <Activity className="w-8 h-8 text-cyan-400" />
          </div>
          <h2 className="text-2xl font-bold text-white tracking-tight">
            MedShield-FL
          </h2>
          <p className="text-slate-400 mt-2 text-sm">Secure Clinical Access</p>
        </div>

        {error && (
          <div className="mb-6 p-4 rounded-lg bg-red-950/50 border border-red-900/50 flex items-start">
            <AlertCircle className="w-5 h-5 text-red-400 mr-3 flex-shrink-0 mt-0.5" />
            <p className="text-sm text-red-200">{error}</p>
          </div>
        )}

        <form onSubmit={handleLogin} className="space-y-5">
          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-300 ml-1">
              Username
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <User className="h-5 w-5 text-slate-500" />
              </div>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full bg-slate-950/50 border border-slate-700 text-slate-100 rounded-lg pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500 transition-all placeholder-slate-600"
                placeholder="Enter clinical ID"
                required
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-slate-300 ml-1">
              Password
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Lock className="h-5 w-5 text-slate-500" />
              </div>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full bg-slate-950/50 border border-slate-700 text-slate-100 rounded-lg pl-10 pr-4 py-3 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500 transition-all placeholder-slate-600"
                placeholder="••••••••"
                required
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className={`w-full py-3 px-4 flex justify-center items-center rounded-lg text-sm font-bold text-slate-950 bg-gradient-to-r from-cyan-400 to-emerald-400 hover:from-cyan-300 hover:to-emerald-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-cyan-400 transition-all shadow-[0_0_20px_rgba(6,182,212,0.4)] ${
              isLoading ? "opacity-70 cursor-not-allowed" : ""
            }`}
          >
            {isLoading ? "Authenticating..." : "Secure Login"}
          </button>
        </form>

        <div className="mt-8 text-center border-t border-slate-800/50 pt-6">
          <p className="text-xs text-slate-500">
            Protected by MedShield-FL Homomorphic Encryption.
            <br />
            Unauthorised access is strictly prohibited.
          </p>
        </div>
      </div>
    </div>
  );
}
