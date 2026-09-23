import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./lib/AuthContext";
import AppShell from "./layouts/AppShell";
import ProtectedRoute from "./components/ProtectedRoute";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Admin from "./pages/Admin";
import DoctorQueue from "./pages/DoctorQueue";
import Explainability from "./pages/Explainability";
import VolumetricViewer from "./pages/VolumetricViewer"; // Refresh TS Server

import Prediction from "./pages/Prediction";

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<Login />} />

          <Route element={<ProtectedRoute />}>
            <Route path="/" element={<AppShell />}>
              <Route index element={<Dashboard />} />
              <Route path="admin" element={<Admin />} />
              <Route path="doctor" element={<DoctorQueue />} />
              <Route path="predict" element={<Prediction />} />
              <Route path="explain" element={<Explainability />} />
              <Route path="viewer3d" element={<VolumetricViewer />} />
            </Route>
          </Route>

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
