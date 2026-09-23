import { Navigate, Outlet } from "react-router-dom";
import { useAuth } from "../lib/AuthContext";

export default function ProtectedRoute() {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    // Redirect them to the /login page, but save the current location they were
    // trying to go to if you want to enhance this later.
    return <Navigate to="/login" replace />;
  }

  // If authenticated, render the child routes (the AppShell usually)
  return <Outlet />;
}
