import { Button } from "@/components/ui/button";

function App() {
  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4">
      <div className="bg-white p-8 rounded-xl shadow-sm text-center max-w-md w-full">
        <h1 className="text-2xl font-bold text-slate-900 mb-4">
          MedShield-FL Dashboard
        </h1>
        <p className="text-slate-600 mb-8">
          Welcome to the privacy-preserving federated learning dashboard.
        </p>
        <Button onClick={() => alert("Tailwind & shadcn/ui are working!")}>
          Test Component
        </Button>
      </div>
    </div>
  );
}

export default App;
