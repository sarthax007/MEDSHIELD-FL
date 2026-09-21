import React from "react";

export default function Explainability() {
  return (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
        <h2 className="text-lg font-medium text-gray-900 mb-4">
          AI Explainability (Grad-CAM)
        </h2>
        <p className="text-gray-500">
          Upload an MRI scan to receive a diagnostic heatmap and confidence
          score.
        </p>
      </div>
    </div>
  );
}
