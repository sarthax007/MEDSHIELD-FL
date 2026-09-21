import React from "react";

export default function DoctorQueue() {
  return (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg border border-gray-200 shadow-sm">
        <h2 className="text-lg font-medium text-gray-900 mb-4">
          Active Learning Labelling Queue
        </h2>
        <p className="text-gray-500">
          Uncertain MRI images will be queued here for expert review.
        </p>
      </div>
    </div>
  );
}
