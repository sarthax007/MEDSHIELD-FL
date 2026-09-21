import React from "react";
import {
  ComposableMap,
  Geographies,
  Geography,
  Marker,
  Line,
} from "react-simple-maps";

const geoUrl = "/features.json";

export interface NodeMarker {
  id: string;
  name: string;
  coordinates: [number, number];
  isCentral?: boolean;
}

export const markers: NodeMarker[] = [
  {
    id: "NODE-4D",
    name: "Mumbai Care Institute",
    coordinates: [72.8777, 19.076],
  },
  {
    id: "NODE-3C",
    name: "Nanded General",
    coordinates: [77.311, 19.1383],
  },
  {
    id: "NODE-1A",
    name: "VIT Medical Center (Central)",
    coordinates: [73.8567, 18.5204],
    isCentral: true,
  },
  {
    id: "NODE-2B",
    name: "Pune City Hospital",
    coordinates: [73.9567, 18.6204],
  },
];

interface IndiaGeospatialMapProps {
  interactive?: boolean;
  onMapClick?: () => void;
  onNodeClick?: (node: NodeMarker) => void;
}

export default function IndiaGeospatialMap({
  interactive = false,
  onMapClick,
  onNodeClick,
}: IndiaGeospatialMapProps) {
  const centralNode = markers.find((m) => m.isCentral)!;

  // View configurations based on interactive mode
  const scale = interactive ? 6500 : 1200;
  const center: [number, number] = interactive ? [75, 19] : [80, 22];

  const mapContainerClasses = `w-full h-full flex items-center justify-center relative overflow-hidden bg-slate-900/40 rounded-xl ${
    !interactive
      ? "cursor-pointer hover:border-cyan-500/50 transition-colors border-cyan-500/20 shadow-[inset_0_0_30px_rgba(6,182,212,0.05)] border"
      : ""
  }`;

  return (
    <div
      className={mapContainerClasses}
      onClick={!interactive ? onMapClick : undefined}
    >
      {/* Background glow effects */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-cyan-500/10 blur-[100px] rounded-full pointer-events-none" />

      {/* Full screen title if interactive */}
      {interactive && (
        <div className="absolute top-8 left-8 z-20 pointer-events-none">
          <h2 className="text-3xl font-bold text-white flex items-center gap-3">
            <span className="text-emerald-400">●</span> Live Network Topology
          </h2>
          <p className="text-slate-400 mt-2 font-mono text-sm">
            Select a hospital node to view streaming security logs.
          </p>
        </div>
      )}

      <ComposableMap
        projection="geoMercator"
        projectionConfig={{
          scale,
          center,
        }}
        className="w-full h-full object-contain"
      >
        <Geographies geography={geoUrl}>
          {({ geographies }) =>
            geographies
              .filter((geo) => geo.properties.name === "India")
              .map((geo) => (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  fill="#0f172a" // slate-950
                  stroke="#06b6d4" // cyan-500
                  strokeWidth={0.5}
                  strokeOpacity={0.4}
                  style={
                    {
                      default: { outline: "none" },
                      hover: {
                        fill: "#1e293b",
                        outline: "none",
                        strokeOpacity: 0.8,
                      },
                      pressed: { outline: "none" },
                    } as any
                  }
                />
              ))
          }
        </Geographies>

        {/* Draw Connection Lines from Nodes to Central Server */}
        {markers.map((marker, i) => {
          if (marker.isCentral) return null;
          return (
            <Line
              key={`line-${i}`}
              from={marker.coordinates}
              to={centralNode.coordinates}
              stroke="#10b981" // emerald-500
              strokeWidth={interactive ? 2 : 1.5}
              strokeLinecap="round"
              className="animate-[dash_3s_linear_infinite]"
              style={{
                strokeDasharray: interactive ? "6 6" : "4 4",
                opacity: interactive ? 0.8 : 0.6,
              }}
            />
          );
        })}

        {/* Draw Nodes */}
        {markers.map((marker) => {
          const { name, coordinates, isCentral } = marker;
          return (
            <Marker key={name} coordinates={coordinates}>
              <g
                className={`group ${
                  interactive && !isCentral ? "cursor-pointer" : ""
                }`}
                onClick={(e) => {
                  if (interactive && onNodeClick && !isCentral) {
                    e.stopPropagation();
                    onNodeClick(marker);
                  }
                }}
              >
                {/* Outer pulsing ring */}
                <circle
                  r={isCentral ? (interactive ? 12 : 8) : interactive ? 10 : 6}
                  fill={isCentral ? "#06b6d4" : "#10b981"}
                  className="animate-ping opacity-75"
                />
                {/* Inner dot */}
                <circle
                  r={isCentral ? (interactive ? 7 : 5) : interactive ? 6 : 4}
                  fill={isCentral ? "#22d3ee" : "#34d399"}
                  stroke="#fff"
                  strokeWidth={1}
                  className="animate-pulse"
                />

                {interactive ? (
                  // Large Text for Interactive Mode
                  <>
                    <rect
                      x="-80"
                      y="-40"
                      width="160"
                      height="25"
                      rx="6"
                      className="fill-slate-900/90 stroke-cyan-500/50 opacity-90 transition-opacity duration-300 group-hover:stroke-emerald-400 group-hover:shadow-[0_0_15px_rgba(16,185,129,0.5)]"
                    />
                    <text
                      textAnchor="middle"
                      y={-22}
                      fontSize={15}
                      fontWeight="bold"
                      className={`fill-white font-mono opacity-100 drop-shadow-[0_0_10px_rgba(6,182,212,1)] ${
                        !isCentral ? "group-hover:fill-emerald-300" : ""
                      } transition-colors`}
                    >
                      {name}
                    </text>
                  </>
                ) : (
                  // Small Text for Thumbnail Mode
                  <>
                    <rect
                      x="-60"
                      y="-32"
                      width="120"
                      height="20"
                      rx="4"
                      className="fill-slate-900/80 stroke-cyan-500/30 opacity-70 group-hover:opacity-100 transition-opacity duration-300"
                    />
                    <text
                      textAnchor="middle"
                      y={-18}
                      className="fill-white text-xs font-bold font-mono opacity-80 group-hover:opacity-100 transition-all duration-300 drop-shadow-[0_0_8px_rgba(6,182,212,0.8)]"
                    >
                      {name}
                    </text>
                  </>
                )}
              </g>
            </Marker>
          );
        })}
      </ComposableMap>

      {/* Stats Overlay */}
      {!interactive && (
        <div className="absolute bottom-4 left-4 flex flex-col gap-2 pointer-events-none">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span className="text-xs font-mono text-cyan-400">
              Central Aggregator
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs font-mono text-emerald-400">
              Active FL Node
            </span>
          </div>
        </div>
      )}

      {/* Interactive Legend Overlay */}
      {interactive && (
        <div className="absolute bottom-12 right-12 flex flex-col gap-4 pointer-events-none bg-slate-900/80 p-6 rounded-xl border border-slate-700/50 backdrop-blur-md">
          <h3 className="text-white font-bold mb-2">Network Legend</h3>
          <div className="flex items-center gap-3">
            <span className="w-3 h-3 rounded-full bg-cyan-400 animate-pulse" />
            <span className="text-sm font-mono text-slate-300">
              Central Aggregator Server
            </span>
          </div>
          <div className="flex items-center gap-3">
            <span className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-sm font-mono text-slate-300">
              Active SecAgg Client Node
            </span>
          </div>
          <div className="flex items-center gap-3 mt-2">
            <div className="w-6 h-0 border-t-2 border-dashed border-emerald-500 animate-[dash_3s_linear_infinite]" />
            <span className="text-sm font-mono text-slate-300">
              Encrypted Homomorphic Tunnel
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
