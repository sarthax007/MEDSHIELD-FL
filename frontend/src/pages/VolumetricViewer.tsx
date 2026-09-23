import {
  useState,
  useRef,
  useMemo,
  Suspense,
  Component,
  type ErrorInfo,
  type ReactNode,
} from "react";
import { useLocation, Link } from "react-router-dom";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  OrbitControls,
  Html,
  Center,
  Bounds,
  useGLTF,
} from "@react-three/drei";
import {
  ArrowLeft,
  Hexagon,
  Eye,
  SlidersHorizontal,
  Share2,
  X,
} from "lucide-react";
import * as THREE from "three";

// --- Custom UI Components ---

function createNoisyGeometry(
  baseGeo: THREE.BufferGeometry,
  noiseScale: number,
  ridgeFrequency: number,
  amplitude: number,
) {
  const posAttribute = baseGeo.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < posAttribute.count; i++) {
    v.fromBufferAttribute(posAttribute, i);
    const noise =
      Math.sin(v.x * noiseScale) *
      Math.cos(v.y * (noiseScale + 1)) *
      Math.sin(v.z * noiseScale);
    const ridges = Math.sin(v.x * ridgeFrequency + v.y * ridgeFrequency) * 0.05;
    v.multiplyScalar(1 + noise * amplitude + ridges);
    posAttribute.setXYZ(i, v.x, v.y, v.z);
  }
  baseGeo.computeVertexNormals();
  return baseGeo;
}

class ErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }
  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("ErrorBoundary caught an error", error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            padding: 20,
            background: "red",
            color: "white",
            position: "absolute",
            inset: 0,
            zIndex: 9999,
          }}
        >
          <h1>Something went wrong.</h1>
          <pre>{this.state.error?.toString()}</pre>
          <pre>{this.state.error?.stack}</pre>
        </div>
      );
    }
    return this.props.children;
  }
}

function ColorPicker({
  value,
  onChange,
}: {
  value: string;
  onChange: (c: string) => void;
}) {
  const PREDEFINED_COLORS = [
    { id: "none", value: "none", bg: "bg-slate-800 border-2 border-slate-600" },
    {
      id: "black",
      value: "#0f172a",
      bg: "bg-[#0f172a] border border-slate-700",
    },
    {
      id: "white",
      value: "#f8fafc",
      bg: "bg-[#f8fafc] border border-slate-300",
    },
    { id: "red", value: "#ef4444", bg: "bg-[#ef4444] border border-red-500" },
    { id: "blue", value: "#3b82f6", bg: "bg-[#3b82f6] border border-blue-500" },
    {
      id: "green",
      value: "#22c55e",
      bg: "bg-[#22c55e] border border-green-500",
    },
    {
      id: "yellow",
      value: "#eab308",
      bg: "bg-[#eab308] border border-yellow-500",
    },
  ];

  return (
    <div className="flex flex-wrap items-center gap-1.5 mt-2">
      {PREDEFINED_COLORS.map((c) => (
        <button
          key={c.id}
          onClick={() => onChange(c.value)}
          className={`w-6 h-6 rounded-md flex items-center justify-center transition-all ${
            c.bg
          } ${
            value === c.value
              ? "ring-2 ring-white ring-offset-2 ring-offset-slate-900 scale-110"
              : "hover:scale-105"
          }`}
          title={c.id}
        >
          {c.id === "none" && <X className="w-4 h-4 text-slate-400" />}
        </button>
      ))}
    </div>
  );
}

// --- 3D Scene Components ---

function ProceduralBrain({
  wireframe,
  opacity,
  isolateTumor,
  color,
}: {
  wireframe: boolean;
  opacity: number;
  isolateTumor: boolean;
  color: string;
}) {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state, delta) => {
    if (groupRef.current && !isolateTumor) {
      groupRef.current.rotation.y += delta * 0.05;
      groupRef.current.rotation.z =
        Math.sin(state.clock.elapsedTime * 0.5) * 0.02;
    }
  });

  const brainMaterial = useMemo(
    () =>
      new THREE.MeshPhysicalMaterial({
        color: color === "none" ? "#ffffff" : color,
        emissive: color === "none" ? "#ffffff" : color,
        emissiveIntensity: color === "none" ? 0.05 : wireframe ? 0.5 : 0.2,
        transparent: true,
        opacity: color === "none" ? 0.15 : opacity,
        wireframe: wireframe,
        roughness: color === "none" ? 0 : 0.4,
        transmission: color === "none" ? 1 : 0,
        thickness: color === "none" ? 2 : 0,
        ior: color === "none" ? 1.5 : 1,
        clearcoat: color === "none" ? 1 : 0,
        side: THREE.DoubleSide,
      }),
    [color, wireframe, opacity],
  );

  // Procedurally generate a highly recognizable composite anatomical brain
  const geometries = useMemo(() => {
    // Left Hemisphere
    const leftHemi = createNoisyGeometry(
      new THREE.SphereGeometry(2.3, 64, 64),
      5,
      12,
      0.04,
    );
    leftHemi.scale(0.85, 0.85, 1.25);
    leftHemi.translate(-0.85, 0, 0);

    // Right Hemisphere
    const rightHemi = createNoisyGeometry(
      new THREE.SphereGeometry(2.3, 64, 64),
      5,
      12,
      0.04,
    );
    rightHemi.scale(0.85, 0.85, 1.25);
    rightHemi.translate(0.85, 0, 0);

    // Cerebellum (bottom back)
    const cerebellum = createNoisyGeometry(
      new THREE.SphereGeometry(1.2, 32, 32),
      8,
      20,
      0.06,
    );
    cerebellum.scale(1.4, 0.6, 0.8);
    cerebellum.translate(0, -1.7, -1.4);

    // Brainstem
    const stem = new THREE.CylinderGeometry(0.4, 0.3, 2.0, 32);
    stem.translate(0, -2.5, -0.7);
    stem.rotateX(0.2);

    return { leftHemi, rightHemi, cerebellum, stem };
  }, []);

  if (isolateTumor) return null;

  return (
    <group ref={groupRef} position={[0, -0.5, 0]}>
      <Center>
        {/* 4 Anatomical Components sharing the same material */}
        <mesh geometry={geometries.leftHemi} material={brainMaterial} />
        <mesh geometry={geometries.rightHemi} material={brainMaterial} />
        <mesh geometry={geometries.cerebellum} material={brainMaterial} />
        <mesh geometry={geometries.stem} material={brainMaterial} />
      </Center>

      {!isolateTumor && (
        <Html position={[0, 2.6, 0]} center>
          <div className="flex flex-col items-center pointer-events-none group">
            <div className="w-8 h-8 rounded-full bg-slate-900/80 border-2 border-slate-400 flex items-center justify-center text-white font-bold shadow-lg group-hover:scale-110 transition-transform">
              1
            </div>
            <div className="mt-2 px-3 py-1 bg-slate-900/90 text-slate-200 text-sm font-semibold rounded shadow-xl whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity">
              Cranial Volume
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}

// Preload the GLB to avoid suspense flickering
useGLTF.preload("/brain.glb");

function TumorMass({ color }: { color: string }) {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state, delta) => {
    if (meshRef.current) {
      const scale = 1 + Math.sin(state.clock.elapsedTime * 3) * 0.03;
      meshRef.current.scale.set(scale, scale, scale);
      // Ensure the tumor rotates in sync if we want it to, or just pulse
      meshRef.current.rotation.y -= delta * 0.1;
    }
  });

  return (
    <mesh ref={meshRef} position={[1.2, 0.8, 1.0]}>
      <dodecahedronGeometry args={[0.6, 3]} />
      <meshStandardMaterial
        color={color === "none" ? "#ffffff" : color}
        emissive={color === "none" ? "#ffffff" : color}
        emissiveIntensity={color === "none" ? 0.1 : 0.8}
        roughness={color === "none" ? 0.1 : 0.4}
        transparent={color === "none"}
        opacity={color === "none" ? 0.4 : 1}
        wireframe={false}
      />
      <Html position={[0, 1.2, 0]} center>
        <div className="flex flex-col items-center pointer-events-none group">
          <div className="w-8 h-8 rounded-full bg-slate-900/80 border-2 border-rose-500 flex items-center justify-center text-rose-400 font-bold shadow-lg group-hover:scale-110 transition-transform">
            2
          </div>
          <div className="mt-2 px-3 py-1 bg-slate-900/90 text-rose-400 text-sm font-semibold rounded shadow-xl whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity">
            Malignant Tumor
          </div>
        </div>
      </Html>
    </mesh>
  );
}

// Unified group for coherent rotation and OrbitControls center
function AnatomicalVolume({
  wireframe,
  opacity,
  isolateTumor,
  brainColor,
  tumorColor,
}: any) {
  const groupRef = useRef<THREE.Group>(null);

  // Slowly rotate the entire anatomical group so the tumor and brain stay perfectly relative to each other
  useFrame((_, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.05;
    }
  });

  return (
    <group ref={groupRef}>
      <ProceduralBrain
        wireframe={wireframe}
        opacity={opacity}
        isolateTumor={isolateTumor}
        color={brainColor}
      />
      <TumorMass color={tumorColor} />
    </group>
  );
}

function NeuralNetworkNodes() {
  const pointsRef = useRef<THREE.Points>(null);

  const particlesPosition = useMemo(() => {
    const positions = new Float32Array(600 * 3);
    for (let i = 0; i < 600; i++) {
      // Scatter points in a sphere around the brain to simulate nodes
      const r = 3 + Math.random() * 3;
      const theta = 2 * Math.PI * Math.random();
      const phi = Math.acos(2 * Math.random() - 1);
      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.sin(phi) * Math.sin(theta);
      const z = r * Math.cos(phi);
      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;
    }
    return positions;
  }, []);

  useFrame((_state, delta) => {
    if (pointsRef.current) {
      pointsRef.current.rotation.y += delta * 0.02;
      pointsRef.current.rotation.z -= delta * 0.01;
    }
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={particlesPosition.length / 3}
          array={particlesPosition}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.06}
        color="#06b6d4"
        transparent
        opacity={0.6}
        sizeAttenuation={true}
      />
    </points>
  );
}

// --- Main Page Component ---

export default function VolumetricViewer() {
  const location = useLocation();
  const [imageUrl] = useState<string | null>(() => {
    if (location.state?.imageUrl) {
      sessionStorage.setItem("prediction_previewUrl", location.state.imageUrl);
      return location.state.imageUrl;
    }
    return sessionStorage.getItem("prediction_previewUrl");
  });

  const [wireframe, setWireframe] = useState(true);
  const [isolateTumor, setIsolateTumor] = useState(false);
  const [opacity, setOpacity] = useState(0.5);
  const [brainColor, setBrainColor] = useState("#ef4444");
  const [tumorColor, setTumorColor] = useState("#22c55e");
  const [isExporting, setIsExporting] = useState(false);

  const handleExport = () => {
    setIsExporting(true);
    setTimeout(() => {
      const dummyData = JSON.stringify({
        model: "Brain Tumor Volumetric",
        color: brainColor,
        tumorColor: tumorColor,
        opacity: opacity,
        mesh: "Procedural",
      });
      const blob = new Blob([dummyData], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "brain_tumor_model.gltf.json";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setIsExporting(false);
    }, 1500);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-slate-800 shrink-0">
        <div>
          <div className="flex items-center gap-3">
            <Link
              to="/predict"
              className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-400 hover:text-white"
            >
              <ArrowLeft className="w-5 h-5" />
            </Link>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <Hexagon className="w-8 h-8 text-purple-400" />
              Interactive 3D Volumetric
            </h1>
          </div>
          <p className="text-slate-400 mt-2 text-sm ml-14">
            High-fidelity neural network spatial visualization.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleExport}
            disabled={isExporting}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-sm font-semibold flex items-center gap-2 transition-colors disabled:opacity-50"
          >
            <Share2
              className={`w-4 h-4 ${
                isExporting ? "animate-pulse text-purple-400" : ""
              }`}
            />
            {isExporting ? "Exporting..." : "Export Model"}
          </button>
        </div>
      </div>

      {!imageUrl ? (
        <div className="mt-12 flex flex-col items-center justify-center py-20 px-4 text-center rounded-3xl backdrop-blur-2xl bg-slate-900/60 border-2 border-dashed border-slate-700 flex-1">
          <Hexagon className="w-16 h-16 text-slate-600 mb-4" />
          <h2 className="text-xl font-bold text-slate-300 mb-2">
            No Scan Data Found
          </h2>
          <p className="text-slate-500 max-w-md mb-8">
            Please run an AI inference on an MRI scan first to construct the 3D
            volume.
          </p>
          <Link
            to="/predict"
            className="px-6 py-3 bg-purple-600 hover:bg-purple-500 text-white font-bold rounded-xl transition-all shadow-[0_0_20px_rgba(168,85,247,0.4)]"
          >
            Run New Scan
          </Link>
        </div>
      ) : (
        <div className="flex flex-col md:flex-row gap-6 mt-6 flex-1 min-h-0">
          {/* Model Inspector Sidebar */}
          <div className="w-full md:w-80 flex flex-col gap-4 shrink-0 overflow-y-auto">
            <div className="backdrop-blur-2xl bg-slate-900/80 border border-purple-500/30 rounded-2xl p-5 shadow-[inset_0_0_20px_rgba(168,85,247,0.05)]">
              <h3 className="text-lg font-bold text-white flex items-center gap-2 mb-6 pb-4 border-b border-slate-800">
                <SlidersHorizontal className="w-5 h-5 text-purple-400" />
                Model Inspector
              </h3>

              <div className="space-y-6">
                {/* Wireframe Toggle */}
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-semibold text-slate-200">
                      Wireframe Mesh
                    </p>
                    <p className="text-xs text-slate-500">
                      Render structural topology
                    </p>
                  </div>
                  <button
                    onClick={() => setWireframe(!wireframe)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-slate-900 ${
                      wireframe ? "bg-purple-500" : "bg-slate-700"
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        wireframe ? "translate-x-6" : "translate-x-1"
                      }`}
                    />
                  </button>
                </div>

                {/* Isolate Tumor Toggle */}
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-semibold text-slate-200">
                      Isolate Anomaly
                    </p>
                    <p className="text-xs text-slate-500">
                      Hide healthy brain tissue
                    </p>
                  </div>
                  <button
                    onClick={() => setIsolateTumor(!isolateTumor)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-rose-500 focus:ring-offset-2 focus:ring-offset-slate-900 ${
                      isolateTumor ? "bg-rose-500" : "bg-slate-700"
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        isolateTumor ? "translate-x-6" : "translate-x-1"
                      }`}
                    />
                  </button>
                </div>

                {/* Opacity Slider */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-semibold text-slate-200">
                      Tissue Opacity
                    </p>
                    <span className="text-xs font-mono text-purple-400">
                      {(opacity * 100).toFixed(0)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1"
                    step="0.05"
                    value={opacity}
                    onChange={(e) => setOpacity(parseFloat(e.target.value))}
                    disabled={isolateTumor}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500 focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50"
                  />
                </div>

                {/* Surface / Wireframe Color */}
                <div className="pt-2 border-t border-slate-800">
                  <div className="mb-2">
                    <p className="text-sm font-semibold text-slate-200">
                      Surface / Wireframe Color
                    </p>
                    <p className="text-xs text-slate-500">
                      Solid color or wireframe tint
                    </p>
                  </div>
                  <ColorPicker value={brainColor} onChange={setBrainColor} />
                </div>

                {/* Tumor Color */}
                <div className="pt-2 border-t border-slate-800">
                  <div className="mb-2">
                    <p className="text-sm font-semibold text-slate-200">
                      Tumor Color
                    </p>
                  </div>
                  <ColorPicker value={tumorColor} onChange={setTumorColor} />
                </div>
              </div>

              <div className="mt-8 p-4 bg-purple-500/10 border border-purple-500/20 rounded-xl">
                <div className="flex items-start gap-3">
                  <Eye className="w-5 h-5 text-purple-400 shrink-0 mt-0.5" />
                  <div>
                    <p className="text-xs font-bold uppercase tracking-wider text-purple-400 mb-1">
                      Spatial Navigation
                    </p>
                    <p className="text-xs text-slate-400">
                      Left-click and drag to rotate the volume. Scroll to zoom.
                      Right-click to pan across the topological space.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* 3D Canvas */}
          <div className="flex-1 relative rounded-2xl overflow-hidden border border-slate-700/50 bg-black shadow-[inset_0_0_50px_rgba(0,0,0,1)]">
            <ErrorBoundary>
              <Canvas camera={{ position: [0, 0, 8], fov: 45 }}>
                <color attach="background" args={["#020617"]} />

                {/* Cyber-Dark Lighting */}
                <ambientLight intensity={0.5} />
                <directionalLight
                  position={[10, 10, 5]}
                  intensity={1.5}
                  color="#06b6d4"
                />
                <directionalLight
                  position={[-10, -10, -5]}
                  intensity={1}
                  color="#10b981"
                />
                <pointLight
                  position={[0, 0, 0]}
                  intensity={2}
                  color="#e11d48"
                  distance={5}
                />

                <Suspense fallback={null}>
                  <AnatomicalVolume
                    wireframe={wireframe}
                    opacity={opacity}
                    isolateTumor={isolateTumor}
                    brainColor={brainColor}
                    tumorColor={tumorColor}
                  />
                </Suspense>

                <NeuralNetworkNodes />

                <OrbitControls
                  enablePan={true}
                  enableZoom={true}
                  enableRotate={true}
                  autoRotate={!isolateTumor && !wireframe}
                  autoRotateSpeed={0.5}
                />
              </Canvas>
            </ErrorBoundary>
          </div>
        </div>
      )}
    </div>
  );
}
