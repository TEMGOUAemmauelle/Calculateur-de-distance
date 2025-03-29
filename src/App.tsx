import React, { useRef, useState, useEffect } from 'react';
import { Camera, Ruler } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

const KNOWN_OBJECTS = {
  'visage humain': { width: 15, cocoClass: 'person' },
  'feuille A4': { width: 21, cocoClass: 'book' },
  'carte bancaire': { width: 8.5, cocoClass: 'cell phone' },
  'téléphone portable': { width: 7, cocoClass: 'cell phone' },
};

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedObject, setSelectedObject] = useState<string>('visage humain');
  const [distance, setDistance] = useState<number | null>(null);
  const [error, setError] = useState<string>('');
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [detectedObjects, setDetectedObjects] = useState<cocoSsd.DetectedObject[]>([]);
  const animationFrameRef = useRef<number>();

  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        const loadedModel = await cocoSsd.load();
        setModel(loadedModel);
      } catch (err) {
        setError('Erreur de chargement du modèle de détection');
        console.error(err);
      }
    };

    loadModel();
  }, []);

  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { facingMode: 'user' } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsStreaming(true);
          setError('');
        }
      } catch (err) {
        setError('Erreur d\'accès à la caméra');
        console.error(err);
      }
    };

    startCamera();

    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach(track => track.stop());
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  const detectObjects = async () => {
    if (!model || !videoRef.current || !canvasRef.current) return;

    const predictions = await model.detect(videoRef.current);
    setDetectedObjects(predictions);

    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    if (!context) return;

    // Ajuster la taille du canvas à celle de la vidéo
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    // Dessiner l'image de la vidéo
    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    // Dessiner les cadres de détection
    predictions.forEach(prediction => {
      const [x, y, width, height] = prediction.bbox;
      
      // Cadre principal
      context.strokeStyle = '#22c55e'; // vert
      context.lineWidth = 4;
      context.strokeRect(x, y, width, height);

      // Coins décoratifs
      context.strokeStyle = '#eab308'; // jaune
      const cornerSize = 20;
      
      // Coin supérieur gauche
      context.beginPath();
      context.moveTo(x, y + cornerSize);
      context.lineTo(x, y);
      context.lineTo(x + cornerSize, y);
      context.stroke();

      // Coin supérieur droit
      context.beginPath();
      context.moveTo(x + width - cornerSize, y);
      context.lineTo(x + width, y);
      context.lineTo(x + width, y + cornerSize);
      context.stroke();

      // Coin inférieur gauche
      context.beginPath();
      context.moveTo(x, y + height - cornerSize);
      context.lineTo(x, y + height);
      context.lineTo(x + cornerSize, y + height);
      context.stroke();

      // Coin inférieur droit
      context.beginPath();
      context.moveTo(x + width - cornerSize, y + height);
      context.lineTo(x + width, y + height);
      context.lineTo(x + width, y + height - cornerSize);
      context.stroke();

      // Étiquette
      const label = `${prediction.class} (${Math.round(prediction.score * 100)}%)`;
      context.fillStyle = 'rgba(37, 99, 235, 0.9)';
      context.fillRect(x, y - 30, context.measureText(label).width + 10, 30);
      context.fillStyle = 'white';
      context.font = '16px Arial';
      context.fillText(label, x + 5, y - 10);

      // Calculer la distance si l'objet correspond à la sélection
      if (prediction.class === KNOWN_OBJECTS[selectedObject].cocoClass) {
        const knownWidth = KNOWN_OBJECTS[selectedObject].width;
        const focalLength = 500; // Calibration approximative
        const estimatedDistance = (knownWidth * focalLength) / width;
        setDistance(Math.round(estimatedDistance));
      }
    });

    animationFrameRef.current = requestAnimationFrame(detectObjects);
  };

  useEffect(() => {
    if (isStreaming && model) {
      detectObjects();
    }
  }, [isStreaming, model, selectedObject]);

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-2xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          <div className="p-6">
            <div className="flex items-center gap-3 mb-6">
              <Camera className="w-8 h-8 text-blue-600" />
              <h1 className="text-2xl font-bold text-gray-800">
                Calculateur de Distance
              </h1>
            </div>

            {error && (
              <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
                {error}
              </div>
            )}

            <div className="relative aspect-video bg-black rounded-lg overflow-hidden mb-6">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                className="absolute top-0 left-0 w-full h-full object-cover"
                onPlay={() => setIsStreaming(true)}
              />
              <canvas
                ref={canvasRef}
                className="absolute top-0 left-0 w-full h-full"
              />
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Objet de référence
                </label>
                <select
                  value={selectedObject}
                  onChange={(e) => setSelectedObject(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {Object.entries(KNOWN_OBJECTS).map(([key]) => (
                    <option key={key} value={key}>
                      {key}
                    </option>
                  ))}
                </select>
              </div>

              {distance !== null && (
                <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-md">
                  <p className="text-center text-lg">
                    Distance estimée : 
                    <span className="font-bold"> {distance} cm</span>
                  </p>
                </div>
              )}

              <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-md">
                <h3 className="font-medium text-blue-800 mb-2">Objets détectés :</h3>
                <ul className="space-y-1">
                  {detectedObjects.map((obj, index) => (
                    <li key={index} className="text-blue-600">
                      {obj.class} ({Math.round(obj.score * 100)}% de confiance)
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

/*
import React, { useRef, useState, useEffect } from 'react';
import { Camera, Ruler, Moon, Sun } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import Confetti from 'react-confetti';

const KNOWN_OBJECTS = {
  'visage humain': { width: 15, cocoClass: 'person', emoji: '👤' },
  'feuille A4': { width: 21, cocoClass: 'book', emoji: '📄' },
  'carte bancaire': { width: 8.5, cocoClass: 'credit card', emoji: '💳' },
  'téléphone portable': { width: 7, cocoClass: 'cell phone', emoji: '📱' },
};

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedObject, setSelectedObject] = useState<string>('visage humain');
  const [distance, setDistance] = useState<number | null>(null);
  const [error, setError] = useState<string>('');
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [detectedObjects, setDetectedObjects] = useState<cocoSsd.DetectedObject[]>([]);
  const [darkMode, setDarkMode] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);
  const animationFrameRef = useRef<number>();

  // Chargement du modèle TensorFlow
  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.ready();
        const loadedModel = await cocoSsd.load();
        setModel(loadedModel);
      } catch (err) {
        setError('Erreur de chargement du modèle IA');
        console.error(err);
      }
    };

    loadModel();
  }, []);

  // Initialisation de la caméra
  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { facingMode: 'user' } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsStreaming(true);
          setError('');
        }
      } catch (err) {
        setError('Accès caméra refusé - Activez les permissions');
        console.error(err);
      }
    };

    startCamera();

    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach(track => track.stop());
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Détection des objets et calcul de distance
  const detectObjects = async () => {
    if (!model || !videoRef.current || !canvasRef.current) return;

    const predictions = await model.detect(videoRef.current);
    setDetectedObjects(predictions);

    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    if (!context) return;

    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    let foundSelected = false;
    
    predictions.forEach(prediction => {
      const [x, y, width, height] = prediction.bbox;
      const isSelected = prediction.class === KNOWN_OBJECTS[selectedObject].cocoClass;
      
      // Style du cadre selon l'objet
      context.strokeStyle = isSelected ? '#22d3ee' : '#a78bfa';
      context.lineWidth = isSelected ? 4 : 2;
      context.strokeRect(x, y, width, height);

      // Étiquette colorée
      const label = `${prediction.class} (${Math.round(prediction.score * 100)}%)`;
      context.fillStyle = isSelected ? 'rgba(34, 211, 238, 0.8)' : 'rgba(167, 139, 250, 0.8)';
      context.fillRect(x, y - 25, context.measureText(label).width + 10, 25);
      context.fillStyle = 'white';
      context.font = '14px Arial';
      context.fillText(label, x + 5, y - 8);

      // Calcul de distance pour l'objet sélectionné
      if (isSelected) {
        foundSelected = true;
        const knownWidth = KNOWN_OBJECTS[selectedObject].width;
        const focalLength = 600; // Valeur calibrée
        const estimatedDistance = (knownWidth * focalLength) / width;
        setDistance(Math.round(estimatedDistance));
      }
    });

    if (foundSelected && !showConfetti) {
      setShowConfetti(true);
      setTimeout(() => setShowConfetti(false), 3000);
    } else if (!foundSelected) {
      setDistance(null);
    }

    animationFrameRef.current = requestAnimationFrame(detectObjects);
  };

  useEffect(() => {
    if (isStreaming && model) {
      detectObjects();
    }
  }, [isStreaming, model, selectedObject]);

  return (
    <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'bg-gray-900 text-white' : 'bg-gradient-to-br from-blue-50 to-purple-50 text-gray-900'}`}>
      {showConfetti && (
        <Confetti 
          width={window.innerWidth}
          height={window.innerHeight}
          recycle={false}
          numberOfPieces={150}
          colors={['#22d3ee', '#a78bfa', '#f472b6', '#fbbf24']}
        />
      )}

      <div className="max-w-4xl mx-auto p-4 md:p-8">
        <div className={`rounded-3xl overflow-hidden shadow-2xl transition-all duration-300 ${darkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white/90 backdrop-blur-md border border-white/20'}`}>
         
          <div className={`p-6 flex justify-between items-center ${darkMode ? 'bg-gray-700' : 'bg-gradient-to-r from-blue-100 to-purple-100'}`}>
            <div className="flex items-center gap-4">
              <div className={`p-3 rounded-full ${darkMode ? 'bg-gray-600' : 'bg-white'} shadow-lg`}>
                <Camera className={`w-6 h-6 ${darkMode ? 'text-blue-300' : 'text-blue-600'}`} />
              </div>
              <h1 className={`text-3xl font-bold ${darkMode ? 'text-blue-300' : 'bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600'}`}>
                Magic Distance Meter
              </h1>
            </div>
            <button 
              onClick={() => setDarkMode(!darkMode)}
              className={`p-3 rounded-full ${darkMode ? 'bg-gray-600 text-yellow-300' : 'bg-white text-gray-700'} shadow-lg hover:scale-110 transition-transform`}
            >
              {darkMode ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </div>

         
          <div className="relative aspect-video group">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className={`absolute top-0 left-0 w-full h-full object-cover ${darkMode ? 'opacity-90' : 'opacity-100'}`}
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full"
            />
            
            
            {distance && (
              <div className={`absolute bottom-6 left-6 px-4 py-3 rounded-full shadow-lg flex items-center gap-2 ${darkMode ? 'bg-gray-700/90 text-blue-300' : 'bg-white/90 text-blue-600'} animate-bounce`}>
                <Ruler size={20} />
                <span className="font-bold text-lg">{distance} cm</span>
              </div>
            )}

            
            <div className="absolute top-4 right-4 space-y-2">
              {detectedObjects.slice(0, 3).map((obj, i) => {
                const emoji = Object.values(KNOWN_OBJECTS).find(
                  item => item.cocoClass === obj.class
                )?.emoji || '❓';
                
                return (
                  <div 
                    key={i}
                    className={`p-2 rounded-full shadow-lg flex items-center justify-center ${darkMode ? 'bg-gray-700/80' : 'bg-white/80'} animate-float`}
                    style={{ 
                      animationDelay: `${i * 0.2}s`,
                      width: '40px',
                      height: '40px'
                    }}
                  >
                    <span className="text-xl">{emoji}</span>
                  </div>
                );
              })}
            </div>
          </div>

         
          <div className="p-6 space-y-6">
            <div>
              <h2 className={`text-lg font-medium mb-3 ${darkMode ? 'text-blue-300' : 'text-blue-600'}`}>
                Choisissez un objet de référence
              </h2>
              <div className="flex flex-wrap gap-3">
                {Object.entries(KNOWN_OBJECTS).map(([key, { emoji }]) => (
                  <button
                    key={key}
                    onClick={() => setSelectedObject(key)}
                    className={`px-4 py-2 rounded-full transition-all flex items-center gap-2 ${
                      selectedObject === key
                        ? darkMode 
                          ? 'bg-blue-600 text-white shadow-lg scale-105' 
                          : 'bg-blue-500 text-white shadow-lg scale-105'
                        : darkMode 
                          ? 'bg-gray-700 hover:bg-gray-600 text-gray-200' 
                          : 'bg-white hover:bg-gray-100 text-gray-700'
                    }`}
                  >
                    <span className="text-lg">{emoji}</span>
                    {key}
                  </button>
                ))}
              </div>
            </div>

           
            <div className={`p-5 rounded-xl transition-all duration-300 ${
              distance 
                ? darkMode 
                  ? 'bg-blue-900/30 border border-blue-700 scale-[1.02]' 
                  : 'bg-blue-100 border border-blue-200 scale-[1.02]'
                : darkMode 
                  ? 'bg-gray-700/50 border border-gray-600' 
                  : 'bg-white border border-gray-200'
            }`}>
              <div className="flex items-center gap-3">
                <Ruler className={darkMode ? 'text-blue-300' : 'text-blue-500'} />
                <p className="text-lg font-medium">
                  {distance 
                    ? `Distance estimée : ${distance} cm` 
                    : "Pointez la caméra vers un objet pour mesurer"}
                </p>
              </div>
            </div>

      
            {detectedObjects.length > 0 && (
              <div className={`p-4 rounded-xl ${darkMode ? 'bg-gray-700/50' : 'bg-white'}`}>
                <h3 className={`font-medium mb-2 ${darkMode ? 'text-blue-300' : 'text-blue-600'}`}>
                  Objets détectés :
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {detectedObjects.map((obj, index) => {
                    const emoji = Object.values(KNOWN_OBJECTS).find(
                      item => item.cocoClass === obj.class
                    )?.emoji || '❓';
                    
                    return (
                      <div 
                        key={index} 
                        className={`flex items-center gap-2 p-2 rounded-lg ${darkMode ? 'bg-gray-600/50' : 'bg-gray-50'}`}
                      >
                        <span className="text-xl">{emoji}</span>
                        <div>
                          <p className={darkMode ? 'text-white' : 'text-gray-800'}>
                            {obj.class}
                          </p>
                          <p className={`text-sm ${darkMode ? 'text-blue-300' : 'text-blue-500'}`}>
                            Confiance : {Math.round(obj.score * 100)}%
                          </p>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

     
      <div className={`text-center mt-8 pb-4 text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
        Utilisez la caméra pour mesurer des distances en temps réel !
      </div>

      <style jsx global>{`
        @keyframes float {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-8px); }
        }
        .animate-float {
          animation: float 3s ease-in-out infinite;
        }
        .animate-bounce {
          animation: bounce 2s infinite;
        }
        @keyframes bounce {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-10px); }
        }
      `}</style>
    </div>
  );
}

export default App;
*/