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