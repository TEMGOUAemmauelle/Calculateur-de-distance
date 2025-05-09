# 🚀 Magic Distance Meter - Mesure par Vision IA

<div align="center">
  <img src="demo.gif" alt="Démonstration de l'application" width="600"/>
  
  [![React](https://img.shields.io/badge/React-18.2.0-blue)](https://reactjs.org/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow.js-3.18.0-orange)](https://www.tensorflow.org/js)
  [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
</div>

## ✨ Fonctionnalités

### 🎯 Détection Intelligente
- Reconnaissance d'objets en temps réel via webcam
- Prise en charge de multiples objets de référence
- Visualisation des résultats avec cadres colorés

### 📏 Mesure Précise
- Calcul de distance en centimètres
- Estimation basée sur la taille réelle des objets
- Affichage dynamique des résultats

### 🎨 Expérience Utilisateur
- Interface moderne avec animations fluides
- Mode clair/sombre adaptable
- Design responsive (mobile & desktop)
- Effets visuels engageants (confettis, emojis)

## 🛠 Technologies

| Catégorie        | Technologies                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **Framework**    | React 18, TypeScript 4.9                                                     |
| **IA**           | TensorFlow.js, modèle COCO-SSD                                               |
| **Styling**      | Tailwind CSS 3, Framer Motion                                                |
| **UI**           | Lucide Icons, react-confetti                                                 |
| **Build Tool**   | Vite 4                                                                       |

## 📦 Objets Supportés

| Objet            | Emoji | Largeur Réelle | Classe COCO        |
|------------------|-------|----------------|--------------------|
| Visage humain    | 👤    | 15 cm          | person             |
| Téléphone mobile | 📱    | 7 cm           | cell phone         |
| Feuille A4       | 📄    | 21 cm          | book               |
| Carte bancaire   | 💳    | 8.5 cm         | credit card        |

## 🚀 Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/votre-utilisateur/magic-distance-meter.git
cd magic-distance-meter
```

npm install
# ou
yarn install

npm run dev
# ou
yarn dev