# FakeScope Frontend

A modern, production-grade React frontend for fake news detection.

## Features

- **Ensemble AI Detection**: Uses 7 ML models with weighted voting
- **Modern UI**: Dark theme with glassmorphism effects
- **Real-time Analysis**: Fast inference with detailed results
- **Model Insights**: View individual model predictions and contributions
- **Responsive**: Works on desktop, tablet, and mobile

## Tech Stack

- React 18 + TypeScript
- Vite for fast builds
- CSS Modules for styling
- Deep Space theme with custom design system

## Getting Started

### Prerequisites

- Node.js 18+
- Backend running on http://localhost:5000

### Install & Run

```bash
cd frontend
npm install
npm run dev
```

### Build for Production

```bash
npm run build
```

### Environment Variables

Create `.env` file:

```
VITE_API_URL=http://localhost:5000/api/v1
```

## Project Structure

```
src/
├── components/
│   ├── ui/          # Reusable UI components
│   ├── layout/      # Layout components
│   └── features/    # Feature-specific components
├── pages/           # Page components
├── hooks/           # Custom React hooks
├── services/        # API services
├── styles/          # Global styles and theme
└── App.tsx          # Root component
```

## API Integration

The frontend connects to the backend at `/api/v1`:
- `POST /api/v1/analyze` - Analyze text for fake news
- `GET /api/v1/history` - Get analysis history

## Design System

- **Fonts**: Outfit (display), DM Sans (body), JetBrains Mono (code)
- **Colors**: Deep space theme with teal/purple accents
- **Components**: Badge, Button, Card, Input, Progress, Spinner