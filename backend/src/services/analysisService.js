"""Analysis service using local ML models."""

import { sanitizeText } from "../utils/sanitize.js";
import { makeAnalysisId, toConfidenceLabel, toRiskLevel } from "../utils/response.js";
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Model info cache
let modelInfo = null;


function getModelInfo() {
    if (modelInfo) return modelInfo;
    
    try {
        const leaderboardPath = path.join(__dirname, '../../artifacts/models/leaderboard.json');
        if (fs.existsSync(leaderboardPath)) {
            const data = JSON.parse(fs.readFileSync(leaderboardPath, 'utf8'));
            modelInfo = {
                totalModels: data.total_datasets,
                bestModel: data.best_model,
                weights: data.weights.reduce((acc, w) => {
                    acc[w.dataset_name] = w.final_weight;
                    return acc;
                }, {})
            };
        }
    } catch (e) {
        console.error('Error loading model info:', e);
    }
    
    return modelInfo || { totalModels: 0, bestModel: null, weights: {} };
}


async function runAnalysis({ title, text }) {
    const sanitizedTitle = sanitizeText(title);
    const sanitizedText = sanitizeText(text);
    
    const start = performance.now();
    
    try {
        // Use Python inference
        const pythonScript = path.join(__dirname, '../../ml-model/inference_cli.py');
        const fullText = `${sanitizedTitle} ${sanitizedText}`.replace(/"/g, '\\"');
        
        const cmd = `python "${pythonScript}" --text "${fullText}"`;
        const output = execSync(cmd, { 
            encoding: 'utf8',
            timeout: 10000,
            cwd: path.join(__dirname, '../../ml-model')
        });
        
        const result = JSON.parse(output.trim());
        
        const duration = Math.max(0, performance.now() - start);
        
        const payload = {
            success: true,
            prediction: result.prediction,
            confidence: result.confidence,
            confidenceLabel: toConfidenceLabel(result.confidence),
            riskLevel: toRiskLevel(result.prediction),
            processingTime: `${Math.round(duration)}ms`,
            timestamp: new Date().toISOString(),
            analysisId: makeAnalysisId(),
            modelUsed: result.model_used,
            availableModels: result.available_models || [],
            weights: result.weights || {}
        };
        
        const historyEntry = {
            analysisId: payload.analysisId,
            prediction: payload.prediction,
            confidence: payload.confidence,
            timestamp: payload.timestamp,
            title: sanitizedTitle.slice(0, 120),
        };
        
        return { payload, historyEntry };
        
    } catch (error) {
        console.error('ML Inference error:', error.message);
        
        // Return fallback response
        const payload = {
            success: false,
            prediction: 'ERROR',
            confidence: 0,
            confidenceLabel: 'LOW',
            riskLevel: 'UNKNOWN',
            processingTime: '0ms',
            timestamp: new Date().toISOString(),
            analysisId: makeAnalysisId(),
            error: error.message,
            modelInfo: getModelInfo()
        };
        
        const historyEntry = {
            analysisId: payload.analysisId,
            prediction: 'ERROR',
            confidence: 0,
            timestamp: payload.timestamp,
            title: sanitizedTitle.slice(0, 120),
        };
        
        return { payload, historyEntry };
    }
}


export { runAnalysis, getModelInfo };