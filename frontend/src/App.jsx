import React, { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import ResultsPanel from './components/ResultsPanel';
import './index.css';

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleImageSelect = (selectedFile) => {
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResults(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed. Please ensure the backend is running.');
      }

      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      
      setResults(data);
    } catch (err) {
      console.error(err);
      setError(err.message || 'An error occurred during analysis.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="container">
      <header className="header">
        <h1>Diabetic Retinopathy Analyzer</h1>
        <p>Advanced Deep Learning & Vessel Segmentation for Retinal Diagnostics</p>
      </header>

      <main>
        {!results && !loading && (
          <div className="animate-up">
            <ImageUpload 
              previewUrl={previewUrl} 
              onImageSelect={handleImageSelect} 
            />
            {previewUrl && (
              <div style={{ textAlign: 'center', marginTop: '2rem' }}>
                <button 
                  className="btn" 
                  onClick={handleAnalyze}
                  style={{ fontSize: '1.25rem', padding: '1rem 3rem', borderRadius: '50px' }}
                >
                  Analyze Retina
                </button>
              </div>
            )}
            {error && (
              <div style={{ color: 'var(--danger)', textAlign: 'center', marginTop: '1rem' }}>
                {error}
              </div>
            )}
          </div>
        )}

        {loading && (
          <div className="glass-panel animate-up" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
            <div className="loader">
              <div style={{ width: '60px', height: '60px', border: '5px solid rgba(255,255,255,0.1)', borderTopColor: 'var(--primary)', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 2rem' }}></div>
              <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            </div>
            <h2>Analyzing Retinal Image...</h2>
            <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>Extracting vessels and running ViT classification</p>
          </div>
        )}

        {results && !loading && (
          <div className="animate-up">
            <ResultsPanel results={results} onReset={handleReset} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
