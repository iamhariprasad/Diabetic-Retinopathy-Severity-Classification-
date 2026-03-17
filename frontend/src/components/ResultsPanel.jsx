import React from 'react';

function ResultsPanel({ results, onReset }) {
  const { prediction, confidence, class_idx, original_image, vessel_mask, refined_mask } = results;

  // Formatting class name for styling
  const statusClass = `status-${prediction.replace(/\s+/g, '-')}`;

  return (
    <div className="glass-panel" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '3rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '2rem' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '2px', marginBottom: '0.5rem' }}>
            Diagnosis Result
          </h2>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '1rem' }}>
            <h1 className={statusClass} style={{ fontSize: '3.5rem', fontWeight: '800', lineHeight: 1 }}>
              {prediction}
            </h1>
            <span style={{ fontSize: '1.25rem', color: 'var(--text-muted)' }}>Class {class_idx}</span>
          </div>
        </div>
        
        <div style={{ textAlign: 'right' }}>
          <p style={{ fontSize: '1rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>Confidence Score</p>
          <p style={{ fontSize: '2.5rem', fontWeight: 'bold', color: 'var(--text-main)' }}>
            {(confidence * 100).toFixed(2)}%
          </p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '2rem', marginBottom: '2rem' }}>
        
        {/* Original Image */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ fontSize: '1.1rem', color: 'var(--text-muted)' }}>Original Image</h3>
          <img 
            src={original_image} 
            alt="Original" 
            style={{ width: '100%', borderRadius: '12px', border: '1px solid var(--glass-border)', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.3)' }} 
          />
        </div>

        {/* Vessel Mask */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ fontSize: '1.1rem', color: 'var(--text-muted)' }}>Frangi Vessel Mask</h3>
          <img 
            src={vessel_mask} 
            alt="Vessels" 
            style={{ width: '100%', borderRadius: '12px', border: '1px solid var(--glass-border)', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.3)' }} 
          />
        </div>

        {/* Refined Mask */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <h3 style={{ fontSize: '1.1rem', color: 'var(--text-muted)' }}>RANSAC Refined Mask</h3>
          <img 
            src={refined_mask} 
            alt="Refined Vessels" 
            style={{ width: '100%', borderRadius: '12px', border: '1px solid var(--glass-border)', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.3)' }} 
          />
        </div>

      </div>

      <div style={{ textAlign: 'center', marginTop: '3rem' }}>
        <button className="btn btn-secondary" onClick={onReset} style={{ padding: '0.75rem 2rem' }}>
          New Analysis
        </button>
      </div>

    </div>
  );
}

export default ResultsPanel;
