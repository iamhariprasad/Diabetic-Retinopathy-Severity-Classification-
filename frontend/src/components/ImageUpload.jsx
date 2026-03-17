import React, { useCallback } from 'react';

function ImageUpload({ previewUrl, onImageSelect }) {
  
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onImageSelect(e.dataTransfer.files[0]);
    }
  }, [onImageSelect]);

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      onImageSelect(e.target.files[0]);
    }
  };

  return (
    <div 
      className="glass-panel" 
      style={{
        maxWidth: '800px',
        margin: '0 auto',
        textAlign: 'center',
        padding: '3rem',
        borderStyle: previewUrl ? 'solid' : 'dashed',
        borderWidth: '2px',
        borderColor: previewUrl ? 'var(--secondary)' : 'var(--glass-border)',
        position: 'relative',
        overflow: 'hidden'
      }}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
      {previewUrl ? (
        <div style={{ position: 'relative' }}>
          <img 
            src={previewUrl} 
            alt="Preview" 
            style={{ 
              maxWidth: '100%', 
              maxHeight: '400px', 
              borderRadius: '12px',
              boxShadow: '0 10px 25px -5px rgba(0,0,0,0.5)'
            }} 
          />
          <div style={{ marginTop: '1.5rem' }}>
            <label className="btn btn-secondary">
              Choose Another Image
              <input 
                type="file" 
                accept="image/*" 
                onChange={handleChange} 
                style={{ display: 'none' }} 
              />
            </label>
          </div>
        </div>
      ) : (
        <div style={{ padding: '2rem 1rem' }}>
          <div style={{ marginBottom: '1.5rem', color: 'var(--text-muted)' }}>
            <svg style={{ width: '64px', height: '64px', margin: '0 auto' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <h3 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Drag & Drop Retinal Image</h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: '2rem' }}>or click to browse from your computer</p>
          <label className="btn">
            Select File
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleChange} 
              style={{ display: 'none' }} 
            />
          </label>
        </div>
      )}
    </div>
  );
}

export default ImageUpload;
