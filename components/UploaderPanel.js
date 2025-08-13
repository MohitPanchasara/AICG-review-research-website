import styles from './UploaderPanel.module.css';
import { track } from '@vercel/analytics';

export default function UploaderPanel({ file, videoUrl, status, onFileChange, onAnalyze, onReset }) {
  return (
    <>
      <label className={styles.dropzone}>
        <input type="file" accept="video/*" onChange={onFileChange} className={styles.hiddenInput} />
        <span className={styles.dropText}>{file ? 'Change video' : 'Click to choose a video (or drag & drop)'}</span>
        <span className={styles.hint}>Best under 100MB for faster analysis</span>
      </label>

      {file && (
        <div className={styles.fileInfo}>
          <div className={styles.fileMeta}>
            <strong>{file.name}</strong>
            <span>{(file.size / (1024 * 1024)).toFixed(1)} MB</span>
          </div>
          <video className={styles.preview} src={videoUrl} controls muted />
        </div>
      )}

      <div className={styles.actions}>
        <button
          className={styles.primaryBtn}
          onClick={() => { track('analyze_clicked'); onAnalyze(); }}
          disabled={!file || status === 'processing'}
        >
          {status === 'processing' ? 'Analyzing…' : 'Analyze Video'}
        </button>
        <button className={styles.secondaryBtn} onClick={onReset} disabled={status === 'processing' && !file}>
          Upload Another Video
        </button>
      </div>
    </>
  );
}
