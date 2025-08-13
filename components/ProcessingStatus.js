import styles from './ProcessingStatus.module.css';

export default function ProcessingStatus({ progress, lastUpdated, isMock }) {
  const pct = Math.max(8, Number(progress || 0));
  return (
    <div className={styles.box} role="status" aria-live="polite">
      <div className={styles.spinner} />
      <div className={styles.row}>
        <div className={styles.barOuter} aria-label="progress">
          <div className={styles.barInner} style={{ width: `${pct}%` }} />
        </div>
        <span className={styles.label}>{pct}%</span>
      </div>
      <div className={styles.text}>Status: processing… {lastUpdated ? `(updated ${lastUpdated})` : ''}</div>
      {isMock && <div className={styles.mock}>Mock Mode (no backend set)</div>}
    </div>
  );
}
