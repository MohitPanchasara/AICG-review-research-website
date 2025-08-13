import styles from './IntuitiveMeter.module.css';

export default function IntuitiveMeter({ value }) {
  if (value == null) return '—';
  const pct = Math.min(100, Math.max(0, value));
  return (
    <>
      <div className={styles.outer} aria-label="Intuitive Score">
        <div className={styles.inner} style={{ width: `${pct}%` }} />
      </div>
      <div className={styles.label}>{pct}%</div>
    </>
  );
}
