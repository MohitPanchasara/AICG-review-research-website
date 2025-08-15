import styles from './IntuitiveMeter.module.css';

/**
 * value: 0..100 (higher = more random)
 * label: optional string
 */
export default function IntuitiveMeter({ value = 0, label = 'Intuition Score' }) {
  const v = Math.max(0, Math.min(100, Number(value || 0)));
  return (
    <div className={styles.card}>
      <div className={styles.head}>
        <span className={styles.label}>{label}</span>
        <span className={styles.value}>{v.toFixed(1)}</span>
      </div>
      <div className={styles.track}>
        <div className={styles.fill} style={{ width: `${v}%` }} />
      </div>
      <div className={styles.scale}>
        <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
      </div>
      <div className={styles.note}>0 = natural · 100 = random</div>
    </div>
  );
}
