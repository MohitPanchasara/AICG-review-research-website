import { jumpTo } from './utils';
import styles from './AnomalyList.module.css';

export default function AnomalyList({ anomalies }) {
  if (!anomalies?.length) return '—';
  return (
    <div className={styles.row}>
      {anomalies.map((a, i) => (
        <div key={i} className={styles.chip}>
          <span className={styles.label}>{a.label}</span>
          <span className={styles.count}>×{a.count}</span>
          {Array.isArray(a.timestamps) && a.timestamps.length > 0 && (
            <span className={styles.times}>
              {a.timestamps.map((t, j) => (
                <button
                  key={j}
                  className={styles.tsBtn}
                  onClick={() => jumpTo(t)}
                  title={`Jump to ${t}s`}
                >
                  {Number(t).toFixed(1)}s
                </button>
              ))}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
