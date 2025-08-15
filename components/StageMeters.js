import styles from './StageMeters.module.css';

function clamp01(x){ return Math.max(0, Math.min(1, x)); }

/**
 * Maps global progress (0..100) to 3 stage bars:
 * Summary:   10..55  (45 span)
 * Randomness:55..65  (10 span)
 * Scoring:   65..100 (35 span)
 */
export default function StageMeters({ progress=0 }) {
  const p = Math.max(0, Math.min(100, Number(progress||0)));

  const s1 = clamp01((p - 10) / 45);  // 0..1
  const s2 = clamp01((p - 55) / 10);
  const s3 = clamp01((p - 65) / 35);

  const stages = [
    { name: 'Summary',    value: s1 },
    { name: 'Randomness', value: s2 },
    { name: 'Scoring',    value: s3 },
  ];

  return (
    <div className={styles.wrap}>
      {stages.map((st) => (
        <div key={st.name} className={styles.row}>
          <div className={styles.label}>{st.name}</div>
          <div className={styles.track}>
            <div className={styles.fill} style={{ width: `${st.value*100}%` }} />
          </div>
          <div className={styles.pct}>{Math.round(st.value*100)}%</div>
        </div>
      ))}
    </div>
  );
}
