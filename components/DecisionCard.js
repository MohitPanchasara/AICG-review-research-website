import styles from './DecisionCard.module.css';

export default function DecisionCard({ thresholdScore, finalModelScore, decision }) {
  if (finalModelScore == null) return '—';
  return (
    <div className={styles.card}>
      <div className={styles.score}>{finalModelScore.toFixed(2)}</div>
      <div className={styles.text}>
        {thresholdScore != null ? (
          <>Threshold <b>{thresholdScore.toFixed(2)}</b> → <b>{decision}</b></>
        ) : '—'}
      </div>
    </div>
  );
}
