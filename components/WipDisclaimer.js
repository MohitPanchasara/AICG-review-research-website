import { useState } from 'react';
import styles from './WipDisclaimer.module.css';

export default function WipDisclaimer() {
  const [show, setShow] = useState(true);
  if (!show) return null;

  return (
    <div className={styles.wrap} role="note" aria-live="polite">
      <div className={styles.icon}>❗</div>
      <div className={styles.text}>
        <div className={styles.title}>Work-in-progress alert</div>
        <p className={styles.copy}>
          This is our <b>smallest</b> baseline model (DenseNet169) and it’s currently
          <b> overfitting like it’s cramming for finals</b>. The real, multi-model architecture (smarter summaries, anomaly timelines, person checks, the
          whole buffet) is in the oven. In the meantime, <b>upload a clip, and roast our baseline</b>. <br/> Big upgrade landing soon. ✨
        </p>
      </div>
      <button className={styles.close} onClick={() => setShow(false)} aria-label="Dismiss disclaimer">×</button>
    </div>
  );
}
