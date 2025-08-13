import { useState } from 'react';
import styles from './FrameGrid.module.css';

export default function FrameGrid({ frames }) {
  const [modal, setModal] = useState({ open: false, url: '', note: '' });

  return (
    <>
      <div className={styles.grid}>
        {frames.map((f, i) => (
          <button key={i} className={styles.btn} onClick={() => setModal({ open: true, url: f.url, note: f.note || '' })}>
            <img src={f.url} alt={f.note || `frame at ${f.t}s`} className={styles.img} />
            <span className={styles.tag}>{f.t?.toFixed ? `${f.t.toFixed(1)}s` : ''}</span>
          </button>
        ))}
      </div>

      {modal.open && (
        <div className={styles.modal} role="dialog" aria-modal="true">
          <div className={styles.inner}>
            <img src={modal.url} alt={modal.note || 'frame'} className={styles.modalImg} />
            {modal.note && <div className={styles.note}>{modal.note}</div>}
            <button className={styles.close} onClick={() => setModal({ open: false, url: '', note: '' })}>Close</button>
          </div>
          <div className={styles.backdrop} onClick={() => setModal({ open: false, url: '', note: '' })} />
        </div>
      )}
    </>
  );
}
