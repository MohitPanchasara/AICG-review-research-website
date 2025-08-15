import { useState, useMemo } from 'react';
import { jumpTo } from './utils';
import styles from './SummaryTimeline.module.css';

export default function SummaryTimeline({ items = [], onSelect }) {
  const [active, setActive] = useState(0);

  const safeItems = useMemo(() => {
    return (items || []).map(([start, end, text]) => ({
      start: Number(start || 0),
      end: Number(end || 0),
      text: String(text || ''),
    }));
  }, [items]);

  function handleClick(i, s) {
    setActive(i);
    jumpTo(s);
    onSelect?.(safeItems[i]);
  }

  if (!safeItems.length) return <div className={styles.placeholder}>No timeline yet</div>;

  return (
    <div className={styles.wrap}>
      <div className={styles.chips}>
        {safeItems.map((seg, i) => (
          <button
            key={i}
            className={`${styles.chip} ${i===active ? styles.active : ''}`}
            onClick={() => handleClick(i, seg.start)}
            title={`${seg.start.toFixed(1)}–${seg.end.toFixed(1)}s`}
          >
            {seg.start.toFixed(1)}–{seg.end.toFixed(1)}s
          </button>
        ))}
      </div>
      <div className={styles.caption}>
        {safeItems[active]?.text || '—'}
      </div>
    </div>
  );
}
