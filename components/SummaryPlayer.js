import { useRef, useState, useMemo, useEffect, useCallback } from 'react';
import styles from './SummaryPlayer.module.css';

export default function SummaryPlayer({ videoUrl, items = [] }) {
  const videoRef = useRef(null);
  const [activeIdx, setActiveIdx] = useState(-1);
  const [caption, setCaption] = useState('');

  const segments = useMemo(
    () => (items || []).map(([s, e, text]) => ({
      s: Number(s || 0),
      e: Number(e || 0),
      text: String(text || '')
    })),
    [items]
  );

  const locate = useCallback((t) => {
    if (!segments.length) return -1;
    for (let i = segments.length - 1; i >= 0; i--) {
      if (t >= segments[i].s) return i;
    }
    return -1;
  }, [segments]);

  const applyTime = useCallback((t) => {
    const idx = locate(t);
    if (idx !== activeIdx) {
      setActiveIdx(idx);
      setCaption(idx >= 0 ? segments[idx].text : '');
    }
  }, [locate, activeIdx, segments]);

  useEffect(() => { setActiveIdx(-1); setCaption(''); }, [segments.length, videoUrl]);

  const onTimeUpdate = () => {
    const el = videoRef.current;
    if (el) applyTime(el.currentTime || 0);
  };

  const onLoadedMetadata = () => {
    const el = videoRef.current;
    if (el) {
      // ensure muted by default
      el.muted = true;
      el.volume = 0;
    }
    applyTime(0);
  };

  if (!videoUrl || segments.length === 0) {
    return <div className={styles.placeholder}>Run analysis to view timeline captions.</div>;
  }

  return (
    <div className={styles.wrap}>
      <video
        ref={videoRef}
        src={videoUrl}
        className={styles.video}
        controls
        preload="metadata"
        muted
        defaultMuted
        playsInline
        onTimeUpdate={onTimeUpdate}
        onSeeked={onTimeUpdate}
        onLoadedMetadata={onLoadedMetadata}
      />
      <div className={styles.caption}>{caption || '—'}</div>
    </div>
  );
}
