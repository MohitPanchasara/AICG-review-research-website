import styles from './ClipGrid.module.css';

export default function ClipGrid({ clips }) {
  return (
    <div className={styles.grid}>
      {clips.map((c, i) => (
        <figure key={i} className={styles.item}>
          <video className={styles.video} src={c.url} controls />
          <figcaption className={styles.caption}>
            {c.caption} • {c.start?.toFixed?.(1)}–{c.end?.toFixed?.(1)}s
          </figcaption>
        </figure>
      ))}
    </div>
  );
}
