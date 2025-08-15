// components/ResultSection.js
import styles from './ResultSection.module.css';

function formatTs(ts) {
  if (!ts) return null;
  try {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return null;
  }
}

export default function ResultSection({ title, updatedAt, show = true, children }) {
  if (!show) return null;
  const when = formatTs(updatedAt);
  return (
    <section className={styles.section}>
      <div className={styles.header}>
        <h3 className={styles.title}>{title}</h3>
        {when ? <div className={styles.updated}>Last updated {when}</div> : null}
      </div>
      <div className={styles.body}>{children}</div>
    </section>
  );
}
