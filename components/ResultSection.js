import styles from './ResultSection.module.css';

export default function ResultSection({ title, updatedAt, show, children }) {
  return (
    <section className={styles.wrap}>
      <h3 className={styles.h3}>{title}</h3>
      {show ? children : <div className={styles.placeholder}>No data yet</div>}
      {show && updatedAt && <div className={styles.updated}>Last updated {updatedAt}</div>}
    </section>
  );
}
