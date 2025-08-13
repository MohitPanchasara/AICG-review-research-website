import styles from './HeaderBar.module.css';

export default function HeaderBar({ title, subtitle }) {
  return (
    <header className={styles.header}>
      <div className={styles.logoDot} />
      <div>
        <h1 className={styles.title}>{title}</h1>
        <p className={styles.subtitle}>{subtitle}</p>
      </div>
    </header>
  );
}
