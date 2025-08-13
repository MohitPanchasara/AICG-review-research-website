import styles from './FooterBar.module.css';

export default function FooterBar({ mock }) {
  return (
    <footer className={styles.footer}>
      <span>© {new Date().getFullYear()} AICG — Cyberpunk demo</span>
      {mock && <span className={styles.mock}>Mock Mode</span>}
    </footer>
  );
}
