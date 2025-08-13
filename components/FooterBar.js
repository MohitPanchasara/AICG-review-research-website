import styles from './FooterBar.module.css';

export default function FooterBar({ mock }) {
  return (
    <footer className={styles.footer}>
      <span>© {new Date().getFullYear()} | Made with 🤍 by Team AI Content Detection (Mohit, Bhoomika, Vrunal)</span>
      {mock && <span className={styles.mock}>Mock Mode</span>}
    </footer>
  );
}
