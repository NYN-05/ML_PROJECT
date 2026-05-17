import { useState } from 'react';
import { Header } from './Header';
import styles from './MainLayout.module.css';

export function MainLayout({ children }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  return (
    <div className={styles.layout}>
      <Header onMenuToggle={() => setIsSidebarOpen(!isSidebarOpen)} />
      <div className={styles.container}>
        <main className={styles.main}>
          <div className={styles.content}>{children}</div>
        </main>
      </div>
    </div>
  );
}

export default MainLayout;