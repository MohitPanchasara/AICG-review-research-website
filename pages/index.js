import styles from '@/styles/Home.module.css';
import HeaderBar from '@/components/HeaderBar';
import UploaderPanel from '@/components/UploaderPanel';
import ProcessingStatus from '@/components/ProcessingStatus';
import ResultSection from '@/components/ResultSection';
import DecisionCard from '@/components/DecisionCard';
import FooterBar from '@/components/FooterBar';
import useVideoAnalysis from '@/components/useVideoAnalysis';

export default function Home() {
  const {
    file, videoUrl, status, progress, lastUpdated, isMock, errorMsg,
    // results
    // summary, intuitiveScore, anomalies, personPresent, thresholdScore, finalModelScore, clips, frames,
    thresholdScore, finalModelScore,
    // actions
    onFileChange, handleAnalyze, hardReset,
  } = useVideoAnalysis();

  return (
    <main className={styles.page}>
      <div className={styles.bgGlow} aria-hidden />

      <HeaderBar title="AI vs Real Video Classifier" subtitle="Upload a video → Analyze → Final score" />

      <section className={styles.panelGrid}>
        {/* Left: Uploader */}
        <div className={styles.panel}>
          <h2 className={styles.h2}>1) Upload</h2>
          <UploaderPanel
            file={file}
            videoUrl={videoUrl}
            status={status}
            onFileChange={onFileChange}
            onAnalyze={handleAnalyze}
            onReset={hardReset}
          />

          {status === 'processing' && (
            <ProcessingStatus
              progress={progress}
              lastUpdated={lastUpdated}
              isMock={isMock}
            />
          )}

          {status === 'failed' && errorMsg && (
            <div className={styles.errorBox}>
              <strong>Processing failed.</strong> {errorMsg}
            </div>
          )}
        </div>

        {/* Right: Score only */}
        <div className={styles.panel}>
          <h2 className={styles.h2}>2) Result</h2>

          <ResultSection
            title="Final Model Score"
            updatedAt={lastUpdated}
            show={finalModelScore != null}
          >
            <DecisionCard
              thresholdScore={thresholdScore ?? 0.5}
              finalModelScore={finalModelScore}
              decision={null}  // decision text hidden for now
            />
          </ResultSection>

          {/* --- Temporarily hidden sections (keep for later) ---
          <ResultSection title="Summary (Model 1)" ...>...</ResultSection>
          <ResultSection title="Intuitive Score" ...>...</ResultSection>
          <ResultSection title="Key Anomalies" ...>...</ResultSection>
          <ResultSection title="Person Present" ...>...</ResultSection>
          <ResultSection title="Threshold Score" ...>...</ResultSection>
          <ResultSection title="Sub-clips" ...>...</ResultSection>
          <ResultSection title="Frames" ...>...</ResultSection>
          ------------------------------------------------------ */}
        </div>
      </section>

      <FooterBar mock={isMock} />
    </main>
  );
}
